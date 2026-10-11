"""The FastAPI app: ``POST /api/ask``, ``GET /api/health``, ``GET /documents``,
and the operator-only ``/api/monitor/*`` (see ``hoa_qa.web.monitor``).

Run locally with ``uv run uvicorn hoa_qa.web.app:app --reload``. On Vercel,
``api/index.py`` re-exports ``app``.

Privacy: logs carry the request ID, outcome, latency, cost and exception
*type* only. Question and answer text are never logged.
"""

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from hoa_qa import telemetry
from hoa_qa.ask import QASettings
from hoa_qa.budget import (
    BudgetStore,
    CounterStore,
    Reservation,
    check_redis_config,
    is_production,
    resolve_redis_config,
    select_budget_store,
    select_counter_store,
)
from hoa_qa.models import Answer, Outcome, validate_https
from hoa_qa.trace import NO_TRACE
from hoa_qa.web import monitor
from hoa_qa.web.deps import (
    Asker,
    ServiceUnavailable,
    aclose_if_present,
    get_asker,
    get_budget_store,
    get_rate_limiter,
    get_settings,
    load_corpus_once,
)
from hoa_qa.web.ratelimit import RateLimiter, client_ip
from hoa_qa.web.security import SecurityHeadersMiddleware, origin_allowed
from hoa_qa.web.settings import (
    DEFAULT_DOCUMENTS_URL,
    DISCLAIMER,
    REPO_ROOT,
    WebSettings,
    check_sdk_debug_logging,
)

logger = logging.getLogger("hoa_qa.web")

MAX_QUESTION_CHARS = 500
TRACED_PATHS = frozenset({"/api/ask", "/api/monitor/ask"})


def sse(event: dict[str, Any]) -> bytes:
    """One server-sent event. JSON never contains a raw newline, so a
    question or answer cannot break the framing."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


PUBLIC_DIR = REPO_ROOT / "public"

Question = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS
    ),
]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question = Field(description="At most 500 characters.")


def web_answer(outcome: Outcome, text: str) -> Answer:
    """An ``Answer`` produced by the web layer itself (no AI call)."""
    return Answer(
        request_id=uuid.uuid4().hex,
        outcome=outcome,
        answer_text=text,
        citations=(),
        confidence=None,
        conflicts_noted=(),
        disclaimer=DISCLAIMER,
    )


def answer_response(answer: Answer, status_code: int = 200) -> JSONResponse:
    return JSONResponse(answer.model_dump(mode="json"), status_code=status_code)


def budget_exhausted_answer(documents_url: str) -> Answer:
    return web_answer(
        Outcome.budget_exhausted,
        "This month's budget for answering questions has been used up; it "
        "resets on the 1st (UTC). In the meantime, you can read the HOA "
        f"documents directly: {documents_url}",
    )


def configure_logging() -> None:
    """Send ``hoa_qa`` INFO logs to stderr (Vercel's log drain) unless the host
    application has already configured logging."""
    package_logger = logging.getLogger("hoa_qa")
    if package_logger.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


def create_app(
    settings: WebSettings | None = None,
    *,
    env: Mapping[str, str] | None = None,
    budget_store: BudgetStore | None = None,
    counters: CounterStore | None = None,
    public_dir: Path = PUBLIC_DIR,
) -> FastAPI:
    """Build the app. Stores default to what the environment selects."""
    env = os.environ if env is None else env
    configure_logging()
    check_sdk_debug_logging(env)
    check_redis_config(env, log=logger)
    telemetry.configure(env)
    config = settings or WebSettings.from_env(env)
    counter_store = counters or select_counter_store(env)
    budget = budget_store or select_budget_store(env, counter_store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # R here is the env default; the asker's own bound is checked again
        # when the first request builds it (see ``_check_budget_once``).
        warn_if_budget_below_reservation(config, None)
        yield
        await telemetry.flush_async()
        await aclose_if_present(getattr(app.state, "asker", None))
        stores = (counter_store, getattr(budget, "counters", None))
        for store in {id(store): store for store in stores}.values():
            await aclose_if_present(store)

    app = FastAPI(
        title="Lakewood Creek HOA Q&A",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = config
    app.state.env = env
    app.state.budget_checked = False
    app.state.budget_store = budget
    app.state.rate_limiter = RateLimiter(
        counter_store, config.rate_limit_per_hour, config.rate_limit_per_day
    )
    app.state.asker = None
    # The operator monitor: off (and recording nothing) unless a token is set.
    app.state.monitor_token = monitor.monitor_token(env)
    app.state.recent_store = counter_store
    app.state.monitor = None
    app.add_middleware(SecurityHeadersMiddleware)

    @app.middleware("http")
    async def flush_traces(request: Request, call_next: Any) -> Any:
        """Send a question's trace before its response goes out.

        A serverless function is frozen once it has responded, so spans left
        in the queue could be lost or arrive with the next request. Costs one
        short HTTP call per answer, and nothing when tracing is off.
        """
        response = await call_next(request)
        if request.url.path in TRACED_PATHS:
            await telemetry.flush_async()
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, exc: RequestValidationError) -> Any:
        # Documented choice: HTTP 422 with an Answer-shaped body, so the page
        # renders every response the same way. The input is not echoed back.
        return answer_response(
            web_answer(
                Outcome.invalid_input,
                f"Please enter a question of 1 to {MAX_QUESTION_CHARS} characters.",
            ),
            status_code=422,
        )

    @app.exception_handler(ServiceUnavailable)
    async def unavailable(request: Request, exc: ServiceUnavailable) -> Any:
        return answer_response(web_answer(Outcome.error, str(exc)), status_code=503)

    async def admit(
        request: Request,
        settings: WebSettings,
        budget_store: BudgetStore,
        limiter: RateLimiter,
        asker: Asker,
    ) -> tuple[JSONResponse, Reservation | None, str]:
        """Origin, rate limit and budget, in that order, for one question.

        Returns ``(refusal, None, id)`` when the question may not run, and
        ``(unused, reservation, id)`` when it may.
        """
        request_id = uuid.uuid4().hex
        if not origin_allowed(request, on_vercel=settings.on_vercel):
            refusal = answer_response(
                web_answer(Outcome.error, "Cross-origin requests aren't allowed."),
                status_code=403,
            )
            return refusal, None, request_id
        ip = client_ip(request, on_vercel=settings.on_vercel)
        if not await limiter.hit(ip):
            logger.info("ask outcome=rate_limited")
            refusal = answer_response(
                web_answer(
                    Outcome.error,
                    "You're asking questions faster than this service allows. "
                    "Please slow down and try again later.",
                ),
                status_code=429,
            )
            return refusal, None, request_id

        # Hold the worst-case cost before spending anything. Any failure to
        # reserve (over budget, or the store erroring) refuses the question.
        reserve_usd = reservation_amount(settings, asker)
        if not request.app.state.budget_checked:
            request.app.state.budget_checked = True
            warn_if_budget_below_reservation(settings, asker)
        try:
            reservation = await budget_store.reserve(
                reserve_usd, settings.monthly_budget_usd
            )
        except Exception as exc:
            logger.error(
                "budget reservation failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            reservation = None
        answer = budget_exhausted_answer(settings.documents_url)
        if reservation is None:
            logger.info(
                "ask outcome=%s request_id=%s", answer.outcome, answer.request_id
            )
        return answer_response(answer), reservation, request_id

    @app.post("/api/ask/stream")
    async def ask_stream(
        body: AskRequest,
        request: Request,
        settings: Annotated[WebSettings, Depends(get_settings)],
        budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
        limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
        asker: Annotated[Asker, Depends(get_asker)],
    ) -> Any:
        """``/api/ask`` as server-sent events, so the page can show progress.

        Same admission checks, budget and logging. Events: ``stage`` as each
        stage starts, ``claim`` for each sentence of the answer once it has
        passed every check, ``reset`` if a draft is withdrawn, and finally
        ``answer`` with the same body ``/api/ask`` returns. The ``answer``
        event is the answer; everything before it is a preview. A question
        that may not run gets the ordinary JSON refusal, not a stream.
        """
        refusal, reservation, request_id = await admit(
            request, settings, budget_store, limiter, asker
        )
        if reservation is None:
            return refusal
        traced = getattr(asker, "ask_traced", None)

        async def events() -> AsyncIterator[bytes]:
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def run() -> Any:
                try:
                    if traced is None:
                        return await asker(body.question)
                    return await traced(body.question, NO_TRACE, emit=queue.put_nowait)
                finally:
                    queue.put_nowait(None)

            started = time.perf_counter()
            task = asyncio.ensure_future(run())
            result: Any = None
            reported: object = None
            try:
                while (event := await queue.get()) is not None:
                    yield sse(event)
                result = await task
                reported = result.estimated_cost_usd
                answer = result.answer
            except Exception as exc:
                # Exception messages can quote the question; log the type only.
                logger.error(
                    "asker failed request_id=%s error_type=%s",
                    request_id,
                    type(exc).__name__,
                )
                reported = getattr(exc, "estimated_cost_usd", None)
                answer = web_answer(
                    Outcome.error,
                    "Something went wrong while answering. Please try again later.",
                ).model_copy(update={"request_id": request_id})
            finally:
                # Also reached when the browser goes away mid-answer: stop the
                # work and settle the budget (the full reservation stays
                # counted if no cost was reported).
                task.cancel()
                cost = await _reconcile(budget_store, reservation, reported, request_id)
            yield sse({"type": "answer", "answer": answer.model_dump(mode="json")})
            # The answer is out; what follows costs the reader no time.
            latency_ms = round((time.perf_counter() - started) * 1000)
            logger.info(
                "ask outcome=%s request_id=%s latency_ms=%d cost_usd=%.6f streamed",
                answer.outcome,
                answer.request_id,
                latency_ms,
                cost,
            )
            if result is not None and request.app.state.monitor_token is not None:
                await monitor.record(
                    request.app.state.recent_store,
                    monitor.feed_row(result, latency_ms, cost, operator=False),
                )
            await telemetry.flush_async()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            # Proxies must pass each event on as it is written.
            headers={"X-Accel-Buffering": "no"},
        )

    @app.post("/api/ask")
    async def ask(
        body: AskRequest,
        request: Request,
        settings: Annotated[WebSettings, Depends(get_settings)],
        budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
        limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
        asker: Annotated[Asker, Depends(get_asker)],
    ) -> JSONResponse:
        refusal, reservation, request_id = await admit(
            request, settings, budget_store, limiter, asker
        )
        if reservation is None:
            return refusal

        started = time.perf_counter()
        try:
            result = await asker(body.question)
            answer = result.answer
        except Exception as exc:
            # Exception messages can quote the question; log the type only.
            logger.error(
                "asker failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            # An asker may attach the cost it incurred before failing.
            await _reconcile(
                budget_store,
                reservation,
                getattr(exc, "estimated_cost_usd", None),
                request_id,
            )
            return answer_response(
                web_answer(
                    Outcome.error,
                    "Something went wrong while answering. Please try again later.",
                ).model_copy(update={"request_id": request_id}),
                status_code=500,
            )

        latency_ms = round((time.perf_counter() - started) * 1000)
        cost = await _reconcile(
            budget_store, reservation, result.estimated_cost_usd, answer.request_id
        )
        logger.info(
            "ask outcome=%s request_id=%s latency_ms=%d cost_usd=%.6f",
            answer.outcome,
            answer.request_id,
            latency_ms,
            cost,
        )
        if request.app.state.monitor_token is not None:
            await monitor.record(
                request.app.state.recent_store,
                monitor.feed_row(result, latency_ms, cost, operator=False),
            )
        return answer_response(answer)

    def operator_only(request: Request) -> None:
        """404, exactly like a route that doesn't exist, without the token."""
        if not monitor.authorized(request, request.app.state.monitor_token):
            raise HTTPException(status_code=404)

    def get_monitor(request: Request, settings: WebSettings) -> monitor.Monitor:
        state = request.app.state
        if state.monitor is None:
            state.monitor = monitor.Monitor(
                load_corpus_once(settings.corpus_path), QASettings.from_env(state.env)
            )
        return state.monitor

    @app.get("/api/monitor/config", dependencies=[Depends(operator_only)])
    async def monitor_config(
        request: Request,
        settings: Annotated[WebSettings, Depends(get_settings)],
    ) -> JSONResponse:
        return JSONResponse(get_monitor(request, settings).config())

    @app.get("/api/monitor/recent", dependencies=[Depends(operator_only)])
    async def monitor_recent(request: Request) -> JSONResponse:
        try:
            return JSONResponse(await monitor.recent(request.app.state.recent_store))
        except Exception as exc:
            logger.error("monitor feed read failed error_type=%s", type(exc).__name__)
            return JSONResponse({"error": "The traffic feed can't be read right now."})

    @app.post("/api/monitor/ask", dependencies=[Depends(operator_only)])
    async def monitor_ask(
        body: AskRequest,
        request: Request,
        settings: Annotated[WebSettings, Depends(get_settings)],
        budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
        asker: Annotated[Asker, Depends(get_asker)],
    ) -> JSONResponse:
        """The operator's own question, with the full trace of what happened.

        No per-IP rate limit (the token is the gate), but the budget still
        applies: this spends real money. The trace is returned, never stored.
        """
        if not origin_allowed(request, on_vercel=settings.on_vercel):
            raise HTTPException(status_code=404)
        traced = getattr(asker, "ask_traced", None)
        if traced is None:
            return JSONResponse({"error": "This asker can't be traced."})
        request_id = uuid.uuid4().hex
        try:
            reservation = await budget_store.reserve(
                reservation_amount(settings, asker), settings.monthly_budget_usd
            )
        except Exception as exc:
            logger.error(
                "budget reservation failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            reservation = None
        if reservation is None:
            return JSONResponse({"error": "This month's budget is used up."})
        seen = monitor.MonitorRecorder()
        started = time.perf_counter()
        try:
            # Lets the operator's own testing be filtered out of the traffic.
            with telemetry.tagged("monitor"):
                result = await traced(body.question, seen)
        except Exception as exc:
            logger.error(
                "asker failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            await _reconcile(
                budget_store,
                reservation,
                getattr(exc, "estimated_cost_usd", None),
                request_id,
            )
            return JSONResponse(
                {"error": f"The pipeline failed ({type(exc).__name__})."},
                status_code=500,
            )
        latency_ms = round((time.perf_counter() - started) * 1000)
        cost = await _reconcile(
            budget_store, reservation, result.estimated_cost_usd, request_id
        )
        logger.info(
            "monitor ask outcome=%s request_id=%s latency_ms=%d cost_usd=%.6f",
            result.answer.outcome,
            request_id,
            latency_ms,
            cost,
        )
        await monitor.record(
            request.app.state.recent_store,
            monitor.feed_row(result, latency_ms, cost, operator=True),
        )
        return JSONResponse(
            get_monitor(request, settings).trace(body.question, result, seen)
        )

    @app.get("/api/health")
    async def health(
        request: Request,
        settings: Annotated[WebSettings, Depends(get_settings)],
    ) -> JSONResponse:
        try:
            corpus = load_corpus_once(settings.corpus_path)
        except (ServiceUnavailable, ValidationError, OSError) as exc:
            logger.error("corpus unavailable error_type=%s", type(exc).__name__)
            return JSONResponse(
                {"status": "unavailable", "documents_url": settings.documents_url},
                status_code=503,
            )
        app_env = getattr(request.app.state, "env", None)
        return JSONResponse(
            {
                "status": "ok",
                "corpus_build_time": corpus.manifest.build_time.isoformat(),
                "chunk_count": corpus.manifest.chunk_count,
                "documents_url": settings.documents_url,
                # Uses the asker's max_cost_usd once a request has built it.
                "budget_config": budget_config(
                    settings,
                    getattr(request.app.state, "asker", None),
                    env=app_env,
                ),
                "budget_config_missing": budget_config_missing(app_env),
            }
        )

    @app.get("/documents", include_in_schema=False)
    async def documents(
        settings: Annotated[WebSettings, Depends(get_settings)],
    ) -> RedirectResponse:
        # A same-origin link to the HOA documents that works without JS. The
        # budget and rate limit are /api/ask dependencies, so they never apply.
        return RedirectResponse(safe_documents_url(settings.documents_url), 307)

    # On Vercel, public/ is served by the CDN (Vercel says not to mount it).
    # Locally, serve it from the app so one uvicorn process runs everything.
    if not config.on_vercel and public_dir.is_dir():
        app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")

    return app


def safe_documents_url(url: str) -> str:
    """The configured documents URL if it's https without userinfo, else the default."""
    try:
        return validate_https(url)
    except ValueError:
        logger.error("documents_url invalid; redirecting to the default")
        return DEFAULT_DOCUMENTS_URL


def reservation_amount(settings: WebSettings, asker: object) -> float:
    """R = max(BUDGET_RESERVE_PER_REQUEST_USD, the asker's ``max_cost_usd``)."""
    reserve = settings.budget_reserve_per_request_usd
    declared = getattr(asker, "max_cost_usd", None)
    if declared is None:
        return reserve
    try:
        declared = float(declared)
    except (TypeError, ValueError):
        declared = math.nan
    if not math.isfinite(declared) or declared < 0:
        logger.warning("ignoring invalid asker max_cost_usd")
        return reserve
    return max(reserve, declared)


BudgetConfig = Literal[
    "ok",
    "budget_below_reservation",
    "redis_config_incomplete",
    "redis_not_configured",
]


def budget_config(
    settings: WebSettings,
    asker: object,
    env: Mapping[str, str] | None = None,
) -> BudgetConfig:
    """Check budget and Redis configuration status for /api/health.

    In production, reports a specific reason for an incomplete Redis pair
    (``redis_config_incomplete``) or missing Redis (``redis_not_configured``).
    When configured, reports ``budget_below_reservation`` if
    MONTHLY_BUDGET_USD < R, or ``ok``.
    """
    env_map = os.environ if env is None else env
    if is_production(env_map):
        redis = resolve_redis_config(env_map)
        if redis.incomplete:
            return "redis_config_incomplete"
        if redis.credentials is None:
            return "redis_not_configured"
    if settings.monthly_budget_usd < reservation_amount(settings, asker):
        return "budget_below_reservation"
    return "ok"


def budget_config_missing(
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return missing Redis variable names if Redis is incomplete in production."""
    env_map = os.environ if env is None else env
    if is_production(env_map):
        redis = resolve_redis_config(env_map)
        if redis.incomplete:
            return list(redis.missing_variables)
    return []


def warn_if_budget_below_reservation(settings: WebSettings, asker: object) -> None:
    if settings.monthly_budget_usd < reservation_amount(settings, asker):
        logger.warning(
            "MONTHLY_BUDGET_USD=%.6f is below the per-request reservation "
            "R=%.6f; every question will get budget_exhausted",
            settings.monthly_budget_usd,
            reservation_amount(settings, asker),
        )


async def _reconcile(
    budget: BudgetStore,
    reservation: Reservation,
    reported_cost: object,
    request_id: str,
) -> float:
    """Swap the reservation for the reported cost; return the cost now counted.

    With no usable cost, or if the store write fails, the full reservation
    stays counted: accounting errors over-count, never under-count. There is
    no retry, so a failing store can't hold the request open.
    """
    held = reservation.amount_usd
    try:
        cost = float(reported_cost)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        cost = math.nan
    if not math.isfinite(cost) or cost < 0:
        logger.error("no usable cost; keeping reservation request_id=%s", request_id)
        return held
    if cost > held:
        logger.warning(
            "cost exceeded reservation request_id=%s excess_usd=%.6f",
            request_id,
            cost - held,
        )
    try:
        await budget.reconcile(reservation, cost)
    except Exception as exc:
        logger.error(
            "reconcile failed; keeping reservation request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
        )
        return held
    return cost


app = create_app()

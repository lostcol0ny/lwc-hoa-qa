"""The FastAPI app: ``POST /api/ask`` and ``GET /api/health``.

Run locally with ``uv run uvicorn hoa_qa.web.app:app --reload``. On Vercel,
``api/index.py`` re-exports ``app``.

Privacy: logs carry the request ID, outcome, latency, cost and exception
*type* only. Question and answer text are never logged.
"""

import logging
import math
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from hoa_qa.budget import (
    BudgetStore,
    CounterStore,
    Reservation,
    select_budget_store,
    select_counter_store,
)
from hoa_qa.models import Answer, Outcome
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
    DISCLAIMER,
    REPO_ROOT,
    WebSettings,
    check_sdk_debug_logging,
)

logger = logging.getLogger("hoa_qa.web")

MAX_QUESTION_CHARS = 500
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
    config = settings or WebSettings.from_env(env)
    counter_store = counters or select_counter_store(env)
    budget = budget_store or select_budget_store(env, counter_store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
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
    app.state.budget_store = budget
    app.state.rate_limiter = RateLimiter(
        counter_store, config.rate_limit_per_hour, config.rate_limit_per_day
    )
    app.state.asker = None
    app.add_middleware(SecurityHeadersMiddleware)

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

    @app.post("/api/ask")
    async def ask(
        body: AskRequest,
        request: Request,
        settings: Annotated[WebSettings, Depends(get_settings)],
        budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
        limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
        asker: Annotated[Asker, Depends(get_asker)],
    ) -> JSONResponse:
        if not origin_allowed(request, on_vercel=settings.on_vercel):
            return answer_response(
                web_answer(Outcome.error, "Cross-origin requests aren't allowed."),
                status_code=403,
            )
        ip = client_ip(request, on_vercel=settings.on_vercel)
        if not await limiter.hit(ip):
            logger.info("ask outcome=rate_limited")
            return answer_response(
                web_answer(
                    Outcome.error,
                    "You're asking questions faster than this service allows. "
                    "Please slow down and try again later.",
                ),
                status_code=429,
            )

        # Hold the worst-case cost before spending anything. Any failure to
        # reserve (over budget, or the store erroring) refuses the question.
        request_id = uuid.uuid4().hex
        reserve_usd = reservation_amount(settings, asker)
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
        if reservation is None:
            answer = budget_exhausted_answer(settings.documents_url)
            logger.info(
                "ask outcome=%s request_id=%s", answer.outcome, answer.request_id
            )
            return answer_response(answer)

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
        return answer_response(answer)

    @app.get("/api/health")
    async def health(
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
        return JSONResponse(
            {
                "status": "ok",
                "corpus_build_time": corpus.manifest.build_time.isoformat(),
                "chunk_count": corpus.manifest.chunk_count,
                "documents_url": settings.documents_url,
            }
        )

    # On Vercel, public/ is served by the CDN (Vercel says not to mount it).
    # Locally, serve it from the app so one uvicorn process runs everything.
    if not config.on_vercel and public_dir.is_dir():
        app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")

    return app


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

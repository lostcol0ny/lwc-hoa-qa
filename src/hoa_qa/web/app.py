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
from hoa_qa.web.security import SecurityHeadersMiddleware, is_cross_origin
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
        if is_cross_origin(request.headers.get("origin"), request.headers.get("host")):
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

        try:
            spend = await budget_store.get_month_spend()
        except Exception as exc:
            # Fail closed: without a trustworthy counter there's no cap.
            logger.error("budget store unavailable error_type=%s", type(exc).__name__)
            spend = math.inf
        if spend >= settings.monthly_budget_usd:
            answer = budget_exhausted_answer(settings.documents_url)
            logger.info(
                "ask outcome=%s request_id=%s", answer.outcome, answer.request_id
            )
            return answer_response(answer)

        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        try:
            result = await asker(body.question)
            answer = result.answer
            cost = float(result.estimated_cost_usd)
        except Exception as exc:
            # Exception messages can quote the question; log the type only.
            logger.error(
                "asker failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            return answer_response(
                web_answer(
                    Outcome.error,
                    "Something went wrong while answering. Please try again later.",
                ).model_copy(update={"request_id": request_id}),
                status_code=500,
            )

        latency_ms = round((time.perf_counter() - started) * 1000)
        await _record_spend(budget_store, cost, answer.request_id)
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


async def _record_spend(budget: BudgetStore, cost: float, request_id: str) -> None:
    if not math.isfinite(cost) or cost < 0:
        logger.error("invalid estimated cost request_id=%s", request_id)
        return
    try:
        await budget.add_spend(cost)
    except Exception as exc:
        logger.error(
            "failed to record spend request_id=%s error_type=%s",
            request_id,
            type(exc).__name__,
        )


app = create_app()

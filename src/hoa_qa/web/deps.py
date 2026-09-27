"""FastAPI dependencies: the corpus, the asker, and the per-app stores.

``Asker``/``AskResult`` are structural protocols matching the QA core's
contract, so the web layer never imports the QA core's types. The real asker
is built lazily, once per process, and only if ``hoa_qa.ask`` exists.
"""

import importlib
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from fastapi import Request

from hoa_qa import models
from hoa_qa.budget import BudgetStore
from hoa_qa.web.ratelimit import RateLimiter
from hoa_qa.web.settings import WebSettings

logger = logging.getLogger("hoa_qa.web")


class AskResult(Protocol):
    answer: models.Answer
    estimated_cost_usd: float


class Asker(Protocol):
    """Answers one question.

    An asker may also expose ``max_cost_usd: float``, its worst-case cost per
    call. It's optional (read with ``getattr``), so it isn't a protocol member;
    when present and larger than ``BUDGET_RESERVE_PER_REQUEST_USD`` it sets the
    budget reservation.
    """

    async def __call__(self, question: str) -> AskResult: ...


class ServiceUnavailable(Exception):
    """The Q&A backend can't serve requests; the message is safe to show users."""


@lru_cache(maxsize=4)
def _load_corpus_cached(path: Path) -> models.Corpus:
    return models.load_corpus(path)


def load_corpus_once(path: Path) -> models.Corpus:
    """Load and validate the corpus once per process (per path)."""
    try:
        return _load_corpus_cached(path.resolve())
    except FileNotFoundError as exc:
        raise ServiceUnavailable("The document index isn't available yet.") from exc


def get_settings(request: Request) -> WebSettings:
    return request.app.state.settings


def get_budget_store(request: Request) -> BudgetStore:
    return request.app.state.budget_store


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


async def get_asker(request: Request) -> Asker:
    """Return this app's asker, building it on first use.

    ``build_asker`` is synchronous, so there's no await between the check and
    the assignment and concurrent requests can't build two askers.
    """
    state = request.app.state
    if getattr(state, "asker", None) is None:
        state.asker = build_asker(state.settings)
    return state.asker


def build_asker(settings: WebSettings) -> Asker:
    corpus = load_corpus_once(settings.corpus_path)
    if settings.fake_asker:
        from hoa_qa.web.dev import FakeAsker

        logger.warning("HOA_QA_FAKE_ASKER=1: serving canned answers (dev only)")
        return FakeAsker(corpus)
    try:
        ask_module = importlib.import_module("hoa_qa.ask")
    except ModuleNotFoundError as exc:
        if exc.name != "hoa_qa.ask":
            raise
        logger.error("hoa_qa.ask is not installed; /api/ask is unavailable")
        raise ServiceUnavailable("The Q&A service isn't set up yet.") from exc
    build = getattr(ask_module, "build_asker", None)
    settings_cls = getattr(ask_module, "QASettings", None)
    if build is None or settings_cls is None:
        logger.error("hoa_qa.ask lacks build_asker/QASettings")
        raise ServiceUnavailable("The Q&A service isn't set up yet.")
    try:
        from_env = getattr(settings_cls, "from_env", None)
        qa_settings = from_env() if callable(from_env) else settings_cls()
        return build(corpus, qa_settings)
    except Exception as exc:
        # e.g. a missing API key. Log the type only; messages may hold config.
        logger.error("building the asker failed error_type=%s", type(exc).__name__)
        raise ServiceUnavailable("The Q&A service isn't set up yet.") from exc


async def aclose_if_present(obj: object) -> None:
    """Await ``obj.aclose()`` if it exists (the QA core asker and Upstash have one)."""
    aclose = getattr(obj, "aclose", None)
    if callable(aclose):
        result = aclose()
        if inspect.isawaitable(result):
            await result

"""FastAPI dependencies: the corpus, the asker, and the per-app stores.

``Asker``/``AskResult`` are structural protocols: the web layer depends only on
the members it reads, so test doubles and the dev ``FakeAsker`` fit too. Their
members are read-only properties, which the QA core's frozen ``AskResult``
satisfies; ``build_asker`` below assigns the QA core's asker to ``Asker``, so
pyright proves the two units agree. The real asker is built lazily, once per
app.
"""

import inspect
import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from fastapi import Request

from hoa_qa import models
from hoa_qa.ask import QASettings
from hoa_qa.ask import build_asker as build_qa_asker
from hoa_qa.budget import BudgetStore
from hoa_qa.web.ratelimit import RateLimiter
from hoa_qa.web.settings import WebSettings

logger = logging.getLogger("hoa_qa.web")


class AskResult(Protocol):
    @property
    def answer(self) -> models.Answer: ...

    @property
    def estimated_cost_usd(self) -> float: ...


class Asker(Protocol):
    """Answers one question.

    An asker may also expose ``max_cost_usd: float``, its worst-case cost per
    call. It's optional (read with ``getattr``), so it isn't a protocol member;
    when present and larger than ``BUDGET_RESERVE_PER_REQUEST_USD`` it sets the
    budget reservation. The QA core's ``QAAsker`` always has it.
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
        state.asker = build_asker(state.settings, state.env)
    return state.asker


def build_asker(settings: WebSettings, env: Mapping[str, str] | None = None) -> Asker:
    """The QA core's asker over the configured corpus (or the dev fake).

    ``env`` supplies the QA settings (API keys, thresholds); ``None`` means
    ``os.environ``. Missing keys or bad settings raise ``ServiceUnavailable``.
    """
    corpus = load_corpus_once(settings.corpus_path)
    # ``WebSettings`` already refuses the flag in production; checking again
    # here means a hand-built settings object can't enable it there either.
    if settings.fake_asker and not settings.production:
        from hoa_qa.web.dev import FakeAsker

        logger.warning("HOA_QA_FAKE_ASKER=1: serving canned answers (dev only)")
        return FakeAsker(corpus)
    try:
        qa_settings = QASettings.from_env(env)
        # Typed assignment: pyright checks QAAsker against the Asker protocol.
        asker: Asker = build_qa_asker(corpus, qa_settings)
    except Exception as exc:
        # e.g. a missing API key. Log the type only; messages may hold config.
        logger.error("building the asker failed error_type=%s", type(exc).__name__)
        raise ServiceUnavailable("The Q&A service isn't set up yet.") from exc
    return asker


async def aclose_if_present(obj: object) -> None:
    """Await ``obj.aclose()`` if it exists (the QA core asker and Upstash have one)."""
    aclose = getattr(obj, "aclose", None)
    if callable(aclose):
        result = aclose()
        if inspect.isawaitable(result):
            await result

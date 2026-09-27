"""End-to-end question answering (spec §4): validate → gate → sweep → answer → verify.

``build_asker(corpus, settings)`` returns an async callable satisfying the web
unit's ``Asker`` protocol. Providers are injectable so tests use fakes.
"""

import logging
import os
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from hoa_qa.answer.pricing import answer_cost_usd, jev_cost_usd
from hoa_qa.answer.prompt import build_prompt
from hoa_qa.answer.provider import (
    DEFAULT_ANSWER_MODEL,
    AnswerProvider,
    AnthropicAnswerProvider,
)
from hoa_qa.models import Answer, Citation, Corpus, Outcome, citation_url
from hoa_qa.retrieval.gate import run_gate
from hoa_qa.retrieval.jev import JevClient, TokenCounter, TypeSafeJevClient
from hoa_qa.retrieval.jev import estimate_tokens as default_token_counter
from hoa_qa.retrieval.sweep import sweep
from hoa_qa.verify.quotes import CheckedCitation, check_quotes
from hoa_qa.verify.support import check_support

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
DEFAULT_DOCUMENTS_URL = "https://lakewoodcreekhoa.com/"

DISCLAIMER = (
    "Unofficial tool, not legal advice; the governing documents and the Board "
    "are authoritative. Don't include personal information; questions are "
    "processed by third-party AI services."
)


class QASettings(BaseModel):
    """Runtime configuration; ``from_env`` reads the documented variables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    typesafe_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    answer_model: str = DEFAULT_ANSWER_MODEL
    jev_model: str = "jev-latest"
    gate_threshold: float = Field(default=0.5, ge=0, le=1)
    sweep_threshold: float = Field(default=0.3, ge=0, le=1)
    sweep_top_k: int = Field(default=8, ge=1)
    sweep_concurrency: int = Field(default=4, ge=1)
    support_threshold: float = Field(default=0.5, ge=0, le=1)
    documents_url: str = DEFAULT_DOCUMENTS_URL

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        env = os.environ if environ is None else environ
        names = {
            "typesafe_api_key": "TYPESAFE_API_KEY",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "answer_model": "ANSWER_MODEL",
            "jev_model": "JEV_MODEL",
            "gate_threshold": "GATE_THRESHOLD",
            "sweep_threshold": "SWEEP_THRESHOLD",
            "sweep_top_k": "SWEEP_TOP_K",
            "sweep_concurrency": "SWEEP_CONCURRENCY",
            "support_threshold": "SUPPORT_THRESHOLD",
            "documents_url": "HOA_DOCUMENTS_URL",
        }
        # Empty values (as in .env.example) mean "use the default".
        values = {
            field_name: env[var].strip()
            for field_name, var in names.items()
            if env.get(var, "").strip()
        }
        return cls.model_validate(values)


class AskResult(BaseModel):
    """What one ``ask`` returns: the public answer plus internal accounting."""

    model_config = ConfigDict(frozen=True)

    answer: Answer
    estimated_cost_usd: float
    jev_input_tokens: int
    answer_input_tokens: int
    answer_output_tokens: int
    latency_ms: float


class Asker(Protocol):
    async def __call__(self, question: str) -> AskResult: ...


def clean_question(question: str) -> str | None:
    """Strip control/format characters; None if empty or over the length cap."""
    chars = []
    for ch in question:
        category = unicodedata.category(ch)
        if category == "Cc":
            chars.append(" " if ch in "\t\n\r\v\f" else "")
        elif category == "Cf":
            continue
        else:
            chars.append(ch)
    cleaned = " ".join("".join(chars).split())
    if not cleaned or len(cleaned) > MAX_QUESTION_CHARS:
        return None
    return cleaned


@dataclass
class _Usage:
    jev_input_tokens: int = 0
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0
    answer_calls: int = 0
    model: str = ""
    notes: list[str] = field(default_factory=list)

    def cost(self) -> float:
        return jev_cost_usd(self.jev_input_tokens) + answer_cost_usd(
            self.model, self.answer_input_tokens, self.answer_output_tokens
        )


class QAAsker:
    """The concrete ``Asker``. Safe to reuse across requests."""

    def __init__(
        self,
        corpus: Corpus,
        settings: QASettings,
        jev: JevClient,
        provider: AnswerProvider,
        count_tokens: TokenCounter = default_token_counter,
    ) -> None:
        self._corpus = corpus
        self._settings = settings
        self._jev = jev
        self._provider = provider
        self._count_tokens = count_tokens

    async def __call__(self, question: str) -> AskResult:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        usage = _Usage(model=self._provider.model)
        try:
            outcome, text, citations, confidence, conflicts = await self._run(
                question, usage
            )
        except Exception as exc:  # any provider failure becomes an error outcome
            # Log only the exception type: SDK messages can echo request bodies.
            logger.error(
                "ask failed request_id=%s error_type=%s",
                request_id,
                type(exc).__name__,
            )
            outcome, text, citations, confidence, conflicts = (
                Outcome.error,
                "Something went wrong answering this question. Please try again "
                f"later, or read the HOA documents at {self._settings.documents_url}",
                (),
                None,
                (),
            )
        answer = Answer(
            request_id=request_id,
            outcome=outcome,
            answer_text=text,
            citations=citations,
            confidence=confidence,
            conflicts_noted=conflicts,
            disclaimer=DISCLAIMER,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        result = AskResult(
            answer=answer,
            estimated_cost_usd=usage.cost(),
            jev_input_tokens=usage.jev_input_tokens,
            answer_input_tokens=usage.answer_input_tokens,
            answer_output_tokens=usage.answer_output_tokens,
            latency_ms=latency_ms,
        )
        _log_result(result, usage)
        return result

    async def _run(
        self, question: str, usage: _Usage
    ) -> tuple[Outcome, str, tuple[Citation, ...], float | None, tuple[str, ...]]:
        s = self._settings
        cleaned = clean_question(question)
        if cleaned is None:
            return (
                Outcome.invalid_input,
                f"Please enter a question between 1 and {MAX_QUESTION_CHARS} "
                "characters.",
                (),
                None,
                (),
            )

        gate = await run_gate(self._jev, cleaned, s.gate_threshold)
        usage.jev_input_tokens += gate.input_tokens
        if not gate.passed:
            return (
                Outcome.refused_off_topic,
                "I can only answer questions about the Lakewood Creek HOA: its "
                "rules, governance, fees, amenities, and neighborhood. You can "
                f"read the HOA documents at {s.documents_url}",
                (),
                None,
                (),
            )

        swept = await sweep(
            self._jev,
            cleaned,
            self._corpus.chunks,
            top_k=s.sweep_top_k,
            threshold=s.sweep_threshold,
            concurrency=s.sweep_concurrency,
            count_tokens=self._count_tokens,
        )
        usage.jev_input_tokens += swept.input_tokens
        if not swept.selected:
            return self._not_found()

        passages = [scored.chunk for scored in swept.selected]
        by_id = {chunk.id: chunk for chunk in passages}
        for attempt in range(2):
            generated = await self._provider.generate(
                build_prompt(cleaned, passages, retry=attempt > 0)
            )
            usage.answer_calls += 1
            usage.answer_input_tokens += generated.input_tokens
            usage.answer_output_tokens += generated.output_tokens
            draft = generated.draft
            if draft is None:
                usage.notes.append("invalid_draft")
                continue
            quoted = check_quotes(draft.citations, by_id)
            supported = await check_support(
                self._jev,
                draft.answer_text,
                quoted,
                threshold=s.support_threshold,
                count_tokens=self._count_tokens,
            )
            usage.jev_input_tokens += supported.input_tokens
            if supported.kept:
                return (
                    Outcome.answered,
                    draft.answer_text,
                    tuple(_citation(c) for c in supported.kept),
                    float(draft.confidence),
                    tuple(draft.conflicts_noted),
                )
            usage.notes.append("citations_failed")
        return self._not_found()

    def _not_found(
        self,
    ) -> tuple[Outcome, str, tuple[Citation, ...], float | None, tuple[str, ...]]:
        return (
            Outcome.not_found,
            "I couldn't find this in the HOA documents. Please contact the Board "
            f"of Directors or management; the documents are at "
            f"{self._settings.documents_url}",
            (),
            None,
            (),
        )

    async def aclose(self) -> None:
        for client in (self._jev, self._provider):
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()


def _citation(checked: CheckedCitation) -> Citation:
    return Citation(
        chunk_id=checked.chunk.id,
        citation_label=checked.chunk.citation_label,
        url=citation_url(checked.chunk),
        quote=checked.quote,
    )


def _log_result(result: AskResult, usage: _Usage) -> None:
    """Structured request log. Never includes question or answer text (§6)."""
    fields = {
        "request_id": result.answer.request_id,
        "outcome": result.answer.outcome.value,
        "latency_ms": round(result.latency_ms, 1),
        "jev_input_tokens": result.jev_input_tokens,
        "answer_model": usage.model,
        "answer_calls": usage.answer_calls,
        "answer_input_tokens": result.answer_input_tokens,
        "answer_output_tokens": result.answer_output_tokens,
        "citations": len(result.answer.citations),
        "estimated_cost_usd": round(result.estimated_cost_usd, 6),
        "notes": ",".join(usage.notes),
    }
    logger.info(
        "ask " + " ".join(f"{k}=%s" for k in fields),
        *fields.values(),
        extra={"hoa_qa": fields},
    )


def build_asker(
    corpus: Corpus,
    settings: QASettings,
    *,
    jev: JevClient | None = None,
    provider: AnswerProvider | None = None,
    count_tokens: TokenCounter = default_token_counter,
) -> QAAsker:
    """Wire the pipeline; real clients are created from ``settings`` if omitted."""
    if jev is None:
        if settings.typesafe_api_key is None:
            raise ValueError("TYPESAFE_API_KEY is not set")
        jev = TypeSafeJevClient(
            api_key=settings.typesafe_api_key.get_secret_value(),
            model=settings.jev_model,
        )
    if provider is None:
        if settings.anthropic_api_key is None:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        provider = AnthropicAnswerProvider(
            api_key=settings.anthropic_api_key.get_secret_value(),
            model=settings.answer_model,
        )
    return QAAsker(corpus, settings, jev, provider, count_tokens)


AskerFactory = Callable[[Corpus, QASettings], Asker]

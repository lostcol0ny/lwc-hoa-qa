"""End-to-end question answering (spec §4): validate → gate → sweep → answer → verify.

``build_asker(corpus, settings)`` returns an async callable satisfying the web
unit's ``Asker`` protocol, plus a ``max_cost_usd`` worst-case bound the web
budget reserves before each call. Providers are injectable so tests use fakes.
"""

import asyncio
import logging
import os
import time
import unicodedata
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from hoa_qa.answer.pricing import answer_cost_usd, answer_price, jev_cost_usd
from hoa_qa.answer.prompt import (
    MAX_CITATIONS_PER_CLAIM,
    MAX_CLAIMS,
    MAX_STATEMENT_CHARS,
    build_prompt,
    render_passage,
)
from hoa_qa.answer.provider import (
    DEFAULT_ANSWER_MODEL,
    MAX_OUTPUT_TOKENS,
    AnswerDraft,
    AnswerProvider,
    AnthropicAnswerProvider,
)
from hoa_qa.models import Answer, Chunk, Citation, Corpus, Outcome, citation_url
from hoa_qa.retrieval.gate import GATE_QUESTION, run_gate
from hoa_qa.retrieval.jev import (
    DEFAULT_LIMITS,
    JevClient,
    JevLimits,
    PartialCostError,
    TokenCounter,
    TypeSafeJevClient,
    conservative_tokens,
    request_tokens,
)
from hoa_qa.retrieval.sweep import batch_request, plan_batches, sweep
from hoa_qa.verify.quotes import CheckedCitation, check_quotes, normalize
from hoa_qa.verify.support import ClaimEvidence, check_support, support_request

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
DEFAULT_DOCUMENTS_URL = "https://lakewoodcreekhoa.com/"

DISCLAIMER = (
    "Unofficial tool, not legal advice; the governing documents and the Board "
    "are authoritative. Don't include personal information; questions are "
    "processed by third-party AI services."
)
# Connective text added by code, never by the model.
BOARD_REFERRAL = (
    "For a decision about a specific situation or dispute, please contact the "
    "Board of Directors or the management company."
)
OMITTED_NOTE = (
    "Some details could not be verified against the documents and were left out."
)

# Allowance for message framing and the structured-output schema that the
# rendered prompt text does not include.
ANSWER_REQUEST_OVERHEAD_TOKENS = 2_048


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
    jev_concurrency: int = Field(default=4, ge=1)
    support_threshold: float = Field(default=0.5, ge=0, le=1)
    documents_url: str = DEFAULT_DOCUMENTS_URL

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Self:
        env = os.environ if environ is None else environ
        names = {
            "typesafe_api_key": ("TYPESAFE_API_KEY",),
            "anthropic_api_key": ("ANTHROPIC_API_KEY",),
            "answer_model": ("ANSWER_MODEL",),
            "jev_model": ("JEV_MODEL",),
            "gate_threshold": ("GATE_THRESHOLD",),
            "sweep_threshold": ("SWEEP_THRESHOLD",),
            "sweep_top_k": ("SWEEP_TOP_K",),
            # SWEEP_CONCURRENCY is the original name, kept as a fallback.
            "jev_concurrency": ("JEV_CONCURRENCY", "SWEEP_CONCURRENCY"),
            "support_threshold": ("SUPPORT_THRESHOLD",),
            "documents_url": ("HOA_DOCUMENTS_URL",),
        }
        # Empty values (as in .env.example) mean "use the default".
        values: dict[str, str] = {}
        for field_name, variables in names.items():
            for var in variables:
                if env.get(var, "").strip():
                    values[field_name] = env[var].strip()
                    break
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


@dataclass(frozen=True)
class _Verified:
    """One claim after both checks, with the citations that passed the quote check."""

    statement: str
    kind: str
    essential: bool
    citations: tuple[CheckedCitation, ...]
    ok: bool


_Result = tuple[Outcome, str, tuple[Citation, ...], float | None, tuple[str, ...]]


class QAAsker:
    """The concrete ``Asker``. Safe to reuse across requests."""

    def __init__(
        self,
        corpus: Corpus,
        settings: QASettings,
        jev: JevClient,
        provider: AnswerProvider,
        count_tokens: TokenCounter = conservative_tokens,
        limits: JevLimits = DEFAULT_LIMITS,
    ) -> None:
        self._corpus = corpus
        self._settings = settings
        self._jev = jev
        self._provider = provider
        self._count_tokens = count_tokens
        self._limits = limits
        self._max_cost_usd = max_cost_usd(
            corpus, settings, provider.model, count_tokens, limits
        )

    @property
    def max_cost_usd(self) -> float:
        """Worst-case spend of one call, for the web budget's reservation."""
        return self._max_cost_usd

    async def __call__(self, question: str) -> AskResult:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        usage = _Usage(model=self._provider.model)
        try:
            outcome, text, citations, confidence, conflicts = await self._run(
                question, usage
            )
        except Exception as exc:  # any provider failure becomes an error outcome
            if isinstance(exc, PartialCostError):
                # Siblings of the failed Jev batch still billed; count them.
                usage.jev_input_tokens += exc.jev_input_tokens
                error_type = type(exc.errors[0]).__name__
            else:
                error_type = type(exc).__name__
            # Log only the exception type: SDK messages can echo request bodies.
            logger.error(
                "ask failed request_id=%s error_type=%s", request_id, error_type
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

    async def _run(self, question: str, usage: _Usage) -> _Result:
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

        # One bound for every Jev request this call makes (sweep and support).
        limit = asyncio.Semaphore(s.jev_concurrency)
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
            limit=limit,
            count_tokens=self._count_tokens,
            limits=self._limits,
        )
        usage.jev_input_tokens += swept.input_tokens
        if not swept.selected:
            return self._not_found()

        passages = [scored.chunk for scored in swept.selected]
        failed: list[str] | None = None
        for attempt in range(2):
            generated = await self._provider.generate(
                build_prompt(cleaned, passages, failed_claims=failed)
            )
            usage.answer_calls += 1
            usage.answer_input_tokens += generated.input_tokens
            usage.answer_output_tokens += generated.output_tokens
            draft = generated.draft
            if draft is None:
                usage.notes.append("invalid_draft")
                failed = []
                continue
            if not draft.claims:
                # The model found nothing to say; regenerating would not help.
                usage.notes.append("no_claims")
                return self._not_found()
            verified = await self._verify(draft, passages, limit, usage)
            failures = [v for v in verified if not v.ok]
            if not failures and _has_answer(verified):
                return self._answered(draft, verified, dropped=False)
            usage.notes.append(f"claims_failed={len(failures)}")
            if attempt == 1:
                # Drop rule: never show an unverified claim; if an essential
                # claim failed, or no answer claim survives, it's not_found.
                if any(v.essential for v in failures) or not _has_answer(verified):
                    return self._not_found()
                return self._answered(draft, verified, dropped=True)
            failed = [v.statement for v in failures]
        return self._not_found()

    async def _verify(
        self,
        draft: AnswerDraft,
        passages: Sequence[Chunk],
        limit: asyncio.Semaphore,
        usage: _Usage,
    ) -> list[_Verified]:
        by_id = {chunk.id: chunk for chunk in passages}
        evidence = [
            ClaimEvidence(
                statement=claim.statement,
                citations=tuple(check_quotes(claim.citations, by_id)),
            )
            for claim in draft.claims
        ]
        support = await check_support(
            self._jev,
            evidence,
            threshold=self._settings.support_threshold,
            limit=limit,
            count_tokens=self._count_tokens,
            limits=self._limits,
        )
        usage.jev_input_tokens += support.input_tokens
        return [
            _Verified(
                statement=claim.statement,
                kind=claim.kind,
                essential=claim.essential,
                citations=ev.citations,
                ok=bool(ev.citations) and ok,
            )
            for claim, ev, ok in zip(
                draft.claims, evidence, support.supported, strict=True
            )
        ]

    def _answered(
        self, draft: AnswerDraft, verified: Sequence[_Verified], *, dropped: bool
    ) -> _Result:
        kept = [v for v in verified if v.ok]
        parts = [v.statement.strip() for v in kept if v.kind == "answer"]
        if dropped:
            parts.append(OMITTED_NOTE)
        if draft.refer_to_board:
            parts.append(BOARD_REFERRAL)
        conflicts = tuple(v.statement.strip() for v in kept if v.kind == "conflict")
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for v in kept:
            for checked in v.citations:
                key = (checked.chunk.id, normalize(checked.quote))
                if key not in seen:
                    seen.add(key)
                    citations.append(_citation(checked))
        return (
            Outcome.answered,
            " ".join(parts),
            tuple(citations),
            float(draft.confidence),
            conflicts,
        )

    def _not_found(self) -> _Result:
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


def _has_answer(verified: Sequence[_Verified]) -> bool:
    return any(v.ok and v.kind == "answer" for v in verified)


def _citation(checked: CheckedCitation) -> Citation:
    return Citation(
        chunk_id=checked.chunk.id,
        citation_label=checked.chunk.citation_label,
        url=citation_url(checked.chunk),
        quote=checked.quote,
    )


def max_cost_usd(
    corpus: Corpus,
    settings: QASettings,
    answer_model: str,
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> float:
    """Conservative upper bound on one ``ask`` call's spend (docs/qa-core.md).

    Every text is sized with the worst case the pipeline allows: a question of
    MAX_QUESTION_CHARS four-byte characters, the largest chunks in the corpus,
    MAX_CLAIMS claims of MAX_STATEMENT_CHARS four-byte characters each citing
    the MAX_CITATIONS_PER_CLAIM largest chunks, and full MAX_OUTPUT_TOKENS
    output on both answer attempts.
    """
    worst_question = "\U0001d538" * MAX_QUESTION_CHARS
    worst_statement = "\U0001d538" * MAX_STATEMENT_CHARS
    chunks = sorted(
        corpus.chunks, key=lambda c: count_tokens(render_passage(c)), reverse=True
    )

    gate = request_tokens(
        {"message": worst_question}, {"on_topic": GATE_QUESTION}, count_tokens
    )
    plan = plan_batches(worst_question, corpus.chunks, count_tokens, limits)
    sweep_tokens = sum(
        request_tokens(*batch_request(worst_question, batch), count_tokens)[0]
        for batch in plan.batches
    )
    # Packing claims together never costs more than one request per claim.
    worst_claim = ClaimEvidence(
        statement=worst_statement,
        citations=tuple(
            CheckedCitation(chunk=c, quote="") for c in chunks[:MAX_CITATIONS_PER_CLAIM]
        ),
    )
    support_tokens = (
        MAX_CLAIMS * request_tokens(*support_request([worst_claim]), count_tokens)[0]
    )
    jev_tokens = gate[0] + sweep_tokens + 2 * support_tokens

    # The retry prompt (with failed claims listed) is the larger of the two.
    prompt = build_prompt(
        worst_question,
        chunks[: settings.sweep_top_k],
        failed_claims=[worst_statement] * MAX_CLAIMS,
    )
    answer_input = (
        count_tokens(prompt.system + prompt.user) + ANSWER_REQUEST_OVERHEAD_TOKENS
    )
    price = answer_price(answer_model)
    answer_usd = (
        2
        * (
            answer_input * price.input_usd_per_mtok
            + MAX_OUTPUT_TOKENS * price.output_usd_per_mtok
        )
        / 1_000_000
    )
    return jev_cost_usd(jev_tokens) + answer_usd


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
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
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
    return QAAsker(corpus, settings, jev, provider, count_tokens, limits)


AskerFactory = Callable[[Corpus, QASettings], Asker]

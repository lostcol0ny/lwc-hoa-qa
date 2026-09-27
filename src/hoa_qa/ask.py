"""End-to-end question answering (spec §4): validate → gate → sweep → answer → verify.

``build_asker(corpus, settings)`` returns an async callable satisfying the web
unit's ``Asker`` protocol, plus a ``max_cost_usd`` worst-case bound the web
budget reserves before each call. Providers are injectable so tests use fakes.
"""

import asyncio
import json
import logging
import os
import re
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
    ANSWER_SCHEMA,
    DEFAULT_ANSWER_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    AnswerDraft,
    AnswerProvider,
    AnthropicAnswerProvider,
    DraftClaim,
)
from hoa_qa.models import (
    Answer,
    Authority,
    Chunk,
    Citation,
    Corpus,
    Outcome,
    citation_url,
)
from hoa_qa.retrieval.gate import GATE_QUESTION, run_gate
from hoa_qa.retrieval.jev import (
    DEFAULT_LIMITS,
    MAX_QUESTION_CHARS,
    WORST_QUESTION,
    JevClient,
    JevLimits,
    PartialCostError,
    TokenCounter,
    TypeSafeJevClient,
    byte_bound_tokens,
    conservative_tokens,
)
from hoa_qa.retrieval.sweep import (
    SweepPassage,
    batch_request,
    plan_passages,
    sweep,
)
from hoa_qa.trace import NO_TRACE, AskTrace, CitationTrace, ClaimTrace
from hoa_qa.verify.quotes import CheckedCitation, check_quotes, normalize
from hoa_qa.verify.support import (
    ClaimEvidence,
    check_support,
    support_passage,
    support_request,
)

logger = logging.getLogger(__name__)

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

# Authority rule (docs/qa-core.md, "Source authority"): support is not
# authority. When any provided passage is AUTHORITATIVE, an answer claim is
# judged only on its citations outside LOW_AUTHORITY, and a claim left with
# none fails verification ("low_authority"): an informal blog quoting an old
# fine schedule can never be stated as fact next to the current Rules. The
# model may still report such a source as a "conflict" claim, which is shown
# in conflicts_noted. Without an authoritative passage, a low-authority-only
# answer claim may stand, unless it presents itself as current.
# `website` (the Board-run site, including the FAQ and the dues banner) and
# `form` are official HOA publications and are not LOW_AUTHORITY.
AUTHORITATIVE = frozenset(
    {Authority.governing, Authority.rules, Authority.board_decision}
)
LOW_AUTHORITY = frozenset({Authority.informal, Authority.superseded})
_PRESENTS_AS_CURRENT = re.compile(
    r"\b(current|currently|in effect|now in force)\b", re.IGNORECASE
)

# Framing the Messages API adds beyond the prompt text and the output schema
# (role markers, special tokens, the structured-output instructions).
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
    answer_max_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, ge=256, le=16_000)
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
            "answer_max_tokens": ("ANSWER_MAX_TOKENS",),
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
    reason: str  # "ok", or why the claim failed
    trace: ClaimTrace


_AUTHORITY_REASONS = frozenset({"low_authority", "informal_as_current"})


def _authority_screen(
    claim: DraftClaim,
    citations: tuple[CheckedCitation, ...],
    *,
    authoritative: bool,
) -> tuple[tuple[CheckedCitation, ...], str | None]:
    """Apply the authority rule: the citations an answer claim may rest on.

    Returns the usable citations and, if the rule rejects the claim, why.
    Conflict claims are exempt: reporting what an older or informal source
    says is exactly what they are for.
    """
    if claim.kind != "answer" or not citations:
        return citations, None
    strong = tuple(c for c in citations if c.chunk.authority not in LOW_AUTHORITY)
    if strong:
        # Next to an authoritative passage, only official sources count.
        return (strong if authoritative else citations), None
    if authoritative:
        return (), "low_authority"
    if _PRESENTS_AS_CURRENT.search(claim.statement):
        return (), "informal_as_current"
    return citations, None


def _claim_trace(
    claim: DraftClaim,
    by_id: Mapping[str, Chunk],
    usable: Sequence[CheckedCitation],
    probability: float | None,
    *,
    ok: bool,
    reason: str,
) -> ClaimTrace:
    used = {(c.chunk.id, normalize(c.quote)) for c in usable}
    citations = []
    for citation in claim.citations:
        chunk = by_id.get(citation.chunk_id)
        quote = normalize(citation.quote)
        citations.append(
            CitationTrace(
                chunk_id=citation.chunk_id,
                authority=chunk.authority if chunk else None,
                quote_ok=bool(chunk and quote and quote in normalize(chunk.text_clean)),
                used=(citation.chunk_id, quote) in used,
            )
        )
    return ClaimTrace(
        statement=claim.statement,
        kind=claim.kind,
        essential=claim.essential,
        citations=tuple(citations),
        support=probability,
        kept=ok,
        reason=reason,
    )


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
        # How each chunk is sent is fixed per corpus (sized for the longest
        # question), so the sweep's work, and its cost bound, are too.
        self._plan = plan_passages(corpus.chunks, count_tokens, limits)
        self._max_cost_usd = max_cost_usd(
            self._plan.passages, settings, provider.model, provider.max_tokens
        )

    @property
    def max_cost_usd(self) -> float:
        """Worst-case spend of one call, for the web budget's reservation."""
        return self._max_cost_usd

    async def __call__(self, question: str) -> AskResult:
        return await self.ask_traced(question, NO_TRACE)

    async def ask_traced(self, question: str, trace: AskTrace) -> AskResult:
        """``__call__`` reporting each stage to ``trace`` (eval diagnostics only)."""
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        usage = _Usage(model=self._provider.model)
        try:
            outcome, text, citations, confidence, conflicts = await self._run(
                question, usage, trace
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

    async def _run(self, question: str, usage: _Usage, trace: AskTrace) -> _Result:
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
        trace.gate(gate.probability, gate.passed)
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
            self._plan.passages,
            top_k=s.sweep_top_k,
            threshold=s.sweep_threshold,
            limit=limit,
            count_tokens=self._count_tokens,
            limits=self._limits,
        )
        usage.jev_input_tokens += swept.input_tokens
        trace.sweep(swept.scores, swept.selected)
        if not swept.selected:
            return self._not_found()

        passages = [scored.chunk for scored in swept.selected]
        trace.passages(passages)
        failed: list[str] | None = None
        authority_failed = False
        # A first attempt that could stand with its failed claims dropped; a
        # retry that turns out worse falls back to it.
        fallback: tuple[AnswerDraft, list[_Verified]] | None = None
        for attempt in range(2):
            generated = await self._provider.generate(
                build_prompt(
                    cleaned,
                    passages,
                    failed_claims=failed,
                    authority_note=authority_failed,
                )
            )
            usage.answer_calls += 1
            usage.answer_input_tokens += generated.input_tokens
            usage.answer_output_tokens += generated.output_tokens
            draft = generated.draft
            if draft is None:
                usage.notes.append("invalid_draft")
                trace.attempt(attempt + 1, None, generated.note)
                failed = []
                continue
            if not draft.claims:
                # The model found nothing to say; regenerating would not help.
                usage.notes.append("no_claims")
                trace.attempt(attempt + 1, [], generated.note)
                break
            verified = await self._verify(draft, passages, limit, usage)
            trace.attempt(attempt + 1, [v.trace for v in verified], generated.note)
            failures = [v for v in verified if not v.ok]
            if not failures and _has_answer(verified):
                return self._answered(draft, verified, dropped=False)
            usage.notes.append(f"claims_failed={len(failures)}")
            if _can_drop(verified):
                if attempt == 1:
                    return self._answered(draft, verified, dropped=True)
                fallback = (draft, verified)
            failed = [v.statement for v in failures]
            authority_failed = any(v.reason in _AUTHORITY_REASONS for v in failures)
        if fallback is not None:
            usage.notes.append("used_first_attempt")
            return self._answered(*fallback, dropped=True)
        return self._not_found()

    async def _verify(
        self,
        draft: AnswerDraft,
        passages: Sequence[Chunk],
        limit: asyncio.Semaphore,
        usage: _Usage,
    ) -> list[_Verified]:
        by_id = {chunk.id: chunk for chunk in passages}
        authoritative = any(chunk.authority in AUTHORITATIVE for chunk in passages)
        checked = [tuple(check_quotes(c.citations, by_id)) for c in draft.claims]
        screened = [
            _authority_screen(claim, cites, authoritative=authoritative)
            for claim, cites in zip(draft.claims, checked, strict=True)
        ]
        evidence = [
            ClaimEvidence(statement=claim.statement, citations=usable)
            for claim, (usable, _) in zip(draft.claims, screened, strict=True)
        ]
        # A claim with no usable citation is not sent and fails closed.
        support = await check_support(
            self._jev,
            evidence,
            threshold=self._settings.support_threshold,
            limit=limit,
            count_tokens=self._count_tokens,
            limits=self._limits,
        )
        usage.jev_input_tokens += support.input_tokens
        probabilities = support.probabilities or (None,) * len(draft.claims)
        verified: list[_Verified] = []
        for claim, cites, (usable, screen_reason), ok, probability in zip(
            draft.claims,
            checked,
            screened,
            support.supported,
            probabilities,
            strict=True,
        ):
            ok = bool(usable) and ok
            if not cites:
                reason = "no_valid_quote"
            elif screen_reason is not None:
                reason = screen_reason
            elif not ok:
                reason = "unsupported"
            else:
                reason = "ok"
            verified.append(
                _Verified(
                    statement=claim.statement,
                    kind=claim.kind,
                    essential=claim.essential,
                    citations=usable,
                    ok=ok,
                    reason=reason,
                    trace=_claim_trace(
                        claim, by_id, usable, probability, ok=ok, reason=reason
                    ),
                )
            )
        return verified

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


def _can_drop(verified: Sequence[_Verified]) -> bool:
    """Drop rule: the answer may stand without its failed claims.

    Never show an unverified claim. If an essential answer claim failed, or
    no answer claim survives, the answer can't stand. A failed conflict claim
    never blocks it: the verified answer is correct without the note, and
    conflict notes are exactly where hedged cross-source statements land.
    """
    blocking = any(not v.ok and v.essential and v.kind == "answer" for v in verified)
    return _has_answer(verified) and not blocking


def _citation(checked: CheckedCitation) -> Citation:
    return Citation(
        chunk_id=checked.chunk.id,
        citation_label=checked.chunk.citation_label,
        url=citation_url(checked.chunk),
        quote=checked.quote,
    )


def max_cost_usd(
    passages: Sequence[SweepPassage],
    settings: QASettings,
    answer_model: str,
    answer_max_tokens: int,
) -> float:
    """Provable upper bound on one ``ask`` call's spend (docs/qa-core.md).

    Input tokens use ``byte_bound_tokens`` / UTF-8 bytes (a token covers at
    least one byte) and the worst input the pipeline accepts: a question of
    MAX_QUESTION_CHARS 4-byte characters. The sweep charges every planned
    passage as its own request (packing only removes repeated bytes), the
    support check charges MAX_CLAIMS single-claim requests citing the
    corpus's largest serialized passages, and the answer model is charged its
    largest possible prompt and full output on both attempts.
    """
    worst_statement = "\U0001d538" * MAX_STATEMENT_CHARS
    chunks = list({p.chunk.id: p.chunk for p in passages}.values())

    gate = byte_bound_tokens({"message": WORST_QUESTION}, {"on_topic": GATE_QUESTION})
    sweep_tokens = sum(
        byte_bound_tokens(*batch_request(WORST_QUESTION, [passage]))
        for passage in passages
    )

    # Largest chunks by their serialized size inside a support request.
    def support_bytes(chunk: Chunk) -> int:
        serialized = json.dumps(support_passage(chunk), ensure_ascii=False)
        return len(serialized.encode("utf-8"))

    widest = sorted(chunks, key=support_bytes, reverse=True)
    worst_claim = ClaimEvidence(
        statement=worst_statement,
        citations=tuple(
            CheckedCitation(chunk=c, quote="") for c in widest[:MAX_CITATIONS_PER_CLAIM]
        ),
    )
    support_tokens = MAX_CLAIMS * byte_bound_tokens(*support_request([worst_claim]))
    jev_tokens = gate + sweep_tokens + 2 * support_tokens

    # The retry prompt (with failed claims listed) is the larger of the two.
    biggest = sorted(
        chunks, key=lambda c: len(render_passage(c).encode("utf-8")), reverse=True
    )
    prompt = build_prompt(
        WORST_QUESTION,
        biggest[: settings.sweep_top_k],
        failed_claims=[worst_statement] * MAX_CLAIMS,
        authority_note=True,
    )
    answer_input = (
        len((prompt.system + prompt.user).encode("utf-8"))
        + len(json.dumps(ANSWER_SCHEMA).encode("utf-8"))
        + ANSWER_REQUEST_OVERHEAD_TOKENS
    )
    price = answer_price(answer_model)
    answer_usd = (
        2
        * (
            answer_input * price.input_usd_per_mtok
            + answer_max_tokens * price.output_usd_per_mtok
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
            max_tokens=settings.answer_max_tokens,
        )
    return QAAsker(corpus, settings, jev, provider, count_tokens, limits)


AskerFactory = Callable[[Corpus, QASettings], Asker]

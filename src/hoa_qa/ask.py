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
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import asdict, dataclass, field
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from hoa_qa.answer.pricing import answer_cost_usd, answer_price, jev_cost_usd
from hoa_qa.answer.prompt import (
    MAX_CITATIONS_PER_CLAIM,
    MAX_CLAIMS,
    MAX_STATEMENT_CHARS,
    REJECTION_REASONS,
    build_prompt,
    rejected,
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
    DraftIssue,
)
from hoa_qa.answer.statute_notes import (
    ApplicabilityNote,
    applicability_note,
    cites_cicaa,
    statute_disclaimer,
)
from hoa_qa.models import (
    Answer,
    Authority,
    Chunk,
    Citation,
    Corpus,
    DocumentLink,
    Outcome,
    OutcomeReason,
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
    ScoredChunk,
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
DEFAULT_JEV_MODEL = "jev-1.13.0"

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
# At most this many related documents are linked from an unverified not_found.
MAX_RELATED_DOCUMENTS = 5

# Authority rule (docs/qa-core.md, "Source authority"): support is not
# authority. When any provided passage is AUTHORITATIVE, an answer claim is
# judged only on its citations outside LOW_AUTHORITY, and a claim left with
# none fails verification ("low_authority"): an informal blog quoting an old
# fine schedule can never be stated as fact next to the current Rules. The
# model may still report such a source as a "conflict" claim, which is shown
# in conflicts_noted, but never as current: a conflict claim citing a
# low-authority passage fails if it presents itself as current. Without an
# authoritative passage, a low-authority-only answer claim may stand, unless
# it presents itself as current.
# `website` (the Board-run site, including the FAQ and the dues banner) and
# `form` are official HOA publications and are not LOW_AUTHORITY. `statute`
# (Illinois law) is authoritative too: an informal post can't answer next to it.
AUTHORITATIVE = frozenset(
    {
        Authority.statute,
        Authority.governing,
        Authority.rules,
        Authority.board_decision,
    }
)
LOW_AUTHORITY = frozenset({Authority.informal, Authority.superseded})
_PRESENTS_AS_CURRENT = re.compile(
    r"\b(current|currently|in effect|in force|as of now|presently|at present|"
    r"today|now)\b",
    re.IGNORECASE,
)
# A match right after one of these reads as "not current" ("no longer in
# effect", "was in force"), and "now" right before one of the words after it
# reads as outdated ("now superseded").
_NEGATED_BEFORE = re.compile(
    r"\b(no longer|not|never|formerly|previously|was|were|isn't|aren't|wasn't)"
    r"(\s+\w+)?\s*$",
    re.IGNORECASE,
)
_OUTDATED_AFTER = re.compile(
    r"^\s*(\w+\s+)?(superseded|outdated|obsolete|replaced|repealed|expired|"
    r"out of date|no longer)\b",
    re.IGNORECASE,
)

# Statute-backed claims state what the law says ("765 ILCS 160/1-30 states
# that ..."), never the reader's rights or what applies in their case, and
# never that the Association is breaking the law (addendum §6.3).
_ADVICE = re.compile(
    r"\byou(?:'ve| have| has)? (?:a |the )?rights?\b|\byour (?:legal )?rights?\b|"
    r"\byou(?:'re| are) (?:legally )?entitled\b|\bin your (?:case|situation)\b|"
    r"\b(?:is|are|was|were) (?:violating|breaking|in violation of)\b|"
    r"\bbreaking the law\b",
    re.IGNORECASE,
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
    # Pinned: the version the thresholds were tuned against (the alias
    # jev-latest pointed to it during tuning). Re-run the eval and re-tune
    # before bumping it.
    jev_model: str = DEFAULT_JEV_MODEL
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


def gives_advice(statement: str) -> bool:
    """True if a statement tells the reader their rights or legal position."""
    return _ADVICE.search(statement) is not None


def _advice_screen(
    claim: DraftClaim, citations: tuple[CheckedCitation, ...]
) -> str | None:
    """A statute-backed claim must say what the statute states, not advise."""
    statute = any(c.chunk.authority is Authority.statute for c in citations)
    return "advice_phrasing" if statute and gives_advice(claim.statement) else None


def _authority_screen(
    claim: DraftClaim,
    citations: tuple[CheckedCitation, ...],
    *,
    authoritative: bool,
) -> tuple[tuple[CheckedCitation, ...], str | None]:
    """Apply the authority rule: the citations an answer claim may rest on.

    Returns the usable citations and, if the rule rejects the claim, why.
    Conflict claims keep their citations (reporting what an older or informal
    source says is what they are for), but one citing a low-authority passage
    must not present it as current.
    """
    if not citations:
        return citations, None
    if claim.kind != "answer":
        low = any(c.chunk.authority in LOW_AUTHORITY for c in citations)
        if low and presents_as_current(claim.statement):
            return (), "informal_as_current"
        return citations, None
    strong = tuple(c for c in citations if c.chunk.authority not in LOW_AUTHORITY)
    if strong:
        # Next to an authoritative passage, only official sources count.
        return (strong if authoritative else citations), None
    if authoritative:
        return (), "low_authority"
    if presents_as_current(claim.statement):
        return (), "informal_as_current"
    return citations, None


def presents_as_current(statement: str) -> bool:
    """True if the statement calls something current ("currently", "now",
    "in effect", ...), ignoring negated or past uses ("no longer in effect",
    "was in force", "now superseded")."""
    for match in _PRESENTS_AS_CURRENT.finditer(statement):
        if _NEGATED_BEFORE.search(statement[: match.start()]):
            continue
        if match.group(1).lower() == "now" and _OUTDATED_AFTER.match(
            statement[match.end() :]
        ):
            continue
        return True
    return False


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
                quote=citation.quote,
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


@dataclass(frozen=True)
class _Result:
    outcome: Outcome
    text: str
    citations: tuple[Citation, ...] = ()
    confidence: float | None = None
    conflicts: tuple[str, ...] = ()
    reason: OutcomeReason | None = None
    related: tuple[DocumentLink, ...] = ()


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
        # Checked up front: a corpus with CICAA but stale evidence fails here.
        self._applicability: ApplicabilityNote | None = (
            applicability_note(corpus.chunks) if cites_cicaa(corpus.chunks) else None
        )
        self._statute_disclaimer: str | None = None
        if any(chunk.authority is Authority.statute for chunk in corpus.chunks):
            compiled = corpus.manifest.statute_compilation
            if compiled is None:
                raise ValueError("corpus has statutes but no statute_compilation")
            self._statute_disclaimer = statute_disclaimer(compiled)

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
            result = await self._run(question, usage, trace, request_id)
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
            result = _Result(
                Outcome.error,
                "Something went wrong answering this question. Please try again "
                f"later, or read the HOA documents at {self._settings.documents_url}",
            )
        answer = Answer(
            request_id=request_id,
            outcome=result.outcome,
            answer_text=result.text,
            citations=result.citations,
            confidence=result.confidence,
            conflicts_noted=result.conflicts,
            disclaimer=DISCLAIMER,
            reason=result.reason,
            related_documents=result.related,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        asked = AskResult(
            answer=answer,
            estimated_cost_usd=usage.cost(),
            jev_input_tokens=usage.jev_input_tokens,
            answer_input_tokens=usage.answer_input_tokens,
            answer_output_tokens=usage.answer_output_tokens,
            latency_ms=latency_ms,
        )
        _log_result(asked, usage)
        return asked

    async def _run(
        self, question: str, usage: _Usage, trace: AskTrace, request_id: str
    ) -> _Result:
        s = self._settings
        cleaned = clean_question(question)
        if cleaned is None:
            return _Result(
                Outcome.invalid_input,
                f"Please enter a question between 1 and {MAX_QUESTION_CHARS} "
                "characters.",
            )

        # One bound for every Jev request this call makes (sweep and support).
        limit = asyncio.Semaphore(s.jev_concurrency)
        gate = await run_gate(self._jev, cleaned, s.gate_threshold)
        usage.jev_input_tokens += gate.input_tokens
        trace.gate(gate.probability, gate.passed)
        if not gate.passed:
            return _Result(
                Outcome.refused_off_topic,
                "I can only answer questions about the Lakewood Creek HOA: its "
                "rules, governance, fees, amenities, and neighborhood. You can "
                f"read the HOA documents at {s.documents_url}",
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
        _log_sweep(request_id, swept.selected)
        if not swept.selected:
            return self._not_found(OutcomeReason.no_relevant_passages)

        passages = [scored.chunk for scored in swept.selected]
        trace.passages(passages)
        passage_ids = {chunk.id for chunk in passages}
        failed: list[str] | None = None
        authority_failed = False
        invalid_output = False
        # Provided passages the model cited, in order, for an unverified
        # not_found's related-document links (never their text).
        cited_ids: list[str] = []
        # A first attempt that could stand with its failed claims dropped; a
        # retry that turns out worse falls back to it.
        fallback: tuple[AnswerDraft, list[_Verified]] | None = None
        for attempt in range(1, 3):
            generated = await self._provider.generate(
                build_prompt(
                    cleaned,
                    passages,
                    failed_claims=failed,
                    authority_note=authority_failed,
                    invalid_output=invalid_output,
                )
            )
            usage.answer_calls += 1
            usage.answer_input_tokens += generated.input_tokens
            usage.answer_output_tokens += generated.output_tokens
            draft = generated.draft
            if draft is None:
                usage.notes.append("invalid_draft")
                trace.attempt(attempt, None, generated.note)
                _log_attempt(
                    request_id,
                    attempt,
                    "invalid_draft",
                    note=generated.note,
                    issues=generated.issues,
                )
                # Code-written, per-claim reasons only: an invalid draft's
                # text is never echoed back.
                failed = [_issue_entry(issue) for issue in generated.issues]
                authority_failed = False
                invalid_output = True
                continue
            if not draft.claims:
                # The model found nothing to say; regenerating would not help.
                usage.notes.append("no_claims")
                trace.attempt(attempt, [], generated.note)
                _log_attempt(request_id, attempt, "no_claims", note=generated.note)
                if attempt == 1:
                    return self._not_found(OutcomeReason.no_answer_in_passages)
                break
            for c in (c for claim in draft.claims for c in claim.citations):
                if c.chunk_id in passage_ids and c.chunk_id not in cited_ids:
                    cited_ids.append(c.chunk_id)
            verified = await self._verify(draft, passages, limit, usage)
            trace.attempt(attempt, [v.trace for v in verified], generated.note)
            failures = [v for v in verified if not v.ok]
            can_drop = _can_drop(verified)
            clean = not failures and _has_answer(verified)
            _log_attempt(
                request_id,
                attempt,
                "verified" if clean else "claims_failed",
                note=generated.note,
                claims=_claim_diagnostics(draft, verified, passage_ids),
                fallback_eligible=can_drop,
            )
            if clean:
                # Claims salvage dropped at parse time count as omitted too.
                return self._answered(draft, verified, dropped=generated.claims_trimmed)
            usage.notes.append(f"claims_failed={len(failures)}")
            if can_drop:
                if attempt == 2:
                    return self._answered(draft, verified, dropped=True)
                fallback = (draft, verified)
            failed = [
                rejected(v.statement, v.reason, index)
                for index, v in enumerate(verified, start=1)
                if not v.ok
            ]
            authority_failed = any(v.reason in _AUTHORITY_REASONS for v in failures)
            invalid_output = False
        if fallback is not None:
            usage.notes.append("used_first_attempt")
            return self._answered(*fallback, dropped=True)
        return self._not_found(
            OutcomeReason.unverified, self._related_documents(cited_ids, passages)
        )

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
        for index, (claim, (usable, _)) in enumerate(
            zip(draft.claims, screened, strict=True)
        ):
            advice = _advice_screen(claim, usable)
            if advice is not None:
                screened[index] = ((), advice)
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
        cited = [checked.chunk for v in kept for checked in v.citations]
        notes: list[Citation] = []
        if self._statute_disclaimer is not None and any(
            chunk.authority is Authority.statute for chunk in cited
        ):
            parts.append(self._statute_disclaimer)
            if self._applicability is not None and cites_cicaa(cited):
                parts.append(self._applicability.text)
                notes += self._applicability.citations
        conflicts = tuple(v.statement.strip() for v in kept if v.kind == "conflict")
        citations: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for citation in [
            *(_citation(checked) for v in kept for checked in v.citations),
            *notes,
        ]:
            key = (citation.chunk_id, normalize(citation.quote))
            if key not in seen:
                seen.add(key)
                citations.append(citation)
        return _Result(
            Outcome.answered,
            " ".join(parts),
            tuple(citations),
            float(draft.confidence),
            conflicts,
        )

    def _not_found(
        self, reason: OutcomeReason, related: tuple[DocumentLink, ...] = ()
    ) -> _Result:
        if reason is OutcomeReason.unverified:
            # Related passages exist but no answer could be verified: say so,
            # and point to whole documents, never to unverified text.
            text = (
                "I found related passages in the HOA documents but couldn't "
                "verify a precise answer. Try rephrasing your question"
            )
            if related:
                titles = "; ".join(link.title for link in related)
                text += f", or see: {titles}."
            else:
                text += f", or read the documents at {self._settings.documents_url}."
            return _Result(Outcome.not_found, text, reason=reason, related=related)
        return _Result(
            Outcome.not_found,
            "I couldn't find this in the HOA documents. Please contact the Board "
            f"of Directors or management; the documents are at "
            f"{self._settings.documents_url}",
            reason=reason,
        )

    @staticmethod
    def _related_documents(
        cited_ids: Sequence[str], passages: Sequence[Chunk]
    ) -> tuple[DocumentLink, ...]:
        """Document-level links: cited passages' documents first, then the
        rest of the retrieved ones, one link per document URL."""
        by_id = {chunk.id: chunk for chunk in passages}
        ordered = [by_id[i] for i in cited_ids if i in by_id] + list(passages)
        links: dict[str, DocumentLink] = {}
        for chunk in ordered:
            if chunk.source_url in links:
                continue
            # A statute's URL is one section's page, so its label names it.
            title = (
                chunk.citation_label
                if chunk.authority is Authority.statute
                else chunk.doc_title
            )
            links[chunk.source_url] = DocumentLink(title=title, url=chunk.source_url)
            if len(links) == MAX_RELATED_DOCUMENTS:
                break
        return tuple(links.values())

    async def aclose(self) -> None:
        for client in (self._jev, self._provider):
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()


def _has_answer(verified: Sequence[_Verified]) -> bool:
    return any(v.ok and v.kind == "answer" for v in verified)


def _issue_entry(issue: DraftIssue) -> str:
    index = None if issue.index is None else issue.index + 1
    return rejected("", issue.code, index)


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
    # At most MAX_CLAIMS entries either way: one per draft claim, or the
    # capped issues of an invalid draft, which carry no statement at all.
    longest_reason = max(REJECTION_REASONS, key=lambda r: len(REJECTION_REASONS[r]))
    prompt = build_prompt(
        WORST_QUESTION,
        biggest[: settings.sweep_top_k],
        failed_claims=[rejected(worst_statement, longest_reason, MAX_CLAIMS)]
        * MAX_CLAIMS,
        authority_note=True,
        invalid_output=True,
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
        "reason": result.answer.reason.value if result.answer.reason else "",
    }
    logger.info(
        "ask " + " ".join(f"{k}=%s" for k in fields),
        *fields.values(),
        extra={"hoa_qa": fields},
    )


# --- diagnostics logs --------------------------------------------------------
# Text-free by construction (§6): only codes, numbers, flags, and the chunk
# IDs of provided passages. A model-written chunk_id that names no provided
# passage is logged as "?", since the model can write anything there.


def _log_sweep(request_id: str, selected: Sequence[ScoredChunk]) -> None:
    fields = {
        "request_id": request_id,
        "selected": [
            {"chunk_id": s.chunk.id, "score": round(s.probability, 3)} for s in selected
        ],
    }
    logger.info(
        "ask sweep request_id=%s selected=%s",
        request_id,
        ",".join(f"{s['chunk_id']}:{s['score']}" for s in fields["selected"]) or "-",
        extra={"hoa_qa_sweep": fields},
    )


@dataclass(frozen=True)
class _ClaimDiagnostic:
    index: int  # 1-based, as the retry prompt numbers claims
    kind: str
    essential: bool
    reason: str
    support: float | None
    chunk_ids: tuple[str, ...]
    statement_chars: int

    def compact(self) -> str:
        essential = "E" if self.essential else "n"
        return (
            f"{self.index}/{self.kind}/{essential}/{self.reason}/s={self.support}"
            f"/{'+'.join(self.chunk_ids)}/len={self.statement_chars}"
        )


def _claim_diagnostics(
    draft: AnswerDraft, verified: Sequence[_Verified], passage_ids: Set[str]
) -> list[_ClaimDiagnostic]:
    return [
        _ClaimDiagnostic(
            index=index,
            kind=claim.kind,
            essential=claim.essential,
            reason=v.reason,
            support=None if v.trace.support is None else round(v.trace.support, 3),
            chunk_ids=tuple(
                c.chunk_id if c.chunk_id in passage_ids else "?"
                for c in claim.citations
            ),
            statement_chars=len(claim.statement),
        )
        for index, (claim, v) in enumerate(
            zip(draft.claims, verified, strict=True), start=1
        )
    ]


def _log_attempt(
    request_id: str,
    attempt: int,
    terminal: str,
    *,
    note: str | None = None,
    issues: Sequence[DraftIssue] = (),
    claims: Sequence[_ClaimDiagnostic] = (),
    fallback_eligible: bool = False,
) -> None:
    """One answer-model attempt: how it ended and why each claim failed."""
    issue_rows = [
        {"index": None if i.index is None else i.index + 1, "code": i.code}
        for i in issues
    ]
    fields = {
        "request_id": request_id,
        "attempt": attempt,
        "terminal": terminal,
        "fallback_eligible": fallback_eligible,
        "note": note or "",
        "issues": issue_rows,
        "claims": [asdict(c) for c in claims],
    }
    logger.info(
        "ask attempt request_id=%s attempt=%s terminal=%s fallback_eligible=%s "
        "claims=%s issues=%s note=%s",
        request_id,
        attempt,
        terminal,
        fallback_eligible,
        ",".join(c.compact() for c in claims) or "-",
        ",".join(f"{i['index'] or '-'}:{i['code']}" for i in issue_rows) or "-",
        note or "-",
        extra={"hoa_qa_attempt": fields},
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

"""Operator monitor: a full trace of the operator's own question, and a
metadata-only feed of everyone's traffic.

Two different privacy positions, on purpose:

- **The operator's own question** (``POST /api/monitor/ask``) comes back with
  everything the pipeline did: scores, the passages sent to the answer model,
  each claim and check. It is returned to the caller and never stored.
- **The traffic feed** records one row per answered request with outcomes,
  timings, token counts and cost. No question, answer, claim, quote or IP is
  ever in a row, so the rule in docs/qa-core.md ("Logging and privacy") holds.

Everything here is off unless ``MONITOR_TOKEN`` is set, and every endpoint
answers 404 without the right token, as if it did not exist.
"""

import hmac
import json
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from starlette.requests import Request

from hoa_qa.answer.prompt import (
    REJECTION_REASONS,
    SYSTEM_PROMPT,
    build_prompt,
    rejected,
    render_passage,
)
from hoa_qa.ask import STAGES, AskResult, QASettings
from hoa_qa.budget import RecentStore
from hoa_qa.cache import CacheEntry, match_request
from hoa_qa.models import Chunk, Corpus
from hoa_qa.retrieval.gate import GATE_QUESTION
from hoa_qa.retrieval.jev import NoulQuestion
from hoa_qa.retrieval.sweep import (
    ScoredChunk,
    SweepPassage,
    batch_request,
    plan_batches,
    plan_passages,
)
from hoa_qa.trace import ClaimTrace
from hoa_qa.verify.quotes import CheckedCitation
from hoa_qa.verify.support import ClaimEvidence, support_request

logger = logging.getLogger("hoa_qa.web")

TOKEN_HEADER = "x-monitor-token"
# Shorter than this is guessable, so the monitor stays off.
MIN_TOKEN_CHARS = 24
RECENT_KEY = "hoa-qa:recent"
RECENT_KEEP = 200
RECENT_TTL_SECONDS = 7 * 24 * 60 * 60
SCORES_SHOWN = 20
SNIPPET_CHARS = 260
_AUTHORITY_REASONS = frozenset({"low_authority", "informal_as_current"})


def monitor_token(env: Mapping[str, str]) -> str | None:
    """The configured operator token, or None when the monitor is off."""
    token = env.get("MONITOR_TOKEN", "").strip()
    if not token:
        return None
    if len(token) < MIN_TOKEN_CHARS:
        logger.warning(
            "MONITOR_TOKEN is shorter than %d characters; the monitor stays off",
            MIN_TOKEN_CHARS,
        )
        return None
    return token


def authorized(request: Request, token: str | None) -> bool:
    if token is None:
        return False
    sent = request.headers.get(TOKEN_HEADER, "")
    # Constant-time, so response timing says nothing about the token.
    return hmac.compare_digest(sent.encode(), token.encode())


def _clip(text: str) -> str:
    if len(text) <= SNIPPET_CHARS:
        return text
    return text[:SNIPPET_CHARS].rstrip() + " …"


def _wire(question: NoulQuestion) -> dict[str, str | None]:
    return {
        "instructions": question.instructions,
        "yes": question.yes,
        "no": question.no,
    }


@dataclass
class MonitorRecorder:
    """An ``AskTrace`` that keeps what each stage saw, for one question."""

    gate_score: float | None = None
    gate_passed: bool | None = None
    cache_ran: bool = False
    cache_entry: CacheEntry | None = None
    cache_probability: float | None = None
    cache_candidates: int = 0
    cache_exact: bool = False
    scores: tuple[ScoredChunk, ...] = ()
    selected: tuple[ScoredChunk, ...] = ()
    swept: bool = False
    chunks: tuple[Chunk, ...] = ()
    attempts: list[tuple[int, tuple[ClaimTrace, ...] | None, str | None]] = field(
        default_factory=list
    )

    def gate(self, probability: float, passed: bool) -> None:
        self.gate_score, self.gate_passed = probability, passed

    def cache(
        self,
        entry: CacheEntry | None,
        probability: float | None,
        candidates: int,
        *,
        exact: bool,
    ) -> None:
        self.cache_ran = True
        self.cache_entry, self.cache_probability = entry, probability
        self.cache_candidates, self.cache_exact = candidates, exact

    def sweep(
        self, scores: Sequence[ScoredChunk], selected: Sequence[ScoredChunk]
    ) -> None:
        self.swept = True
        self.scores, self.selected = tuple(scores), tuple(selected)

    def passages(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = tuple(chunks)

    def attempt(
        self,
        number: int,
        claims: Sequence[ClaimTrace] | None,
        note: str | None = None,
    ) -> None:
        self.attempts.append((number, None if claims is None else tuple(claims), note))


class Monitor:
    """Builds the monitor's payloads for one corpus."""

    def __init__(self, corpus: Corpus, settings: QASettings) -> None:
        self._settings = settings
        self._index = {chunk.id: i for i, chunk in enumerate(corpus.chunks)}
        self._by_id = {chunk.id: chunk for chunk in corpus.chunks}
        self._plan: tuple[SweepPassage, ...] = plan_passages(corpus.chunks).passages
        docs: list[dict[str, Any]] = []
        doc_index: dict[str, int] = {}
        for chunk in corpus.chunks:
            if chunk.doc_id not in doc_index:
                doc_index[chunk.doc_id] = len(docs)
                docs.append({"title": chunk.doc_title})
        self._docs = docs
        self._chunks = [
            {"doc": doc_index[chunk.doc_id], "label": chunk.citation_label}
            for chunk in corpus.chunks
        ]

    def config(self) -> dict[str, Any]:
        """What the page needs once: names, thresholds, prompts, passage labels."""
        s = self._settings
        return {
            "models": {"jev": s.jev_model, "answer": s.answer_model},
            "thresholds": {
                "gate": s.gate_threshold,
                "sweep": s.sweep_threshold,
                "support": s.support_threshold,
                "cache": s.cache_threshold,
                "top_k": s.sweep_top_k,
            },
            "concurrency": s.jev_concurrency,
            "stages": list(STAGES),
            "gate_question": _wire(GATE_QUESTION),
            "system_prompt": SYSTEM_PROMPT,
            "reasons": REJECTION_REASONS,
            "docs": self._docs,
            "chunks": self._chunks,
        }

    def trace(
        self, question: str, result: AskResult, seen: MonitorRecorder
    ) -> dict[str, Any]:
        """Everything the pipeline did for the operator's own question.

        Scores, claims and timings are what happened; the requests are
        rebuilt with the pipeline's own functions, with long passages clipped.
        """
        answer = result.answer
        return {
            "question": question,
            "request_id": answer.request_id,
            "outcome": answer.outcome.value,
            "reason": answer.reason.value if answer.reason else None,
            "latency_ms": round(result.latency_ms, 1),
            "stage_ms": {k: round(v, 1) for k, v in result.stage_ms.items()},
            "cost_usd": result.estimated_cost_usd,
            "tokens": {
                "jev": result.jev_input_tokens,
                "answer_in": result.answer_input_tokens,
                "answer_out": result.answer_output_tokens,
            },
            "answer_text": answer.answer_text,
            "conflicts": list(answer.conflicts_noted),
            "cited": [c.citation_label for c in answer.citations],
            "gate": None
            if seen.gate_score is None
            else {"score": seen.gate_score, "passed": seen.gate_passed},
            "cache": self._cache(question, seen),
            "sweep": self._sweep(question, seen),
            "passages": [self._passage(chunk) for chunk in seen.chunks],
            "attempts": self._attempts(question, seen),
        }

    def _cache(self, question: str, seen: MonitorRecorder) -> dict[str, Any] | None:
        if not seen.cache_ran:
            return None
        entry = seen.cache_entry
        sample = None
        if entry is not None and not seen.cache_exact:
            state, questions = match_request(question, [entry])
            sample = {"state": state, "question": _wire(questions["k0"])}
        return {
            "hit": entry.id if entry else None,
            "matched_question": entry.question if entry else None,
            "exact": seen.cache_exact,
            "probability": seen.cache_probability,
            "candidates": seen.cache_candidates,
            "sample": sample,
        }

    def _sweep(self, question: str, seen: MonitorRecorder) -> dict[str, Any] | None:
        if not seen.swept:
            return None
        batches = plan_batches(question, self._plan).batches
        chosen = {s.chunk.id for s in seen.selected}
        top = [
            {
                "i": self._index[s.chunk.id],
                "score": round(s.probability, 4),
                "selected": s.chunk.id in chosen,
                "heading": " > ".join(s.chunk.heading_path),
            }
            for s in seen.scores[:SCORES_SHOWN]
        ]
        sample = None
        if top:
            scores = {t["i"]: t["score"] for t in top}
            best = top[0]["i"]
            batch = next(
                b for b in batches if any(self._index[p.chunk.id] == best for p in b)
            )
            state, questions = batch_request(question, batch)
            state["passages"] = {
                name: {"heading": p["heading"], "text": _clip(p["text"])}
                for name, p in state["passages"].items()
            }
            sample = {
                "state": state,
                "questions": [
                    {
                        "name": name,
                        **_wire(q),
                        "score": scores.get(self._index[batch[n].chunk.id]),
                    }
                    for n, (name, q) in enumerate(questions.items())
                ],
            }
        return {
            "plan": [[self._index[p.chunk.id] for p in batch] for batch in batches],
            "top": top,
            "sample": sample,
        }

    def _passage(self, chunk: Chunk) -> dict[str, str]:
        return {
            "id": chunk.id,
            "tag": render_passage(chunk).split("\n", 1)[0],
            "label": chunk.citation_label,
            "authority": chunk.authority.value,
            "text": chunk.text_clean,
        }

    def _attempts(self, question: str, seen: MonitorRecorder) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        failed: list[str] | None = None
        authority_note = invalid_output = False
        for number, claims, note in seen.attempts:
            prompt = build_prompt(
                question,
                seen.chunks,
                failed_claims=failed,
                authority_note=authority_note,
                invalid_output=invalid_output,
            )
            kept_for_support = [
                ClaimEvidence(
                    statement=claim.statement,
                    citations=tuple(
                        CheckedCitation(chunk=self._by_id[c.chunk_id], quote=c.quote)
                        for c in claim.citations
                        if c.used and c.chunk_id in self._by_id
                    ),
                )
                for claim in claims or ()
            ]
            checked = [e for e in kept_for_support if e.citations]
            support = None
            if checked:
                state, questions = support_request(checked)
                first = state["claims"][0]
                support = {
                    "claims": len(checked),
                    "state_first": {
                        "statement": first["statement"],
                        "passages": [
                            {**p, "text": _clip(p["text"])} for p in first["passages"]
                        ],
                    },
                    "question": _wire(next(iter(questions.values()))),
                }
            built.append(
                {
                    "n": number,
                    "valid": claims is not None,
                    "note": note,
                    "feedback": prompt.user.split("</question>", 1)[1].strip(),
                    "prompt_chars": len(prompt.system) + len(prompt.user),
                    "claims": [self._claim(claim) for claim in claims or ()],
                    "support": support,
                }
            )
            if claims is None:
                # An unusable draft: the retry prompt lists issue codes this
                # trace does not carry, so the feedback shown is approximate.
                failed, authority_note, invalid_output = [], False, True
            else:
                rejects = [(n, c) for n, c in enumerate(claims, start=1) if not c.kept]
                failed = [rejected(c.statement, c.reason, n) for n, c in rejects]
                authority_note = any(c.reason in _AUTHORITY_REASONS for _, c in rejects)
                invalid_output = False
        return built

    def _claim(self, claim: ClaimTrace) -> dict[str, Any]:
        return {
            "statement": claim.statement,
            "kind": claim.kind,
            "essential": claim.essential,
            "support": claim.support,
            "kept": claim.kept,
            "reason": claim.reason,
            "citations": [
                {
                    "chunk_id": c.chunk_id,
                    "quote": c.quote,
                    "quote_ok": c.quote_ok,
                    "used": c.used,
                }
                for c in claim.citations
            ],
        }


def feed_row(
    result: object, latency_ms: float, cost_usd: float, *, operator: bool
) -> dict[str, Any]:
    """One traffic-feed row: outcomes and numbers only, never any text.

    ``result`` is whatever the asker returned; the QA core's ``AskResult`` has
    all of these, and anything missing (a test double) is left out.
    """
    answer = getattr(result, "answer", None)
    stage_ms = getattr(result, "stage_ms", None) or {}
    return {
        "t": datetime.now(UTC).isoformat(timespec="seconds"),
        "outcome": getattr(getattr(answer, "outcome", None), "value", None),
        "reason": getattr(getattr(answer, "reason", None), "value", None),
        "latency_ms": round(latency_ms),
        "stage_ms": {k: round(float(v)) for k, v in stage_ms.items()},
        "cache": getattr(result, "cache_entry", None),
        "cost_usd": round(cost_usd, 6),
        "jev_tokens": getattr(result, "jev_input_tokens", None),
        "answer_in": getattr(result, "answer_input_tokens", None),
        "answer_out": getattr(result, "answer_output_tokens", None),
        "citations": len(getattr(answer, "citations", ()) or ()),
        "operator": operator,
    }


async def record(store: object, row: dict[str, Any]) -> None:
    """Add a row to the feed. Failing to record never fails the request."""
    push = getattr(store, "push_recent", None)
    if push is None:
        return
    try:
        await push(RECENT_KEY, json.dumps(row), RECENT_KEEP, RECENT_TTL_SECONDS)
    except Exception as exc:
        logger.error("monitor feed write failed error_type=%s", type(exc).__name__)


async def recent(store: RecentStore) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for raw in await store.recent(RECENT_KEY, RECENT_KEEP):
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    answered = [r for r in rows if r.get("outcome") == "answered"]
    latencies = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), int)]
    return {
        "rows": rows,
        "summary": {
            "requests": len(rows),
            "answered": len(answered),
            "cache_hits": sum(1 for r in rows if r.get("cache")),
            "median_latency_ms": statistics.median(latencies) if latencies else None,
            "cost_usd": round(sum(r.get("cost_usd") or 0 for r in rows), 4),
            "keep": RECENT_KEEP,
            "days": RECENT_TTL_SECONDS // 86_400,
        },
    }

"""Curated answer cache: stored answers to questions the operator chose.

Each entry is a question from a committed list and the full answer the
pipeline produced for it, verified in the usual way when the cache was built.
At request time ``lookup`` asks Jev, per entry, whether that entry's answer
would fully answer the resident's question; a confident match returns the
stored answer and skips the sweep and the answer model.

Three properties this design keeps:

- **No resident text is stored.** Only the curated questions are in the file,
  so nobody outside can add or change an entry.
- **A miss is always safe.** Anything uncertain (a low score, differing
  numbers, a failed request, a cache built from another corpus) falls through
  to the full pipeline.
- **Numbers are compared in code.** Jev 1.13 is weak on numeric precision
  (docs.typesafe.ai/model-jaggedness/jev-1.13), and "second violation" vs
  "third violation" is exactly how near-miss questions differ, so an entry is
  only a candidate if it mentions the same numbers as the question.
"""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from hoa_qa.answer.prompt import (
    MAX_CITATIONS_PER_CLAIM,
    MAX_CLAIMS,
    MAX_STATEMENT_CHARS,
)
from hoa_qa.models import Answer, Corpus, Model, Outcome
from hoa_qa.retrieval.jev import (
    MAX_QUESTION_CHARS,
    JevClient,
    NoulQuestion,
    PartialCostError,
    run_nouls,
)

logger = logging.getLogger(__name__)

DEFAULT_CACHE_THRESHOLD = 0.9
# Entries judged in one request. Same reason as the sweep's cap: with long
# lists the judge scored a neighbor instead of the item it was asked about.
MAX_ENTRIES_PER_REQUEST = 8

EntryId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]*$")]
QuestionText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS
    ),
]


class ReviewedCitation(Model):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ReviewedClaim(Model):
    """One sentence of a stored answer, written or approved by the operator.

    The build checks it exactly like a claim the answer model wrote: the quote
    must be in the cited passage word for word, the source must be allowed to
    answer, and Jev must judge that the passages state all of it.
    """

    statement: str = Field(min_length=1, max_length=MAX_STATEMENT_CHARS)
    kind: Literal["answer", "conflict"] = "answer"
    citations: tuple[ReviewedCitation, ...] = Field(
        min_length=1, max_length=MAX_CITATIONS_PER_CLAIM
    )


class CuratedQuestion(Model):
    """``claims`` empty: the answer model drafts the answer at build time.

    With ``claims``, the answer model is not used: the stored answer is these
    claims, in this order, if every one of them passes verification.
    """

    id: EntryId
    question: QuestionText
    claims: tuple[ReviewedClaim, ...] = Field(default=(), max_length=MAX_CLAIMS)
    refer_to_board: bool = False


class CuratedQuestions(Model):
    questions: tuple[CuratedQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique(self) -> "CuratedQuestions":
        ids = [q.id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question ids")
        texts = [normalize_question(q.question) for q in self.questions]
        if len(texts) != len(set(texts)):
            raise ValueError("duplicate questions")
        return self


class CacheEntry(Model):
    id: EntryId
    question: QuestionText
    answer: Answer

    @model_validator(mode="after")
    def answered_only(self) -> "CacheEntry":
        if self.answer.outcome is not Outcome.answered:
            raise ValueError("only answered outcomes may be cached")
        return self


class AnswerCache(Model):
    """The file ``hoa-qa cache build`` writes and the app loads."""

    corpus_fingerprint: str = Field(min_length=1)
    built_at: AwareDatetime
    jev_model: str
    answer_model: str
    entries: tuple[CacheEntry, ...]

    @model_validator(mode="after")
    def unique(self) -> "AnswerCache":
        ids = [entry.id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate entry ids")
        return self


def load_cache(path: str | Path) -> AnswerCache:
    return AnswerCache.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_questions(path: Path) -> CuratedQuestions:
    # PyYAML ships in the ingest/dev dependency groups, not the web runtime.
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "the cache tools need PyYAML: `uv sync --group ingest` (or the dev group)"
        ) from exc
    return CuratedQuestions.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def corpus_fingerprint(corpus: Corpus) -> str:
    """Hash of everything an answer can depend on in the corpus.

    Covers each chunk's id, text, authority, dates and citation label, so a
    rebuilt corpus with the same content keeps its cache and any change to
    what the pipeline could read or cite invalidates it.
    """
    digest = hashlib.sha256()
    for chunk in corpus.chunks:
        record = [
            chunk.id,
            chunk.text_clean,
            chunk.authority.value,
            chunk.effective_date.isoformat() if chunk.effective_date else None,
            chunk.citation_label,
            chunk.source_url,
            chunk.superseded_by,
        ]
        digest.update(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


_NON_WORD = re.compile(r"[^\w$%]+")


def normalize_question(text: str) -> str:
    """Case, punctuation and spacing folded, for the exact-match shortcut."""
    return _NON_WORD.sub(" ", text.casefold()).strip()


_NUMBER_WORDS = {
    word: str(value)
    for value, words in enumerate(
        [
            ("zero",),
            ("one", "first", "once"),
            ("two", "second", "twice"),
            ("three", "third"),
            ("four", "fourth"),
            ("five", "fifth"),
            ("six", "sixth"),
            ("seven", "seventh"),
            ("eight", "eighth"),
            ("nine", "ninth"),
            ("ten", "tenth"),
            ("eleven", "eleventh"),
            ("twelve", "twelfth"),
        ]
    )
    for word in words
}
_NUMERIC = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z]+")


def number_signature(text: str) -> frozenset[str]:
    """The numbers a question mentions, as digits: "3rd", "third" -> "3".

    Deliberately broad ("second" counts even as a unit of time): a wrong
    reading here can only turn a hit into a miss.
    """
    folded = text.casefold().replace(",", "")
    numbers = {match.lstrip("0") or "0" for match in _NUMERIC.findall(folded)}
    numbers.update(
        _NUMBER_WORDS[word] for word in _WORD.findall(folded) if word in _NUMBER_WORDS
    )
    return frozenset(numbers)


def match_request(
    question: str, entries: Sequence[CacheEntry]
) -> tuple[dict[str, Any], dict[str, NoulQuestion]]:
    """Build the (state, questions) for one batch of cached entries.

    Entries are keyed by name (``known.k3``), not list position, and both the
    resident's question and the cached ones are sent as state (data), never
    inside the instructions.
    """
    state = {
        "question": question,
        "known": {f"k{i}": entry.question for i, entry in enumerate(entries)},
    }
    questions = {
        f"k{i}": NoulQuestion(
            instructions=(
                f"Would a complete answer to the known question `known.k{i}` "
                "also fully answer the resident's message in `question`? Judge "
                f"only `known.k{i}`, not the other known questions."
            ),
            yes=(
                "Both ask for the same information about the same thing: the "
                "same rule, fee, amenity, document, role, or situation, with "
                "the same specifics (which violation, which year or period, "
                "which amount, which kind of structure, vehicle, or pet). They "
                "may be worded differently."
            ),
            no=(
                "They are about different things or differ in any specific; "
                "or `question` asks for more than the known question does, is "
                "broader or narrower, adds a condition or a second question, "
                "or describes a particular dispute or personal situation; or "
                "`question` is not a question about the HOA at all."
            ),
        )
        for i in range(len(entries))
    }
    return state, questions


@dataclass(frozen=True)
class CacheLookup:
    """``entry`` is None on a miss. ``probability`` is the best score seen."""

    entry: CacheEntry | None
    probability: float | None
    input_tokens: int
    requests: int
    candidates: int
    error_type: str | None = None


MISS = CacheLookup(None, None, 0, 0, 0)


async def lookup(
    jev: JevClient,
    question: str,
    entries: Sequence[CacheEntry],
    *,
    threshold: float,
    limit: asyncio.Semaphore,
) -> CacheLookup:
    """Find the cached entry whose answer fully answers ``question``, if any.

    Never raises for a provider failure: a failed lookup is a miss that still
    reports the tokens it billed, so the cache can't make a request fail.
    """
    signature = number_signature(question)
    candidates = [e for e in entries if number_signature(e.question) == signature]
    if not candidates:
        return MISS
    batches = [
        candidates[start : start + MAX_ENTRIES_PER_REQUEST]
        for start in range(0, len(candidates), MAX_ENTRIES_PER_REQUEST)
    ]
    try:
        results = await run_nouls(
            jev, [match_request(question, batch) for batch in batches], limit
        )
    except PartialCostError as exc:
        return CacheLookup(
            None,
            None,
            exc.jev_input_tokens,
            len(batches),
            len(candidates),
            error_type=type(exc.errors[0]).__name__,
        )
    best: CacheEntry | None = None
    best_p = 0.0
    for batch, result in zip(batches, results, strict=True):
        for i, entry in enumerate(batch):
            p = result.probabilities[f"k{i}"]
            # Strictly greater: file order breaks ties deterministically.
            if best is None or p > best_p:
                best, best_p = entry, p
    return CacheLookup(
        entry=best if best_p >= threshold else None,
        probability=best_p,
        input_tokens=sum(r.input_tokens for r in results),
        requests=len(batches),
        candidates=len(candidates),
    )


class MatchCase(Model):
    """One check of the matcher: ``expect`` is an entry id, or None for a miss."""

    question: QuestionText
    expect: EntryId | None
    notes: str = ""


class MatchCases(Model):
    cases: tuple[MatchCase, ...] = Field(min_length=1)


def load_match_cases(path: Path) -> MatchCases:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "the cache tools need PyYAML: `uv sync --group ingest` (or the dev group)"
        ) from exc
    return MatchCases.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class MatchRow:
    case: MatchCase
    got: str | None
    probability: float | None

    @property
    def false_hit(self) -> bool:
        """Served a stored answer the case did not expect: the unsafe error."""
        return self.got is not None and self.got != self.case.expect

    @property
    def missed(self) -> bool:
        """Expected a hit and fell through to the pipeline: slower, not wrong."""
        return self.case.expect is not None and self.got is None


async def check_matches(
    jev: JevClient,
    cache: AnswerCache,
    cases: MatchCases,
    *,
    threshold: float,
    limit: asyncio.Semaphore,
) -> list[MatchRow]:
    exact = {normalize_question(e.question): e for e in cache.entries}
    rows: list[MatchRow] = []
    for case in cases.cases:
        entry = exact.get(normalize_question(case.question))
        if entry is not None:
            rows.append(MatchRow(case, entry.id, 1.0))
            continue
        found = await lookup(
            jev, case.question, cache.entries, threshold=threshold, limit=limit
        )
        rows.append(
            MatchRow(case, found.entry.id if found.entry else None, found.probability)
        )
    return rows

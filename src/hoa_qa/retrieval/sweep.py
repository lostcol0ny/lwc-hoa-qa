"""Relevance sweep: one Jev Noul per chunk, batched per document.

Each request carries one document's passages in its state and asks one
question per passage. Passages are keyed by their question's name
(``passages.p3``) and each question repeats the passage heading: the first
live eval showed the judge mis-indexing positional references
(``passages[30]``) in long lists, scoring a neighbor instead of the Rules'
fining schedule. For the same reason a request holds at most
``MAX_PASSAGES_PER_REQUEST`` passages.

``plan_passages`` decides once per corpus, sized for the longest possible
question, how each chunk is sent: whole, split into sub-passages (relevance is
the best part), or skipped. ``plan_batches`` then packs those passages per
document for the actual question; a document too large for one request, or
with more than MAX_PASSAGES_PER_REQUEST passages, is split across several.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoa_qa.models import Chunk
from hoa_qa.retrieval.jev import (
    DEFAULT_LIMITS,
    WORST_QUESTION,
    JevClient,
    JevLimits,
    NoulQuestion,
    TokenCounter,
    conservative_tokens,
    fits,
    pack,
    run_nouls,
)

logger = logging.getLogger(__name__)

# Below this many characters a passage is not split further.
MIN_SPLIT_CHARS = 64
# Passages judged in one request; smaller batches keep references reliable.
MAX_PASSAGES_PER_REQUEST = 8
# Headings repeated in a question are cut to this many characters.
MAX_HEADING_CHARS = 120


@dataclass(frozen=True)
class SweepPassage:
    """A chunk, or one part of a chunk too large to judge in one request."""

    chunk: Chunk
    text: str
    part: int = 0
    parts: int = 1


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    probability: float


@dataclass(frozen=True)
class SweepResult:
    selected: tuple[ScoredChunk, ...]
    input_tokens: int
    requests: int
    # Every judged chunk, best first (for eval diagnostics).
    scores: tuple[ScoredChunk, ...] = ()


@dataclass(frozen=True)
class SweepPlan:
    batches: list[list[SweepPassage]]
    skipped: list[str]


def batch_request(
    question: str, passages: Sequence[SweepPassage]
) -> tuple[dict[str, Any], dict[str, NoulQuestion]]:
    """Build the (state, questions) for one batch of same-document passages."""
    state = {
        "question": question,
        "document": passages[0].chunk.doc_title if passages else "",
        "passages": {
            f"p{i}": {"heading": _heading(p), "text": p.text}
            for i, p in enumerate(passages)
        },
    }
    questions = {
        f"p{i}": NoulQuestion(
            instructions=(
                f"Does the passage `passages.p{i}` (heading: "
                f"{_heading(p)[:MAX_HEADING_CHARS]!r}) help answer the question "
                "in `question`? Judge only `passages.p"
                f"{i}`, not the other passages."
            ),
            yes=(
                "The passage contains information that is part of the answer, "
                "even if it answers only part of the question (for example, "
                "one figure in a series or one step of a process)."
            ),
            no="The passage is unrelated or does not help answer the question.",
        )
        for i, p in enumerate(passages)
    }
    return state, questions


def _heading(passage: SweepPassage) -> str:
    heading = " > ".join(passage.chunk.heading_path)
    if passage.parts > 1:
        heading += f" (part {passage.part + 1} of {passage.parts})"
    return heading


def split_chunk(
    chunk: Chunk,
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
    worst_question: str = WORST_QUESTION,
) -> list[SweepPassage] | None:
    """Split a chunk into parts that each fit a request alone; None if impossible.

    Sized for the longest possible question, so the result does not depend on
    the question actually asked.
    """

    def fits_alone(text: str) -> bool:
        # Label as a multi-part passage so the size check includes the suffix.
        probe = SweepPassage(chunk, text, part=9_999, parts=9_999)
        return fits(*batch_request(worst_question, [probe]), count_tokens, limits)

    def pieces(text: str) -> list[str] | None:
        if fits_alone(text):
            return [text]
        if len(text) <= MIN_SPLIT_CHARS:
            return None
        mid = len(text) // 2
        space = text.rfind(" ", 0, mid)
        cut = space if space > len(text) // 4 else mid
        left, right = pieces(text[:cut]), pieces(text[cut:])
        return None if left is None or right is None else left + right

    texts = pieces(chunk.text_clean)
    if texts is None:
        return None
    return [
        SweepPassage(chunk, t, part=i, parts=len(texts)) for i, t in enumerate(texts)
    ]


@dataclass(frozen=True)
class PassagePlan:
    """Every passage the sweep will judge, in corpus order, and skipped chunks."""

    passages: tuple[SweepPassage, ...]
    skipped: tuple[str, ...]


def plan_passages(
    chunks: Sequence[Chunk],
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
    worst_question: str = WORST_QUESTION,
) -> PassagePlan:
    """Decide once, independent of the question, how each chunk is sent.

    A chunk that fits alone (with the longest possible question) is one
    passage; a larger one is split into sub-passages; one that cannot be split
    small enough is skipped and logged by chunk_id. Because this never depends
    on the question, ``max_cost_usd`` can account for exactly these passages.
    """
    passages: list[SweepPassage] = []
    skipped: list[str] = []
    for chunk in chunks:
        whole = SweepPassage(chunk, chunk.text_clean)
        if fits(*batch_request(worst_question, [whole]), count_tokens, limits):
            passages.append(whole)
            continue
        parts = split_chunk(chunk, count_tokens, limits, worst_question)
        if parts is None:
            logger.warning("sweep skipped oversized chunk_id=%s", chunk.id)
            skipped.append(chunk.id)
            continue
        passages.extend(parts)
    return PassagePlan(passages=tuple(passages), skipped=tuple(skipped))


def plan_batches(
    question: str,
    passages: Sequence[SweepPassage],
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> SweepPlan:
    """Pack planned passages into requests, one document per request."""
    by_doc: dict[str, list[SweepPassage]] = {}
    for passage in passages:
        by_doc.setdefault(passage.chunk.doc_id, []).append(passage)
    batches: list[list[SweepPassage]] = []
    skipped: list[str] = []
    for doc_passages in by_doc.values():
        groups, oversized = pack(
            doc_passages,
            lambda group: batch_request(question, group),
            count_tokens,
            limits,
            max_items=MAX_PASSAGES_PER_REQUEST,
        )
        # Planned passages fit alone with the longest question, so oversized
        # is empty; if the invariant ever broke, fail closed rather than send.
        for passage in oversized:
            logger.warning("sweep skipped oversized chunk_id=%s", passage.chunk.id)
            skipped.append(passage.chunk.id)
        batches.extend(groups)
    return SweepPlan(batches=batches, skipped=skipped)


async def sweep(
    jev: JevClient,
    question: str,
    passages: Sequence[SweepPassage],
    *,
    top_k: int,
    threshold: float,
    limit: asyncio.Semaphore,
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> SweepResult:
    """Score every planned passage; keep the top ``top_k`` chunks >= ``threshold``.

    A chunk split into parts scores as its best part. Raises
    ``PartialCostError`` (after every batch has finished) if any batch failed,
    carrying every billed token.
    """
    plan = plan_batches(question, passages, count_tokens, limits)
    results = await run_nouls(
        jev, [batch_request(question, batch) for batch in plan.batches], limit
    )
    best: dict[str, float] = {}
    for batch, result in zip(plan.batches, results, strict=True):
        for i, passage in enumerate(batch):
            p = result.probabilities[f"p{i}"]
            best[passage.chunk.id] = max(p, best.get(passage.chunk.id, 0.0))
    order: dict[str, int] = {}
    by_id: dict[str, Chunk] = {}
    for passage in passages:
        order.setdefault(passage.chunk.id, len(order))
        by_id[passage.chunk.id] = passage.chunk
    scores = [ScoredChunk(by_id[chunk_id], p) for chunk_id, p in best.items()]
    # Highest probability first; corpus order breaks ties deterministically.
    scores.sort(key=lambda s: (-s.probability, order[s.chunk.id]))
    passing = [s for s in scores if s.probability >= threshold]
    return SweepResult(
        selected=tuple(passing[: max(0, top_k)]),
        input_tokens=sum(r.input_tokens for r in results),
        requests=len(plan.batches),
        scores=tuple(scores),
    )

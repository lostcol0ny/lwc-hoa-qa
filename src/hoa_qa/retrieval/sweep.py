"""Relevance sweep: one Jev Noul per chunk, batched per document.

Each request carries one document's passages in its state and asks one
question per passage. A document too large for one request is split across
several; a single chunk too large for any request is split into sub-passages
(its relevance is the best of its parts). Nothing over a Jev limit is sent.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoa_qa.models import Chunk
from hoa_qa.retrieval.jev import (
    DEFAULT_LIMITS,
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
        "passages": [{"heading": _heading(p), "text": p.text} for p in passages],
    }
    questions = {
        f"p{i}": NoulQuestion(
            instructions=(
                f"Does this passage, `passages[{i}]`, help answer the question "
                "in `question`?"
            ),
            yes="The passage contains information needed to answer the question.",
            no="The passage is unrelated or does not help answer the question.",
        )
        for i in range(len(passages))
    }
    return state, questions


def _heading(passage: SweepPassage) -> str:
    heading = " > ".join(passage.chunk.heading_path)
    if passage.parts > 1:
        heading += f" (part {passage.part + 1} of {passage.parts})"
    return heading


def split_chunk(
    question: str,
    chunk: Chunk,
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> list[SweepPassage] | None:
    """Split a chunk into parts that each fit a request alone; None if impossible."""

    def fits_alone(text: str) -> bool:
        # Label as a multi-part passage so the size check includes the suffix.
        probe = SweepPassage(chunk, text, part=9_999, parts=9_999)
        return fits(*batch_request(question, [probe]), count_tokens, limits)

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


def plan_batches(
    question: str,
    chunks: Sequence[Chunk],
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> SweepPlan:
    """Group chunks by ``doc_id`` (corpus order), splitting anything over a limit."""
    by_doc: dict[str, list[SweepPassage]] = {}
    skipped: list[str] = []
    for chunk in chunks:
        whole = SweepPassage(chunk, chunk.text_clean)
        if fits(*batch_request(question, [whole]), count_tokens, limits):
            by_doc.setdefault(chunk.doc_id, []).append(whole)
            continue
        parts = split_chunk(question, chunk, count_tokens, limits)
        if parts is None:
            logger.warning("sweep skipped oversized chunk_id=%s", chunk.id)
            skipped.append(chunk.id)
            continue
        by_doc.setdefault(chunk.doc_id, []).extend(parts)
    batches: list[list[SweepPassage]] = []
    for passages in by_doc.values():
        groups, oversized = pack(
            passages,
            lambda group: batch_request(question, group),
            count_tokens,
            limits,
        )
        # Every passage here was checked to fit alone, so oversized is empty;
        # if the invariant ever broke, fail closed rather than send it.
        for passage in oversized:
            logger.warning("sweep skipped oversized chunk_id=%s", passage.chunk.id)
            skipped.append(passage.chunk.id)
        batches.extend(groups)
    return SweepPlan(batches=batches, skipped=skipped)


async def sweep(
    jev: JevClient,
    question: str,
    chunks: Sequence[Chunk],
    *,
    top_k: int,
    threshold: float,
    limit: asyncio.Semaphore,
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> SweepResult:
    """Score every chunk and keep the top ``top_k`` at or above ``threshold``.

    Raises ``PartialCostError`` (after every batch has finished) if any batch
    failed, carrying the tokens the successful batches billed.
    """
    plan = plan_batches(question, chunks, count_tokens, limits)
    results = await run_nouls(
        jev, [batch_request(question, batch) for batch in plan.batches], limit
    )
    best: dict[str, float] = {}
    for batch, result in zip(plan.batches, results, strict=True):
        for i, passage in enumerate(batch):
            p = result.probabilities[f"p{i}"]
            best[passage.chunk.id] = max(p, best.get(passage.chunk.id, 0.0))
    order = {chunk.id: i for i, chunk in enumerate(chunks)}
    by_id = {chunk.id: chunk for chunk in chunks}
    passing = [
        ScoredChunk(by_id[chunk_id], p)
        for chunk_id, p in best.items()
        if p >= threshold
    ]
    # Highest probability first; corpus order breaks ties deterministically.
    passing.sort(key=lambda s: (-s.probability, order[s.chunk.id]))
    return SweepResult(
        selected=tuple(passing[: max(0, top_k)]),
        input_tokens=sum(r.input_tokens for r in results),
        requests=len(plan.batches),
    )

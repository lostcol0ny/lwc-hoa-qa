"""Relevance sweep: one Jev Noul per chunk, batched per document.

Each request carries one document's passages in its state and asks one
question per passage, so Jev reads the shared question once per batch. A
document too large for one request is split across several.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hoa_qa.models import Chunk
from hoa_qa.retrieval.jev import (
    MAX_REQUEST_TOKENS,
    MAX_STATE_TOKENS,
    JevClient,
    NoulQuestion,
    TokenCounter,
    estimate_tokens,
    pack,
)


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    probability: float


@dataclass(frozen=True)
class SweepResult:
    selected: tuple[ScoredChunk, ...]
    input_tokens: int
    requests: int


def batch_request(
    question: str, chunks: Sequence[Chunk]
) -> tuple[dict[str, Any], dict[str, NoulQuestion]]:
    """Build the (state, questions) for one batch of same-document chunks."""
    state = {
        "question": question,
        "document": chunks[0].doc_title if chunks else "",
        "passages": [
            {"heading": " > ".join(c.heading_path), "text": c.text_clean}
            for c in chunks
        ],
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
        for i in range(len(chunks))
    }
    return state, questions


def plan_batches(
    question: str,
    chunks: Sequence[Chunk],
    count_tokens: TokenCounter = estimate_tokens,
    *,
    max_request_tokens: int = MAX_REQUEST_TOKENS,
    max_state_tokens: int = MAX_STATE_TOKENS,
) -> list[list[Chunk]]:
    """Group chunks by ``doc_id`` (corpus order), splitting docs over the limit."""
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)
    batches: list[list[Chunk]] = []
    for doc_chunks in by_doc.values():
        batches.extend(
            pack(
                doc_chunks,
                lambda group: batch_request(question, group),
                count_tokens,
                max_request_tokens=max_request_tokens,
                max_state_tokens=max_state_tokens,
            )
        )
    return batches


async def sweep(
    jev: JevClient,
    question: str,
    chunks: Sequence[Chunk],
    *,
    top_k: int,
    threshold: float,
    concurrency: int,
    count_tokens: TokenCounter = estimate_tokens,
) -> SweepResult:
    """Score every chunk and keep the top ``top_k`` at or above ``threshold``."""
    batches = plan_batches(question, chunks, count_tokens)
    limit = asyncio.Semaphore(max(1, concurrency))

    async def score(batch: list[Chunk]) -> tuple[list[ScoredChunk], int]:
        state, questions = batch_request(question, batch)
        async with limit:
            result = await jev.nouls(state, questions)
        return _scored(batch, result.probabilities), result.input_tokens

    results = await asyncio.gather(*(score(batch) for batch in batches))
    order = {chunk.id: i for i, chunk in enumerate(chunks)}
    scored = [item for batch_scores, _ in results for item in batch_scores]
    passing = [s for s in scored if s.probability >= threshold]
    # Highest probability first; corpus order breaks ties deterministically.
    passing.sort(key=lambda s: (-s.probability, order[s.chunk.id]))
    return SweepResult(
        selected=tuple(passing[: max(0, top_k)]),
        input_tokens=sum(tokens for _, tokens in results),
        requests=len(batches),
    )


def _scored(
    batch: Sequence[Chunk], probabilities: Mapping[str, float]
) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=chunk, probability=probabilities[f"p{i}"])
        for i, chunk in enumerate(batch)
    ]

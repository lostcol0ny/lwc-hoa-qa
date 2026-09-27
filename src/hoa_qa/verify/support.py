"""Jev support check: does each claim's cited evidence support THAT claim?

One Noul per claim, over the claim's statement and the distinct passages its
quote-checked citations point to. Claims are batched under the Jev limits and
the shared concurrency bound. A claim with no quote-checked citation, or too
large to judge in any single request, fails closed without a call.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoa_qa.retrieval.jev import (
    DEFAULT_LIMITS,
    JevClient,
    JevLimits,
    NoulQuestion,
    TokenCounter,
    conservative_tokens,
    pack,
    run_nouls,
)
from hoa_qa.verify.quotes import CheckedCitation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimEvidence:
    """A claim's statement and its citations that passed the quote check."""

    statement: str
    citations: tuple[CheckedCitation, ...]


@dataclass(frozen=True)
class SupportResult:
    supported: tuple[bool, ...]  # one per claim, in input order
    input_tokens: int
    requests: int
    # The judged probability per claim; None where no request was made.
    probabilities: tuple[float | None, ...] = ()


def _passages(evidence: ClaimEvidence) -> list[str]:
    seen: dict[str, str] = {}
    for citation in evidence.citations:
        seen.setdefault(citation.chunk.id, citation.chunk.text_clean)
    return list(seen.values())


def support_request(
    claims: Sequence[ClaimEvidence],
) -> tuple[dict[str, Any], dict[str, NoulQuestion]]:
    state = {
        "claims": [{"statement": c.statement, "passages": _passages(c)} for c in claims]
    }
    questions = {
        f"c{i}": NoulQuestion(
            instructions=(
                f"Do the passages in `claims[{i}].passages` support this specific "
                f"claim, `claims[{i}].statement`?"
            ),
            yes=(
                "Everything the statement asserts is stated in the passages, "
                "including every amount, date, condition, and exception."
            ),
            no=(
                "Some part of the statement is missing from the passages, goes "
                "beyond them, or contradicts them."
            ),
        )
        for i in range(len(claims))
    }
    return state, questions


async def check_support(
    jev: JevClient,
    claims: Sequence[ClaimEvidence],
    *,
    threshold: float,
    limit: asyncio.Semaphore,
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> SupportResult:
    """Judge every claim; ``supported[i]`` is False below ``threshold``.

    Raises ``PartialCostError`` (after every batch has finished) if any batch
    failed, carrying the tokens the successful batches billed.
    """
    indexed = [(i, c) for i, c in enumerate(claims) if c.citations]
    groups, oversized = pack(
        indexed,
        lambda group: support_request([c for _, c in group]),
        count_tokens,
        limits,
    )
    for index, _ in oversized:
        logger.warning("support check skipped oversized claim index=%d", index)
    results = await run_nouls(
        jev, [support_request([c for _, c in group]) for group in groups], limit
    )
    supported = [False] * len(claims)
    probabilities: list[float | None] = [None] * len(claims)
    for group, result in zip(groups, results, strict=True):
        for position, (index, _) in enumerate(group):
            p = result.probabilities[f"c{position}"]
            probabilities[index] = p
            supported[index] = p >= threshold
    return SupportResult(
        supported=tuple(supported),
        input_tokens=sum(r.input_tokens for r in results),
        requests=len(groups),
        probabilities=tuple(probabilities),
    )

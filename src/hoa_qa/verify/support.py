"""Jev support check: does each cited passage actually back the answer?"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from hoa_qa.retrieval.jev import (
    JevClient,
    NoulQuestion,
    TokenCounter,
    estimate_tokens,
    pack,
)
from hoa_qa.verify.quotes import CheckedCitation


@dataclass(frozen=True)
class SupportResult:
    kept: tuple[CheckedCitation, ...]
    input_tokens: int


def support_request(
    claim: str, citations: Sequence[CheckedCitation]
) -> tuple[dict[str, Any], dict[str, NoulQuestion]]:
    state = {
        "claim": claim,
        "citations": [
            {"passage": c.chunk.text_clean, "quote": c.quote} for c in citations
        ],
    }
    questions = {
        f"c{i}": NoulQuestion(
            instructions=(
                f"Does this passage, `citations[{i}].passage`, support this "
                "claim, `claim`?"
            ),
            yes="The passage states facts that back up at least part of the claim.",
            no="The passage is unrelated to the claim or does not back it up.",
        )
        for i in range(len(citations))
    }
    return state, questions


async def check_support(
    jev: JevClient,
    claim: str,
    citations: Sequence[CheckedCitation],
    *,
    threshold: float,
    count_tokens: TokenCounter = estimate_tokens,
) -> SupportResult:
    """Drop citations whose support probability is below ``threshold``."""
    if not citations:
        return SupportResult(kept=(), input_tokens=0)
    groups = pack(citations, lambda group: support_request(claim, group), count_tokens)
    results = await asyncio.gather(
        *(jev.nouls(*support_request(claim, group)) for group in groups)
    )
    kept = [
        citation
        for group, result in zip(groups, results, strict=True)
        for i, citation in enumerate(group)
        if result.probabilities[f"c{i}"] >= threshold
    ]
    return SupportResult(
        kept=tuple(kept), input_tokens=sum(r.input_tokens for r in results)
    )

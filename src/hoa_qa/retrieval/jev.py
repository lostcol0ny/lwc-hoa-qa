"""Thin wrapper around the TypeSafe SDK; the only module that imports ``typesafe_sdk``.

Callers describe yes/no judgments as ``NoulQuestion`` values and get back plain
probabilities plus the billed input tokens, so everything above this module can
be tested with a fake ``JevClient``.
"""

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Jev 1.13 limits (docs.typesafe.ai/models, read 2026-09-27): 64K tokens for the
# state plus every question, and 32K for the state plus the longest question.
MAX_REQUEST_TOKENS = 64_000
MAX_STATE_TOKENS = 32_000

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (~4 characters per token, rounded up)."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class NoulQuestion:
    """A yes/no judgment; ``yes``/``no`` optionally describe each outcome."""

    instructions: str
    yes: str | None = None
    no: str | None = None


@dataclass(frozen=True)
class NoulBatchResult:
    """Probabilities of "yes" keyed by question name, plus billed input tokens."""

    probabilities: Mapping[str, float]
    input_tokens: int


class JevClient(Protocol):
    async def nouls(
        self, state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> NoulBatchResult:
        """Ask every question against one shared state in a single request."""
        ...


def state_json(state: Mapping[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def request_tokens(
    state: Mapping[str, Any],
    questions: Mapping[str, NoulQuestion],
    count_tokens: TokenCounter = estimate_tokens,
) -> tuple[int, int]:
    """Return (state + all questions, state + longest question) token estimates."""
    state_tokens = count_tokens(state_json(state))
    sizes = [
        count_tokens(q.instructions + (q.yes or "") + (q.no or ""))
        for q in questions.values()
    ]
    return state_tokens + sum(sizes), state_tokens + max(sizes, default=0)


def fits(
    state: Mapping[str, Any],
    questions: Mapping[str, NoulQuestion],
    count_tokens: TokenCounter = estimate_tokens,
    *,
    max_request_tokens: int = MAX_REQUEST_TOKENS,
    max_state_tokens: int = MAX_STATE_TOKENS,
) -> bool:
    total, state_plus_longest = request_tokens(state, questions, count_tokens)
    return total <= max_request_tokens and state_plus_longest <= max_state_tokens


def pack[T](
    items: Sequence[T],
    build: Callable[
        [Sequence[T]], tuple[Mapping[str, Any], Mapping[str, NoulQuestion]]
    ],
    count_tokens: TokenCounter = estimate_tokens,
    *,
    max_request_tokens: int = MAX_REQUEST_TOKENS,
    max_state_tokens: int = MAX_STATE_TOKENS,
) -> list[list[T]]:
    """Greedily split ``items`` into consecutive groups whose request fits.

    ``build`` turns a group into the (state, questions) it would send. An item
    that does not fit even on its own still gets a group of one, so nothing is
    silently dropped; the API rejects it and the caller sees the error.
    """
    groups: list[list[T]] = []
    current: list[T] = []
    for item in items:
        candidate = [*current, item]
        if fits(
            *build(candidate),
            count_tokens,
            max_request_tokens=max_request_tokens,
            max_state_tokens=max_state_tokens,
        ):
            current = candidate
            continue
        if current:
            groups.append(current)
        current = [item]
        if not fits(
            *build(current),
            count_tokens,
            max_request_tokens=max_request_tokens,
            max_state_tokens=max_state_tokens,
        ):
            logger.warning("jev item exceeds the per-request token limit on its own")
    if current:
        groups.append(current)
    return groups


class TypeSafeJevClient:
    """``JevClient`` backed by ``typesafe_sdk.AsyncTypeSafeClient``."""

    def __init__(self, *, api_key: str, model: str) -> None:
        from typesafe_sdk import AsyncTypeSafeClient

        self._model = model
        self._client = AsyncTypeSafeClient(api_key=api_key, model=model)

    async def nouls(
        self, state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
    ) -> NoulBatchResult:
        from typesafe_sdk import Noul, NoulCriteria

        sdk_questions: dict[str, Noul] = {}
        for name, q in questions.items():
            criteria: NoulCriteria | None = None
            if q.yes is not None or q.no is not None:
                criteria = {"true": q.yes, "false": q.no}
            sdk_questions[name] = Noul(instructions=q.instructions, criteria=criteria)
        response = await self._client.system_one(dict(state), sdk_questions)
        missing = questions.keys() - response.nouls.keys()
        if missing:
            raise RuntimeError(f"Jev response is missing {len(missing)} answers")
        input_tokens = response.usage.input_tokens
        if input_tokens is None:
            # Usage is optional in the API contract; fall back to our estimate so
            # the spend counter never under-reports to zero.
            input_tokens = request_tokens(state, questions)[0]
        return NoulBatchResult(
            probabilities={name: response.nouls[name].noul for name in questions},
            input_tokens=input_tokens,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

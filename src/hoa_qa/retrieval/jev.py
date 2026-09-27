"""Thin wrapper around the TypeSafe SDK; the only module that imports ``typesafe_sdk``.

Callers describe yes/no judgments as ``NoulQuestion`` values and get back plain
probabilities plus the billed input tokens, so everything above this module can
be tested with a fake ``JevClient``.

Request sizing. The SDK exposes no tokenizer or token-counting endpoint (only
``/v1/systemone`` and ``/v1/models``), so sizes use a deliberately conservative
bound: one token per ``ASCII_BYTES_PER_TOKEN`` ASCII bytes (about 1.6-1.8x the
~4-4.5 characters/token typical of English) plus one token per non-ASCII
UTF-8 byte (the byte-fallback worst case), plus fixed per-request and
per-question overheads for the JSON envelope, question names, and criteria
keys. ``pack`` never emits a group over either Jev limit.
"""

import asyncio
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

ASCII_BYTES_PER_TOKEN = 2.5
REQUEST_OVERHEAD_TOKENS = 256
QUESTION_OVERHEAD_TOKENS = 32

TokenCounter = Callable[[str], int]


def conservative_tokens(text: str) -> int:
    """Conservative token count: ASCII bytes / 2.5 + one per non-ASCII byte."""
    ascii_bytes = sum(1 for ch in text if ord(ch) < 128)
    other_bytes = len(text.encode("utf-8")) - ascii_bytes
    return math.ceil(ascii_bytes / ASCII_BYTES_PER_TOKEN) + other_bytes


@dataclass(frozen=True)
class JevLimits:
    max_request_tokens: int = MAX_REQUEST_TOKENS
    max_state_tokens: int = MAX_STATE_TOKENS


DEFAULT_LIMITS = JevLimits()


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


class PartialCostError(Exception):
    """Some Jev requests failed; carries the tokens the successful ones billed."""

    def __init__(self, jev_input_tokens: int, errors: Sequence[BaseException]):
        super().__init__(
            f"{len(errors)} Jev request(s) failed: "
            + ", ".join(type(e).__name__ for e in errors)
        )
        self.jev_input_tokens = jev_input_tokens
        self.errors = tuple(errors)


def state_json(state: Mapping[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def request_tokens(
    state: Mapping[str, Any],
    questions: Mapping[str, NoulQuestion],
    count_tokens: TokenCounter = conservative_tokens,
) -> tuple[int, int]:
    """Return (whole request, state + longest question) token bounds."""
    state_tokens = count_tokens(state_json(state))
    sizes = [
        QUESTION_OVERHEAD_TOKENS
        + count_tokens(name + q.instructions + (q.yes or "") + (q.no or ""))
        for name, q in questions.items()
    ]
    return (
        REQUEST_OVERHEAD_TOKENS + state_tokens + sum(sizes),
        REQUEST_OVERHEAD_TOKENS + state_tokens + max(sizes, default=0),
    )


def fits(
    state: Mapping[str, Any],
    questions: Mapping[str, NoulQuestion],
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> bool:
    total, state_plus_longest = request_tokens(state, questions, count_tokens)
    return (
        total <= limits.max_request_tokens
        and state_plus_longest <= limits.max_state_tokens
    )


def pack[T](
    items: Sequence[T],
    build: Callable[
        [Sequence[T]], tuple[Mapping[str, Any], Mapping[str, NoulQuestion]]
    ],
    count_tokens: TokenCounter = conservative_tokens,
    limits: JevLimits = DEFAULT_LIMITS,
) -> tuple[list[list[T]], list[T]]:
    """Greedily split ``items`` into consecutive groups whose request fits.

    Returns ``(groups, oversized)``. Every group is within both limits; items
    that do not fit even alone are returned in ``oversized`` and never sent.
    """
    groups: list[list[T]] = []
    oversized: list[T] = []
    current: list[T] = []
    for item in items:
        if not fits(*build([item]), count_tokens, limits):
            oversized.append(item)
            continue
        candidate = [*current, item]
        if fits(*build(candidate), count_tokens, limits):
            current = candidate
        else:
            groups.append(current)
            current = [item]
    if current:
        groups.append(current)
    return groups, oversized


async def run_nouls(
    jev: JevClient,
    requests: Sequence[tuple[Mapping[str, Any], Mapping[str, NoulQuestion]]],
    limit: asyncio.Semaphore,
) -> list[NoulBatchResult]:
    """Run every request to completion under ``limit``.

    If any fail, the rest still finish, and ``PartialCostError`` reports the
    tokens the successful ones billed so the caller can still count them.
    """

    async def one(
        request: tuple[Mapping[str, Any], Mapping[str, NoulQuestion]],
    ) -> NoulBatchResult:
        async with limit:
            return await jev.nouls(*request)

    outcomes = await asyncio.gather(*(one(r) for r in requests), return_exceptions=True)
    for outcome in outcomes:
        # Cancellation and interpreter exits are not request failures.
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
            raise outcome
    results = [o for o in outcomes if isinstance(o, NoulBatchResult)]
    errors = [o for o in outcomes if isinstance(o, Exception)]
    if errors:
        raise PartialCostError(
            sum(r.input_tokens for r in results), errors
        ) from errors[0]
    return results


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
            # Usage is optional in the API contract; fall back to our bound so
            # the spend counter never under-reports to zero.
            input_tokens = request_tokens(state, questions)[0]
        return NoulBatchResult(
            probabilities={name: response.nouls[name].noul for name in questions},
            input_tokens=input_tokens,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

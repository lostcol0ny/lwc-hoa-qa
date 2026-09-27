"""Thin wrapper around the TypeSafe SDK; the only module that imports ``typesafe_sdk``.

Callers describe yes/no judgments as ``NoulQuestion`` values and get back plain
probabilities plus the billed input tokens, so everything above this module can
be tested with a fake ``JevClient``.

The SDK exposes no tokenizer or token-counting endpoint (only
``/v1/systemone`` and ``/v1/models``), so there are two estimators:

- ``conservative_tokens`` is a *packing heuristic* for the Jev request limits:
  ASCII bytes / ``ASCII_BYTES_PER_TOKEN`` plus one token per non-ASCII byte,
  plus fixed per-request and per-question overheads. ``pack`` never emits a
  group over either limit under it. Text that tokenizes worse than the
  heuristic can still exceed a limit; the API then rejects the request, which
  fails closed (``error`` outcome, partial cost counted), never overspends.
- ``byte_bound_tokens`` is a *provable upper bound* used for money
  (``max_cost_usd``): at most one token per UTF-8 byte of the request body
  (byte-level BPE and SentencePiece with byte fallback both emit tokens that
  each cover at least one byte), plus bounded framing overhead per request
  and per question.
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
# Framing the provider adds around the request body (prompt template, model
# field, special tokens), per request and per question.
REQUEST_OVERHEAD_TOKENS = 256
QUESTION_OVERHEAD_TOKENS = 32
# Bound-only slack per question for question-name/index digits when passages
# are packed together (``p12345`` vs ``p0``, three occurrences per question).
INDEX_SLACK_TOKENS = 16

# Longest question ask() accepts, and the worst case for every estimator:
# 4 UTF-8 bytes per character, which is the maximum.
MAX_QUESTION_CHARS = 500
WORST_QUESTION = "\U0001d538" * MAX_QUESTION_CHARS

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
    """Some Jev work failed; carries every input token that was still billed."""

    def __init__(self, jev_input_tokens: int, errors: Sequence[BaseException]):
        super().__init__(
            f"{len(errors)} Jev request(s) failed: "
            + ", ".join(type(e).__name__ for e in errors)
        )
        self.jev_input_tokens = jev_input_tokens
        self.errors = tuple(errors)


def state_json(state: Mapping[str, Any]) -> str:
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _noul_wire(q: NoulQuestion) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": q.instructions,
        "criteria": {"true": q.yes, "false": q.no},
    }


def byte_bound_tokens(
    state: Mapping[str, Any], questions: Mapping[str, NoulQuestion]
) -> int:
    """Provable upper bound on one request's billed input tokens.

    UTF-8 bytes of the request body (serialized with spaced separators and
    criteria always present, so never smaller than the SDK's compact body),
    plus framing per request and per question.
    """
    body = {
        "state": state,
        "model": "m" * 64,
        "questions": {name: _noul_wire(q) for name, q in questions.items()},
    }
    body_bytes = len(json.dumps(body, ensure_ascii=False).encode("utf-8"))
    return (
        body_bytes
        + REQUEST_OVERHEAD_TOKENS
        + len(questions) * (QUESTION_OVERHEAD_TOKENS + INDEX_SLACK_TOKENS)
    )


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
    max_items: int | None = None,
) -> tuple[list[list[T]], list[T]]:
    """Greedily split ``items`` into consecutive groups whose request fits.

    Returns ``(groups, oversized)``. Every group is within both limits (and
    holds at most ``max_items`` items, if given); items that do not fit even
    alone are returned in ``oversized`` and never sent.
    """
    groups: list[list[T]] = []
    oversized: list[T] = []
    current: list[T] = []
    for item in items:
        if not fits(*build([item]), count_tokens, limits):
            oversized.append(item)
            continue
        candidate = [*current, item]
        if (max_items is None or len(candidate) <= max_items) and fits(
            *build(candidate), count_tokens, limits
        ):
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

    If any fail, the rest still finish, and ``PartialCostError`` reports every
    billed token: the successful ones plus any a failed request still billed
    (a nested ``PartialCostError``, e.g. a malformed but billed response).
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
    failures = [o for o in outcomes if isinstance(o, Exception)]
    if failures:
        billed = sum(r.input_tokens for r in results)
        errors: list[BaseException] = []
        for failure in failures:
            if isinstance(failure, PartialCostError):
                billed += failure.jev_input_tokens
                errors.extend(failure.errors)
            else:
                errors.append(failure)
        raise PartialCostError(billed, errors) from failures[0]
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
        # Read usage first: a malformed response was still billed.
        input_tokens = response.usage.input_tokens
        if input_tokens is None:
            # Usage is optional in the API contract; fall back to the provable
            # bound so the spend counter never under-reports.
            input_tokens = byte_bound_tokens(state, questions)
        missing = questions.keys() - response.nouls.keys()
        if missing:
            raise PartialCostError(
                input_tokens,
                [RuntimeError(f"Jev response is missing {len(missing)} answers")],
            )
        return NoulBatchResult(
            probabilities={name: response.nouls[name].noul for name in questions},
            input_tokens=input_tokens,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

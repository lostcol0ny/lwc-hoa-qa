"""Monthly spend cap and the shared counter stores behind it.

The web app checks the month's spend before calling the asker and adds each
request's estimated cost afterwards. Counters live in Upstash Redis in
production so every function instance shares one total; in development they
live in process memory.

Reservations: before each asker call the app atomically *reserves* a
worst-case cost R, and only if the month's total plus R stays within the
budget. After the call it reconciles by adding ``actual - R`` (often negative).
Any failure to reserve refuses the question; a failed reconcile keeps the
reservation, so accounting errors only ever over-count spend.

Fail-closed rule: when ``VERCEL_ENV=production`` and Upstash is not configured,
``select_budget_store`` returns a store that always reports the budget as
spent. A misconfigured production deployment refuses questions instead of
running with no cap.
"""

import asyncio
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

BUDGET_TTL_SECONDS = 40 * 24 * 60 * 60
"""Month keys outlive their month by ~10 days, then Redis deletes them."""

BUDGET_EPSILON_USD = 1e-9
"""Float slack so a budget of exactly 3 x R admits three reservations of R."""

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def month_key(now: datetime) -> str:
    """Return the per-calendar-month (UTC) budget key, e.g. ``budget:2026-09``."""
    return f"budget:{now.astimezone(UTC):%Y-%m}"


class CounterStore(Protocol):
    """Numeric counters with expiry; shared by the budget and the rate limiter."""

    async def get(self, key: str) -> float: ...

    async def incr(self, key: str, amount: float, ttl_seconds: int) -> float:
        """Atomically add ``amount`` and (re)set the key's TTL; return the total."""
        ...

    async def reserve(
        self, key: str, amount: float, limit: float, ttl_seconds: int
    ) -> float | None:
        """Atomically add ``amount`` only if the new total stays within ``limit``.

        Return the new total, or None (nothing added) if it would exceed it.
        """
        ...


@dataclass(frozen=True)
class Reservation:
    """Spend held for one in-flight request, pinned to the month it was made in."""

    key: str
    amount_usd: float


class BudgetStore(Protocol):
    async def get_month_spend(self) -> float: ...

    async def add_spend(self, usd: float) -> float: ...

    async def reserve(self, usd: float, budget_usd: float) -> Reservation | None:
        """Hold ``usd`` against this month's budget; None if it doesn't fit."""
        ...

    async def reconcile(self, reservation: Reservation, actual_usd: float) -> float:
        """Replace the held amount with the actual cost (adds ``actual - held``)."""
        ...


class InMemoryCounterStore:
    """Process-local counters for development and tests."""

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._values: dict[str, tuple[float, float]] = {}
        self._monotonic = monotonic
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> float:
        entry = self._values.get(key)
        if entry is None or entry[1] <= self._monotonic():
            self._values.pop(key, None)
            return 0.0
        return entry[0]

    async def incr(self, key: str, amount: float, ttl_seconds: int) -> float:
        async with self._lock:
            return self._add(key, amount, ttl_seconds)

    async def reserve(
        self, key: str, amount: float, limit: float, ttl_seconds: int
    ) -> float | None:
        async with self._lock:
            if await self.get(key) + amount > limit + BUDGET_EPSILON_USD:
                return None
            return self._add(key, amount, ttl_seconds)

    def _add(self, key: str, amount: float, ttl_seconds: int) -> float:
        entry = self._values.get(key)
        current = entry[0] if entry and entry[1] > self._monotonic() else 0.0
        total = current + amount
        self._values[key] = (total, self._monotonic() + ttl_seconds)
        return total


class UpstashError(RuntimeError):
    """The Upstash REST API returned an error or an unexpected payload."""


# Runs atomically on Upstash (EVAL takes a lock for the whole script), so the
# read, the budget comparison and the increment can't interleave with other
# requests. KEYS[1] = month key; ARGV = amount, limit, ttl, epsilon.
RESERVE_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current + tonumber(ARGV[1]) > tonumber(ARGV[2]) + tonumber(ARGV[4]) then
  return {0, tostring(current)}
end
local total = redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[3])
return {1, total}
"""


class UpstashCounterStore:
    """Counters over the Upstash Redis REST API.

    ``incr`` sends ``INCRBYFLOAT`` and ``EXPIRE`` through ``/multi-exec`` so the
    increment and TTL land as one transaction. ``reserve`` runs
    ``RESERVE_SCRIPT`` with ``EVAL`` for an atomic check-and-increment.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def _post(self, path: str, body: list[Any]) -> Any:
        response = await self._client.post(
            f"{self._url}{path}", json=body, headers=self._headers
        )
        try:
            payload = response.json()
        except ValueError as exc:
            status = response.status_code
            raise UpstashError(f"non-JSON response (HTTP {status})") from exc
        if response.status_code != 200 or (
            isinstance(payload, dict) and "error" in payload
        ):
            # Upstash error strings describe the command, never the token.
            error = payload.get("error") if isinstance(payload, dict) else None
            raise UpstashError(f"HTTP {response.status_code}: {error}")
        return payload

    async def get(self, key: str) -> float:
        payload = await self._post("", ["GET", key])
        if not isinstance(payload, dict) or "result" not in payload:
            raise UpstashError("unexpected GET response shape")
        result = payload["result"]
        return 0.0 if result is None else _to_float(result)

    async def incr(self, key: str, amount: float, ttl_seconds: int) -> float:
        payload = await self._post(
            "/multi-exec",
            [["INCRBYFLOAT", key, _decimal(amount)], ["EXPIRE", key, str(ttl_seconds)]],
        )
        if not isinstance(payload, list) or len(payload) != 2:
            raise UpstashError("unexpected multi-exec response shape")
        for item in payload:
            if not isinstance(item, dict) or "error" in item:
                raise UpstashError("multi-exec command failed")
        return _to_float(payload[0].get("result"))

    async def reserve(
        self, key: str, amount: float, limit: float, ttl_seconds: int
    ) -> float | None:
        payload = await self._post(
            "",
            [
                "EVAL",
                RESERVE_SCRIPT,
                1,
                key,
                _decimal(amount),
                _decimal(limit),
                str(ttl_seconds),
                _decimal(BUDGET_EPSILON_USD),
            ],
        )
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, list) or len(result) != 2 or result[0] not in (0, 1):
            raise UpstashError("unexpected EVAL response shape")
        total = _to_float(result[1])
        return total if result[0] == 1 else None

    async def aclose(self) -> None:
        await self._client.aclose()


def _decimal(value: float) -> str:
    """Plain decimal text (no exponent) for Redis float arguments."""
    if not math.isfinite(value):
        raise ValueError("amount must be finite")
    return f"{value:.12f}"


def _to_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise UpstashError("counter value is not a number") from exc
    if not math.isfinite(number):
        raise UpstashError("counter value is not finite")
    return number


class CounterBudgetStore:
    """A ``BudgetStore`` over any ``CounterStore``, keyed per UTC month."""

    def __init__(self, counters: CounterStore, *, clock: Clock = utc_now) -> None:
        self.counters = counters
        self._clock = clock

    async def get_month_spend(self) -> float:
        return await self.counters.get(month_key(self._clock()))

    async def add_spend(self, usd: float) -> float:
        if not math.isfinite(usd) or usd < 0:
            raise ValueError("spend must be a finite, non-negative amount")
        return await self.counters.incr(
            month_key(self._clock()), usd, BUDGET_TTL_SECONDS
        )

    async def reserve(self, usd: float, budget_usd: float) -> Reservation | None:
        if not math.isfinite(usd) or usd <= 0:
            raise ValueError("a reservation must be a finite, positive amount")
        if not math.isfinite(budget_usd):
            raise ValueError("the budget must be finite")
        key = month_key(self._clock())
        total = await self.counters.reserve(key, usd, budget_usd, BUDGET_TTL_SECONDS)
        return None if total is None else Reservation(key, usd)

    async def reconcile(self, reservation: Reservation, actual_usd: float) -> float:
        if not math.isfinite(actual_usd) or actual_usd < 0:
            raise ValueError("actual spend must be a finite, non-negative amount")
        # The reservation's own key, so a request that straddles midnight on
        # the 1st settles against the month that admitted it.
        return await self.counters.incr(
            reservation.key, actual_usd - reservation.amount_usd, BUDGET_TTL_SECONDS
        )


class InMemoryBudgetStore(CounterBudgetStore):
    """Per-process budget for development and tests."""

    def __init__(
        self,
        *,
        clock: Clock = utc_now,
        counters: InMemoryCounterStore | None = None,
    ) -> None:
        super().__init__(counters or InMemoryCounterStore(), clock=clock)


class UpstashBudgetStore(CounterBudgetStore):
    """Budget shared across all function instances via Upstash Redis."""

    def __init__(
        self,
        url: str,
        token: str,
        *,
        clock: Clock = utc_now,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(UpstashCounterStore(url, token, client=client), clock=clock)


class FailClosedBudgetStore:
    """Reports the budget as spent and admits nothing.

    Used in production when Upstash is missing.
    """

    async def get_month_spend(self) -> float:
        return math.inf

    async def add_spend(self, usd: float) -> float:
        return math.inf

    async def reserve(self, usd: float, budget_usd: float) -> Reservation | None:
        return None

    async def reconcile(self, reservation: Reservation, actual_usd: float) -> float:
        return math.inf


def upstash_config(env: Mapping[str, str]) -> tuple[str, str] | None:
    url = env.get("UPSTASH_REDIS_REST_URL", "").strip()
    token = env.get("UPSTASH_REDIS_REST_TOKEN", "").strip()
    return (url, token) if url and token else None


def is_production(env: Mapping[str, str]) -> bool:
    return env.get("VERCEL_ENV") == "production"


def select_counter_store(env: Mapping[str, str]) -> CounterStore:
    """Upstash when configured, else process memory."""
    config = upstash_config(env)
    return UpstashCounterStore(*config) if config else InMemoryCounterStore()


def select_budget_store(
    env: Mapping[str, str], counters: CounterStore | None = None
) -> BudgetStore:
    """Upstash when configured; in-memory in dev; fail closed in production."""
    if upstash_config(env) is None and is_production(env):
        return FailClosedBudgetStore()
    return CounterBudgetStore(counters or select_counter_store(env))

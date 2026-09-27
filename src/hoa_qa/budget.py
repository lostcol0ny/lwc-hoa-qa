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

Units: counters hold **integers**. Budget amounts are stored as micro-dollars
(1 µ$ = $0.000001) with ``INCRBY``, so sums are exact; dollars are converted
only at the edges (``usd_to_micros``/``micros_to_usd``). Charges round *up* and
the budget limit rounds *down*, so rounding never admits extra spend.

Fail-closed rule: when ``VERCEL_ENV=production`` and Upstash/KV Redis is not
configured or has an incomplete configuration, ``select_budget_store`` returns
a store that always reports the budget as spent. A misconfigured production
deployment refuses questions instead of running with no cap.
"""

import asyncio
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

logger = logging.getLogger("hoa_qa.budget")

BUDGET_TTL_SECONDS = 40 * 24 * 60 * 60
"""Month keys outlive their month by ~10 days, then Redis deletes them."""

MICROS_PER_USD = 1_000_000

BUDGET_KEY_PREFIX = "budget_micros"
"""Month keys are ``budget_micros:YYYY-MM``. The earlier float counters used
``budget:YYYY-MM``; a new prefix means an old float value can never reach
``INCRBY`` (which would reject it). No migration: nothing launched on them."""

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def month_key(now: datetime) -> str:
    """Return the per-calendar-month (UTC) budget key, e.g.
    ``budget_micros:2026-09``."""
    return f"{BUDGET_KEY_PREFIX}:{now.astimezone(UTC):%Y-%m}"


def _check_usd(usd: float) -> None:
    if not math.isfinite(usd):
        raise ValueError("amount must be finite")


def usd_to_micros(usd: float) -> int:
    """Dollars to micro-dollars, rounding *up* (charges never under-count).

    Rounding to 6 places first drops float noise (``0.05 * 1e6`` may be
    ``50000.000000000007``) so an exact amount isn't bumped by 1 µ$.
    """
    _check_usd(usd)
    return math.ceil(round(usd * MICROS_PER_USD, 6))


def usd_limit_to_micros(usd: float) -> int:
    """A budget limit in micro-dollars, rounding *down* (never admits extra)."""
    _check_usd(usd)
    return math.floor(round(usd * MICROS_PER_USD, 6))


def micros_to_usd(micros: int) -> float:
    return micros / MICROS_PER_USD


class CounterStore(Protocol):
    """Integer counters with expiry; shared by the budget and the rate limiter."""

    async def get(self, key: str) -> int: ...

    async def incr(self, key: str, amount: int, ttl_seconds: int) -> int:
        """Atomically add ``amount`` and (re)set the key's TTL; return the total."""
        ...

    async def reserve(
        self, key: str, amount: int, limit: int, ttl_seconds: int
    ) -> int | None:
        """Atomically add ``amount`` only if the new total stays within ``limit``.

        Return the new total, or None (nothing added) if it would exceed it.
        """
        ...


@dataclass(frozen=True)
class Reservation:
    """Spend held for one in-flight request, pinned to the month it was made in."""

    key: str
    amount_micros: int

    @property
    def amount_usd(self) -> float:
        return micros_to_usd(self.amount_micros)


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
        self._values: dict[str, tuple[int, float]] = {}
        self._monotonic = monotonic
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> int:
        entry = self._values.get(key)
        if entry is None or entry[1] <= self._monotonic():
            self._values.pop(key, None)
            return 0
        return entry[0]

    async def incr(self, key: str, amount: int, ttl_seconds: int) -> int:
        async with self._lock:
            return self._add(key, amount, ttl_seconds)

    async def reserve(
        self, key: str, amount: int, limit: int, ttl_seconds: int
    ) -> int | None:
        async with self._lock:
            current = await self.get(key)
            # Yield between the read and the write, as a network store would.
            # Only the lock stops another reservation interleaving here, so
            # the concurrency tests fail if the lock is removed.
            await asyncio.sleep(0)
            if current + amount > limit:
                return None
            return self._add(key, amount, ttl_seconds, base=current)

    def _add(
        self, key: str, amount: int, ttl_seconds: int, *, base: int | None = None
    ) -> int:
        if base is None:
            entry = self._values.get(key)
            base = entry[0] if entry and entry[1] > self._monotonic() else 0
        total = base + amount
        self._values[key] = (total, self._monotonic() + ttl_seconds)
        return total


class UpstashError(RuntimeError):
    """The Upstash REST API returned an error or an unexpected payload."""


# Runs atomically on Upstash (EVAL takes a lock for the whole script), so the
# read, the budget comparison and the increment can't interleave with other
# requests. KEYS[1] = month key; ARGV = amount, limit (integer micro-dollars),
# ttl. Integers stay exact in Lua's doubles up to 2^53 µ$ (~$9 billion).
RESERVE_SCRIPT = """
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current + tonumber(ARGV[1]) > tonumber(ARGV[2]) then
  return {0, current}
end
local total = redis.call('INCRBY', KEYS[1], ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[3])
return {1, total}
"""


class UpstashCounterStore:
    """Counters over the Upstash Redis REST API.

    ``incr`` sends ``INCRBY`` and ``EXPIRE`` through ``/multi-exec`` so the
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

    async def get(self, key: str) -> int:
        payload = await self._post("", ["GET", key])
        if not isinstance(payload, dict) or "result" not in payload:
            raise UpstashError("unexpected GET response shape")
        result = payload["result"]
        return 0 if result is None else _to_int(result)

    async def incr(self, key: str, amount: int, ttl_seconds: int) -> int:
        payload = await self._post(
            "/multi-exec",
            [["INCRBY", key, _integer(amount)], ["EXPIRE", key, str(ttl_seconds)]],
        )
        if not isinstance(payload, list) or len(payload) != 2:
            raise UpstashError("unexpected multi-exec response shape")
        for item in payload:
            if not isinstance(item, dict) or "error" in item:
                raise UpstashError("multi-exec command failed")
        return _to_int(payload[0].get("result"))

    async def reserve(
        self, key: str, amount: int, limit: int, ttl_seconds: int
    ) -> int | None:
        payload = await self._post(
            "",
            [
                "EVAL",
                RESERVE_SCRIPT,
                1,
                key,
                _integer(amount),
                _integer(limit),
                str(ttl_seconds),
            ],
        )
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, list) or len(result) != 2 or result[0] not in (0, 1):
            raise UpstashError("unexpected EVAL response shape")
        total = _to_int(result[1])
        return total if result[0] == 1 else None

    async def aclose(self) -> None:
        await self._client.aclose()


def _integer(value: int) -> str:
    """An integer Redis argument; refuses floats and bools outright."""
    if type(value) is not int:
        raise TypeError("counter amounts must be integers")
    return str(value)


def _to_int(value: object) -> int:
    """Parse an integer reply (Upstash may return numbers or numeric strings)."""
    if isinstance(value, bool):
        raise UpstashError("counter value is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    raise UpstashError("counter value is not an integer")


class CounterBudgetStore:
    """A ``BudgetStore`` over any ``CounterStore``, keyed per UTC month."""

    def __init__(self, counters: CounterStore, *, clock: Clock = utc_now) -> None:
        self.counters = counters
        self._clock = clock

    async def get_month_spend(self) -> float:
        return micros_to_usd(await self.counters.get(month_key(self._clock())))

    async def add_spend(self, usd: float) -> float:
        if not math.isfinite(usd) or usd < 0:
            raise ValueError("spend must be a finite, non-negative amount")
        total = await self.counters.incr(
            month_key(self._clock()), usd_to_micros(usd), BUDGET_TTL_SECONDS
        )
        return micros_to_usd(total)

    async def reserve(self, usd: float, budget_usd: float) -> Reservation | None:
        if not math.isfinite(usd) or usd <= 0:
            raise ValueError("a reservation must be a finite, positive amount")
        if not math.isfinite(budget_usd):
            raise ValueError("the budget must be finite")
        key = month_key(self._clock())
        amount = usd_to_micros(usd)
        total = await self.counters.reserve(
            key, amount, usd_limit_to_micros(budget_usd), BUDGET_TTL_SECONDS
        )
        return None if total is None else Reservation(key, amount)

    async def reconcile(self, reservation: Reservation, actual_usd: float) -> float:
        if not math.isfinite(actual_usd) or actual_usd < 0:
            raise ValueError("actual spend must be a finite, non-negative amount")
        # The reservation's own key, so a request that straddles midnight on
        # the 1st settles against the month that admitted it.
        total = await self.counters.incr(
            reservation.key,
            usd_to_micros(actual_usd) - reservation.amount_micros,
            BUDGET_TTL_SECONDS,
        )
        return micros_to_usd(total)


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


UPSTASH_URL_VAR = "UPSTASH_REDIS_REST_URL"
UPSTASH_TOKEN_VAR = "UPSTASH_REDIS_REST_TOKEN"
KV_URL_VAR = "KV_REST_API_URL"
KV_TOKEN_VAR = "KV_REST_API_TOKEN"


@dataclass(frozen=True)
class RedisConfigResult:
    """Result of resolving Redis credentials as matched pairs."""

    credentials: tuple[str, str] | None
    missing_variables: tuple[str, ...]
    incomplete: bool
    family: str | None = None


def _clean_env(env: Mapping[str, str], key: str) -> str:
    raw = env.get(key)
    return "" if raw is None else str(raw).strip()


def resolve_redis_config(env: Mapping[str, str]) -> RedisConfigResult:
    """Resolve Redis credentials as matched pairs.

    - Resolves credentials as pairs: uses the UPSTASH pair if both its URL
      and token are set; otherwise the KV pair if both are set. URLs and
      tokens are never mixed between families.
    - Any incomplete pair (exactly one of URL or token set in either family)
      is a configuration error, even if the other family is complete.
    - KV_REST_API_READ_ONLY_TOKEN is never used.
    """
    upstash_url = _clean_env(env, UPSTASH_URL_VAR)
    upstash_token = _clean_env(env, UPSTASH_TOKEN_VAR)
    kv_url = _clean_env(env, KV_URL_VAR)
    kv_token = _clean_env(env, KV_TOKEN_VAR)

    missing: list[str] = []

    # Check Upstash pair completeness
    upstash_has_url = bool(upstash_url)
    upstash_has_token = bool(upstash_token)
    if upstash_has_url != upstash_has_token:
        if upstash_has_url:
            missing.append(UPSTASH_TOKEN_VAR)
        else:
            missing.append(UPSTASH_URL_VAR)

    # Check KV pair completeness (KV_REST_API_READ_ONLY_TOKEN is never used)
    kv_has_url = bool(kv_url)
    kv_has_token = bool(kv_token)
    if kv_has_url != kv_has_token:
        if kv_has_url:
            missing.append(KV_TOKEN_VAR)
        else:
            missing.append(KV_URL_VAR)

    if missing:
        # Any incomplete pair is a configuration error, even if the other
        # family is complete.
        return RedisConfigResult(
            credentials=None,
            missing_variables=tuple(sorted(missing)),
            incomplete=True,
            family=None,
        )

    # UPSTASH pair takes precedence if complete
    if upstash_has_url and upstash_has_token:
        return RedisConfigResult(
            credentials=(upstash_url, upstash_token),
            missing_variables=(),
            incomplete=False,
            family="upstash",
        )

    # KV pair used if complete
    if kv_has_url and kv_has_token:
        return RedisConfigResult(
            credentials=(kv_url, kv_token),
            missing_variables=(),
            incomplete=False,
            family="kv",
        )

    # Neither pair is set
    return RedisConfigResult(
        credentials=None,
        missing_variables=(),
        incomplete=False,
        family=None,
    )


def upstash_config(env: Mapping[str, str]) -> tuple[str, str] | None:
    return resolve_redis_config(env).credentials


def is_production(env: Mapping[str, str]) -> bool:
    return env.get("VERCEL_ENV") == "production"


def check_redis_config(
    env: Mapping[str, str],
    log: logging.Logger | None = None,
    *,
    production: bool | None = None,
) -> None:
    """Log Redis configuration status on startup.

    In production: logs an ERROR naming missing variables for incomplete pairs
    or logs an ERROR if Redis is unconfigured, without leaking secrets or URLs.
    In development: logs a WARNING naming missing variables for incomplete pairs.
    """
    log = log or logger
    redis = resolve_redis_config(env)
    is_prod = is_production(env) if production is None else production
    if redis.incomplete:
        missing_str = ", ".join(redis.missing_variables)
        if is_prod:
            log.error(
                "Redis configuration incomplete: missing %s; "
                "failing closed in production",
                missing_str,
            )
        else:
            log.warning(
                "Redis configuration incomplete: missing %s; "
                "falling back to in-memory store",
                missing_str,
            )
    elif is_prod and redis.credentials is None:
        log.error("Redis is not configured in production; failing closed")


def select_counter_store(env: Mapping[str, str]) -> CounterStore:
    """Upstash when configured, else process memory."""
    config = upstash_config(env)
    return UpstashCounterStore(*config) if config else InMemoryCounterStore()


def select_budget_store(
    env: Mapping[str, str],
    counters: CounterStore | None = None,
    *,
    production: bool | None = None,
) -> BudgetStore:
    """Upstash when configured; in-memory in dev; fail closed in production."""
    redis = resolve_redis_config(env)
    is_prod = is_production(env) if production is None else production
    if is_prod and (redis.credentials is None or redis.incomplete):
        return FailClosedBudgetStore()
    return CounterBudgetStore(counters or select_counter_store(env))

"""Monthly spend cap and the shared counter stores behind it.

The web app checks the month's spend before calling the asker and adds each
request's estimated cost afterwards. Counters live in Upstash Redis in
production so every function instance shares one total; in development they
live in process memory.

Fail-closed rule: when ``VERCEL_ENV=production`` and Upstash is not configured,
``select_budget_store`` returns a store that always reports the budget as
spent. A misconfigured production deployment refuses questions instead of
running with no cap.
"""

import math
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

BUDGET_TTL_SECONDS = 40 * 24 * 60 * 60
"""Month keys outlive their month by ~10 days, then Redis deletes them."""

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


class BudgetStore(Protocol):
    async def get_month_spend(self) -> float: ...

    async def add_spend(self, usd: float) -> float: ...


class InMemoryCounterStore:
    """Process-local counters for development and tests."""

    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._values: dict[str, tuple[float, float]] = {}
        self._monotonic = monotonic

    async def get(self, key: str) -> float:
        entry = self._values.get(key)
        if entry is None or entry[1] <= self._monotonic():
            self._values.pop(key, None)
            return 0.0
        return entry[0]

    async def incr(self, key: str, amount: float, ttl_seconds: int) -> float:
        total = await self.get(key) + amount
        self._values[key] = (total, self._monotonic() + ttl_seconds)
        return total


class UpstashError(RuntimeError):
    """The Upstash REST API returned an error or an unexpected payload."""


class UpstashCounterStore:
    """Counters over the Upstash Redis REST API.

    ``incr`` sends ``INCRBYFLOAT`` and ``EXPIRE`` through ``/multi-exec`` so the
    increment and TTL land as one transaction.
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
            [["INCRBYFLOAT", key, f"{amount:.10f}"], ["EXPIRE", key, str(ttl_seconds)]],
        )
        if not isinstance(payload, list) or len(payload) != 2:
            raise UpstashError("unexpected multi-exec response shape")
        for item in payload:
            if not isinstance(item, dict) or "error" in item:
                raise UpstashError("multi-exec command failed")
        return _to_float(payload[0].get("result"))

    async def aclose(self) -> None:
        await self._client.aclose()


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
    """Reports the budget as spent. Used in production when Upstash is missing."""

    async def get_month_spend(self) -> float:
        return math.inf

    async def add_spend(self, usd: float) -> float:
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

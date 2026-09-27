import asyncio
import json
import math
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest
from web_fakes import FixedClock

from hoa_qa.budget import (
    BUDGET_TTL_SECONDS,
    CounterBudgetStore,
    FailClosedBudgetStore,
    InMemoryBudgetStore,
    InMemoryCounterStore,
    UpstashBudgetStore,
    UpstashCounterStore,
    UpstashError,
    month_key,
    select_budget_store,
    select_counter_store,
)

UPSTASH_ENV = {
    "UPSTASH_REDIS_REST_URL": "https://example-upstash.io",
    "UPSTASH_REDIS_REST_TOKEN": "test-token",
}


def test_month_key_is_utc_calendar_month() -> None:
    assert month_key(datetime(2026, 9, 30, 23, 59, tzinfo=UTC)) == "budget:2026-09"
    assert month_key(datetime(2026, 10, 1, tzinfo=UTC)) == "budget:2026-10"
    # 20:00 on Sep 30 in UTC-5 is already October in UTC.
    central = timezone(timedelta(hours=-5))
    assert month_key(datetime(2026, 9, 30, 20, tzinfo=central)) == "budget:2026-10"


def test_month_rollover_starts_a_fresh_counter() -> None:
    clock = FixedClock(datetime(2026, 9, 30, 23, 59, tzinfo=UTC))
    store = InMemoryBudgetStore(clock=clock)

    async def scenario() -> tuple[float, float]:
        await store.add_spend(4.0)
        september = await store.get_month_spend()
        clock.now = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
        return september, await store.get_month_spend()

    assert asyncio.run(scenario()) == (4.0, 0.0)


def test_add_spend_rejects_invalid_amounts() -> None:
    store = InMemoryBudgetStore()
    for bad in (-0.01, math.nan, math.inf):
        with pytest.raises(ValueError):
            asyncio.run(store.add_spend(bad))


def test_in_memory_counters_expire() -> None:
    now = [0.0]
    counters = InMemoryCounterStore(monotonic=lambda: now[0])
    asyncio.run(counters.incr("k", 2, ttl_seconds=10))
    now[0] = 9.9
    assert asyncio.run(counters.get("k")) == 2
    now[0] = 10.0
    assert asyncio.run(counters.get("k")) == 0


def test_store_selection() -> None:
    assert isinstance(select_counter_store({}), InMemoryCounterStore)
    assert isinstance(select_counter_store(UPSTASH_ENV), UpstashCounterStore)
    assert isinstance(select_budget_store({}), CounterBudgetStore)
    prod = {"VERCEL_ENV": "production"}
    assert isinstance(select_budget_store(prod), FailClosedBudgetStore)
    upstash = select_budget_store({**prod, **UPSTASH_ENV})
    assert isinstance(upstash, CounterBudgetStore)
    assert isinstance(upstash.counters, UpstashCounterStore)
    # Preview deployments without Upstash fall back to memory, not fail closed.
    preview = select_budget_store({"VERCEL_ENV": "preview"})
    assert isinstance(preview, CounterBudgetStore)


def test_fail_closed_store_reports_infinite_spend() -> None:
    assert asyncio.run(FailClosedBudgetStore().get_month_spend()) == math.inf


class FakeUpstash:
    """A tiny Redis emulation behind the Upstash REST wire format."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.requests: list[tuple[str, object, str | None]] = []

    def run(self, command: list[str]) -> object:
        name, key, *args = command
        if name == "GET":
            return self.data.get(key)
        if name == "INCRBYFLOAT":
            value = float(self.data.get(key, "0")) + float(args[0])
            self.data[key] = repr(value)
            return self.data[key]
        if name == "EXPIRE":
            self.ttls[key] = int(args[0])
            return 1
        raise AssertionError(f"unexpected command {name}")

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(
            (request.url.path, body, request.headers.get("authorization"))
        )
        if request.url.path == "/multi-exec":
            return httpx.Response(200, json=[{"result": self.run(c)} for c in body])
        return httpx.Response(200, json={"result": self.run(body)})


def upstash_store(fake: FakeUpstash, clock: FixedClock) -> UpstashBudgetStore:
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return UpstashBudgetStore(
        UPSTASH_ENV["UPSTASH_REDIS_REST_URL"],
        UPSTASH_ENV["UPSTASH_REDIS_REST_TOKEN"],
        clock=clock,
        client=client,
    )


def test_upstash_budget_store_round_trip() -> None:
    fake = FakeUpstash()
    store = upstash_store(fake, FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))

    async def scenario() -> tuple[float, float, float]:
        empty = await store.get_month_spend()
        await store.add_spend(0.0125)
        total = await store.add_spend(0.0025)
        return empty, total, await store.get_month_spend()

    empty, total, spend = asyncio.run(scenario())
    assert empty == 0.0
    assert math.isclose(total, 0.015)
    assert math.isclose(spend, 0.015)
    assert fake.ttls == {"budget:2026-09": BUDGET_TTL_SECONDS}
    paths = [path for path, _, _ in fake.requests]
    assert paths == ["/", "/multi-exec", "/multi-exec", "/"]
    _, first_incr, auth = fake.requests[1]
    assert first_incr == [
        ["INCRBYFLOAT", "budget:2026-09", "0.0125000000"],
        ["EXPIRE", "budget:2026-09", str(BUDGET_TTL_SECONDS)],
    ]
    assert auth == "Bearer test-token"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"error": "WRONGPASS invalid password"}),
        httpx.Response(200, json={"error": "ERR something"}),
        httpx.Response(200, json=[{"result": "1"}, {"error": "ERR expire"}]),
        httpx.Response(200, json={"result": "not-a-number"}),
        httpx.Response(502, text="bad gateway"),
    ],
)
def test_upstash_errors_raise(response: httpx.Response) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response))
    store = UpstashCounterStore("https://example-upstash.io", "t", client=client)
    with pytest.raises(UpstashError):
        asyncio.run(store.incr("k", 1, 10))
    with pytest.raises(UpstashError):
        asyncio.run(store.get("k"))

import asyncio
import contextlib
import math
from datetime import UTC, datetime, timedelta, timezone

import httpx
import pytest
from web_fakes import UPSTASH_ENV, FakeUpstash, FixedClock, upstash_store

from hoa_qa.budget import (
    BUDGET_TTL_SECONDS,
    RESERVE_SCRIPT,
    CounterBudgetStore,
    FailClosedBudgetStore,
    InMemoryBudgetStore,
    InMemoryCounterStore,
    UpstashCounterStore,
    UpstashError,
    month_key,
    select_budget_store,
    select_counter_store,
    usd_limit_to_micros,
    usd_to_micros,
)


def test_month_key_is_utc_calendar_month() -> None:
    assert (
        month_key(datetime(2026, 9, 30, 23, 59, tzinfo=UTC)) == "budget_micros:2026-09"
    )
    assert month_key(datetime(2026, 10, 1, tzinfo=UTC)) == "budget_micros:2026-10"
    # 20:00 on Sep 30 in UTC-5 is already October in UTC.
    central = timezone(timedelta(hours=-5))
    assert (
        month_key(datetime(2026, 9, 30, 20, tzinfo=central)) == "budget_micros:2026-10"
    )


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
    assert fake.ttls == {"budget_micros:2026-09": BUDGET_TTL_SECONDS}
    paths = [path for path, _, _ in fake.requests]
    assert paths == ["/", "/multi-exec", "/multi-exec", "/"]
    _, first_incr, auth = fake.requests[1]
    assert first_incr == [
        ["INCRBY", "budget_micros:2026-09", "12500"],
        ["EXPIRE", "budget_micros:2026-09", str(BUDGET_TTL_SECONDS)],
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


def test_in_memory_reserve_respects_limit() -> None:
    store = InMemoryBudgetStore(clock=FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))

    async def scenario() -> list[bool]:
        return [await store.reserve(0.05, 0.15) is not None for _ in range(4)]

    assert asyncio.run(scenario()) == [True, True, True, False]
    assert math.isclose(asyncio.run(store.get_month_spend()), 0.15)


def test_reserve_rejects_invalid_amounts() -> None:
    store = InMemoryBudgetStore()
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            asyncio.run(store.reserve(bad, 1.0))
    with pytest.raises(ValueError):
        asyncio.run(store.reserve(0.05, math.inf))
    reservation = asyncio.run(store.reserve(0.05, 1.0))
    assert reservation is not None
    with pytest.raises(ValueError):
        asyncio.run(store.reconcile(reservation, -0.01))


def test_reconcile_settles_against_the_reserving_month() -> None:
    clock = FixedClock(datetime(2026, 9, 30, 23, 59, 59, tzinfo=UTC))
    store = InMemoryBudgetStore(clock=clock)

    async def scenario() -> tuple[float, float]:
        reservation = await store.reserve(0.05, 1.0)
        assert reservation is not None and reservation.key == "budget_micros:2026-09"
        clock.now = datetime(2026, 10, 1, 0, 0, 1, tzinfo=UTC)
        await store.reconcile(reservation, 0.01)
        october = await store.get_month_spend()
        clock.now = datetime(2026, 9, 30, 23, 59, 59, tzinfo=UTC)
        return await store.get_month_spend(), october

    september, october = asyncio.run(scenario())
    assert math.isclose(september, 0.01)
    assert october == 0.0


def test_fail_closed_store_admits_nothing() -> None:
    assert asyncio.run(FailClosedBudgetStore().reserve(0.05, 100.0)) is None


def test_upstash_reserve_uses_eval_and_reconciles() -> None:
    fake = FakeUpstash()
    store = upstash_store(fake, FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))

    async def scenario() -> tuple[bool, float]:
        first = await store.reserve(0.05, 0.08)
        second = await store.reserve(0.05, 0.08)  # 0.10 > 0.08: refused
        assert first is not None
        await store.reconcile(first, 0.01)
        return second is None, await store.get_month_spend()

    second_refused, total = asyncio.run(scenario())
    assert second_refused
    assert math.isclose(total, 0.01)
    _, eval_body, _ = fake.requests[0]
    assert eval_body == [
        "EVAL",
        RESERVE_SCRIPT,
        1,
        "budget_micros:2026-09",
        "50000",
        "80000",
        str(BUDGET_TTL_SECONDS),
    ]
    # Reconcile is a negative INCRBY (micro-dollars) on the reservation's key.
    _, reconcile_body, _ = fake.requests[2]
    assert reconcile_body == [
        ["INCRBY", "budget_micros:2026-09", "-40000"],
        ["EXPIRE", "budget_micros:2026-09", str(BUDGET_TTL_SECONDS)],
    ]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"result": "OK"}),
        httpx.Response(200, json={"result": [2, "0.1"]}),
        httpx.Response(200, json={"result": [1]}),
        httpx.Response(200, json={"result": [1, "nope"]}),
        httpx.Response(200, json={"error": "ERR script"}),
        httpx.Response(503, text="unavailable"),
    ],
)
def test_upstash_reserve_errors_raise(response: httpx.Response) -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response))
    store = UpstashCounterStore("https://example-upstash.io", "t", client=client)
    with pytest.raises(UpstashError):
        asyncio.run(store.reserve("k", 50_000, 1_000_000, 10))


# --- Integer micro-dollars ---------------------------------------------------


def test_usd_micros_conversion_rounds_conservatively() -> None:
    assert usd_to_micros(0.05) == 50_000  # no float-noise bump
    assert usd_to_micros(0.0000001) == 1  # charges round up
    assert usd_to_micros(0.0123456789) == 12_346
    assert usd_limit_to_micros(0.0000019) == 1  # limits round down
    assert usd_limit_to_micros(5.0) == 5_000_000
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            usd_to_micros(bad)


def test_many_small_amounts_sum_exactly() -> None:
    """Exact to 1 µ$: float accumulation (0.1 + 0.2 != 0.3) can't drift."""
    store = InMemoryBudgetStore(clock=FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))

    async def scenario() -> float:
        for _ in range(1000):
            await store.add_spend(0.1)
            await store.add_spend(0.2)
        return await store.get_month_spend()

    assert asyncio.run(scenario()) == 300.0


def test_budget_of_exactly_n_reservations_admits_n() -> None:
    store = InMemoryBudgetStore(clock=FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))

    async def scenario() -> list[bool]:
        return [await store.reserve(0.1, 0.3) is not None for _ in range(4)]

    # With floats, 0.1 + 0.1 + 0.1 > 0.3 needed an epsilon; integers don't.
    assert asyncio.run(scenario()) == [True, True, True, False]


def test_upstash_counters_refuse_non_integer_amounts() -> None:
    store = upstash_store(FakeUpstash(), FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))
    with pytest.raises(TypeError):
        asyncio.run(store.counters.incr("k", 0.5, 10))  # type: ignore[arg-type]


def test_legacy_float_month_key_is_ignored() -> None:
    """A pre-launch ``budget:YYYY-MM`` float value can't break INCRBY."""
    fake = FakeUpstash()
    fake.data["budget:2026-09"] = "0.0500000000000000028"
    store = upstash_store(fake, FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))

    async def scenario() -> float:
        reservation = await store.reserve(0.05, 1.0)
        assert reservation is not None
        await store.reconcile(reservation, 0.01)
        return await store.get_month_spend()

    assert asyncio.run(scenario()) == 0.01
    assert fake.data["budget:2026-09"] == "0.0500000000000000028"
    # And INCRBY really would have rejected it (the emulation matches Redis).
    assert fake.handler_status(["INCRBY", "budget:2026-09", "1"]) == 400


def test_reserve_script_runs_under_lua() -> None:
    """RESERVE_SCRIPT itself, executed by lupa: admit, refuse, TTL, totals."""
    fake = FakeUpstash()
    run = fake.run
    assert run(["EVAL", RESERVE_SCRIPT, 1, "m", "60", "100", "99"]) == [1, 60]
    assert fake.ttls == {"m": 99}
    assert run(["EVAL", RESERVE_SCRIPT, 1, "m", "41", "100", "99"]) == [0, 60]
    assert run(["EVAL", RESERVE_SCRIPT, 1, "m", "40", "100", "99"]) == [1, 100]
    assert fake.data["m"] == "100"


# --- The lock is what makes in-memory reservation atomic ---------------------


async def _concurrent_reservations(store: InMemoryBudgetStore, n: int) -> int:
    results = await asyncio.gather(*(store.reserve(0.05, 0.15) for _ in range(n)))
    return sum(r is not None for r in results)


def test_concurrent_in_memory_reservations_respect_budget() -> None:
    store = InMemoryBudgetStore(clock=FixedClock(datetime(2026, 9, 27, tzinfo=UTC)))
    assert asyncio.run(_concurrent_reservations(store, 20)) == 3
    assert math.isclose(asyncio.run(store.get_month_spend()), 0.15)


def test_without_the_lock_reservations_overspend() -> None:
    """Proves the concurrency test has teeth: remove the lock and it fails."""
    counters = InMemoryCounterStore()
    counters._lock = contextlib.nullcontext()  # type: ignore[assignment]
    store = InMemoryBudgetStore(
        clock=FixedClock(datetime(2026, 9, 27, tzinfo=UTC)), counters=counters
    )
    assert asyncio.run(_concurrent_reservations(store, 20)) > 3

"""Atomic budget reservation: reserve R before the asker, reconcile after."""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import pytest
from web_fakes import (
    QUESTION,
    FakeUpstash,
    FixedClock,
    Harness,
    RecordingAsker,
    Result,
    build_harness,
    make_settings,
    upstash_store,
)

from hoa_qa.budget import (
    BudgetStore,
    InMemoryBudgetStore,
    Reservation,
)
from hoa_qa.models import Outcome
from hoa_qa.web.app import reservation_amount

R = 0.05
CLOCK_NOW = datetime(2026, 9, 27, 12, tzinfo=UTC)


def ask(h: Harness):
    return h.client.post("/api/ask", json={"question": QUESTION})


def spend(budget: BudgetStore) -> float:
    return asyncio.run(budget.get_month_spend())


@dataclass
class SpyBudget(InMemoryBudgetStore):
    """In-memory store that records calls and can fail reconcile."""

    reserved: list[float] = field(default_factory=list)
    fail_reconcile: bool = False

    def __post_init__(self) -> None:
        super().__init__(clock=FixedClock(CLOCK_NOW))

    async def reserve(self, usd: float, budget_usd: float) -> Reservation | None:
        self.reserved.append(usd)
        return await super().reserve(usd, budget_usd)

    async def reconcile(self, reservation: Reservation, actual_usd: float) -> float:
        if self.fail_reconcile:
            raise ConnectionError("redis write failed")
        return await super().reconcile(reservation, actual_usd)


# --- B1: accounting failures refuse instead of spending --------------------


def test_reads_ok_but_writes_fail_never_calls_asker() -> None:
    fake = FakeUpstash()
    fake.fail_writes = True
    store = upstash_store(fake, FixedClock(CLOCK_NOW))
    assert spend(store) == 0.0  # reads work
    h = build_harness(budget=store, monthly_budget_usd=10.0)
    for _ in range(3):
        assert ask(h).json()["outcome"] == "budget_exhausted"
    assert h.asker.questions == []


def test_reserve_error_is_logged_without_question(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken(InMemoryBudgetStore):
        async def reserve(self, usd: float, budget_usd: float) -> Reservation | None:
            raise ConnectionError(QUESTION)

    h = build_harness(budget=Broken(), monthly_budget_usd=10.0)
    with caplog.at_level(logging.INFO):
        assert ask(h).json()["outcome"] == "budget_exhausted"
    assert "budget reservation failed" in caplog.text
    assert "ConnectionError" in caplog.text
    assert QUESTION not in caplog.text
    assert h.asker.questions == []


def test_budget_smaller_than_one_reservation_admits_nothing() -> None:
    h = build_harness(monthly_budget_usd=R / 2)
    assert ask(h).json()["outcome"] == "budget_exhausted"
    assert h.asker.questions == []


# --- B2: concurrent admission is bounded by the budget ---------------------


class SlowAsker(RecordingAsker):
    """Holds each call open so all requests overlap."""

    async def __call__(self, question: str) -> Result:
        await asyncio.sleep(0.05)
        return await super().__call__(question)


async def fire_concurrently(h: Harness, n: int) -> list[str]:
    transport = httpx.ASGITransport(app=h.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        responses = await asyncio.gather(
            *(c.post("/api/ask", json={"question": QUESTION}) for _ in range(n))
        )
    return [r.json()["outcome"] for r in responses]


@pytest.mark.parametrize("backend", ["memory", "upstash"])
def test_50_concurrent_requests_admit_exactly_budget_over_r(backend: str) -> None:
    budget: BudgetStore
    if backend == "upstash":
        budget = upstash_store(FakeUpstash(), FixedClock(CLOCK_NOW))
    else:
        budget = InMemoryBudgetStore(clock=FixedClock(CLOCK_NOW))
    asker = SlowAsker(cost_usd=R)
    h = build_harness(
        asker=asker,
        budget=budget,
        monthly_budget_usd=3 * R,
        rate_limit_per_hour=1000,
        rate_limit_per_day=1000,
    )
    outcomes = asyncio.run(fire_concurrently(h, 50))
    assert outcomes.count("answered") == 3
    assert outcomes.count("budget_exhausted") == 47
    assert len(asker.questions) == 3
    assert math.isclose(spend(budget), 3 * R)


# --- Reconciliation ----------------------------------------------------------


def test_negative_reconciliation_releases_unused_reservation() -> None:
    budget = SpyBudget()
    h = build_harness(asker=RecordingAsker(cost_usd=0.01), budget=budget)
    assert ask(h).json()["outcome"] == "answered"
    assert budget.reserved == [R]
    assert math.isclose(spend(budget), 0.01)  # 0.05 held, then -0.04


def test_reconcile_failure_keeps_full_reservation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    budget = SpyBudget(fail_reconcile=True)
    h = build_harness(asker=RecordingAsker(cost_usd=0.01), budget=budget)
    with caplog.at_level(logging.INFO):
        assert ask(h).json()["outcome"] == "answered"
    assert math.isclose(spend(budget), R)
    assert "reconcile failed; keeping reservation request_id=req-fake" in caplog.text
    assert QUESTION not in caplog.text


def test_cost_above_reservation_adds_excess_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    budget = SpyBudget()
    h = build_harness(asker=RecordingAsker(cost_usd=0.08), budget=budget)
    with caplog.at_level(logging.WARNING):
        ask(h)
    assert math.isclose(spend(budget), 0.08)
    assert "cost exceeded reservation" in caplog.text


def test_error_outcome_reconciles_reported_cost() -> None:
    @dataclass
    class ErrorOutcomeAsker(RecordingAsker):
        async def __call__(self, question: str) -> Result:
            result = await super().__call__(question)
            answer = result.answer.model_copy(update={"outcome": Outcome.error})
            return Result(answer=answer, estimated_cost_usd=0.003)

    budget = SpyBudget()
    h = build_harness(asker=ErrorOutcomeAsker(), budget=budget)
    assert ask(h).json()["outcome"] == "error"
    assert math.isclose(spend(budget), 0.003)


def test_asker_raising_without_cost_keeps_full_reservation() -> None:
    budget = SpyBudget()
    h = build_harness(asker=RecordingAsker(error=RuntimeError("boom")), budget=budget)
    assert ask(h).status_code == 500
    assert math.isclose(spend(budget), R)


def test_asker_raising_with_cost_reconciles_it() -> None:
    class CostlyError(RuntimeError):
        estimated_cost_usd = 0.002

    budget = SpyBudget()
    h = build_harness(asker=RecordingAsker(error=CostlyError()), budget=budget)
    assert ask(h).status_code == 500
    assert math.isclose(spend(budget), 0.002)


@pytest.mark.parametrize("bad_cost", [math.nan, -1.0, math.inf])
def test_unusable_reported_cost_keeps_reservation(bad_cost: float) -> None:
    budget = SpyBudget()
    h = build_harness(asker=RecordingAsker(cost_usd=bad_cost), budget=budget)
    ask(h)
    assert math.isclose(spend(budget), R)


# --- R = max(env default, asker.max_cost_usd) --------------------------------


@dataclass
class CappedAsker(RecordingAsker):
    max_cost_usd: object = 0.2


def test_r_comes_from_asker_max_cost_when_larger() -> None:
    budget = SpyBudget()
    h = build_harness(asker=CappedAsker(max_cost_usd=0.2), budget=budget)
    ask(h)
    assert budget.reserved == [0.2]


def test_r_uses_env_default_when_asker_max_is_smaller_or_invalid() -> None:
    settings = make_settings(budget_reserve_per_request_usd=0.07)
    assert reservation_amount(settings, CappedAsker(max_cost_usd=0.01)) == 0.07
    assert reservation_amount(settings, RecordingAsker()) == 0.07
    for bad in ("lots", math.nan, -5, None):
        assert reservation_amount(settings, CappedAsker(max_cost_usd=bad)) == 0.07


def test_large_max_cost_limits_admission() -> None:
    budget = SpyBudget()
    h = build_harness(
        asker=CappedAsker(max_cost_usd=0.4), budget=budget, monthly_budget_usd=0.3
    )
    assert ask(h).json()["outcome"] == "budget_exhausted"
    assert h.asker.questions == []

import asyncio
from datetime import UTC, datetime, timedelta

from web_fakes import QUESTION, FixedClock, Harness, build_harness

from hoa_qa.budget import InMemoryCounterStore
from hoa_qa.models import Answer, Outcome
from hoa_qa.web.ratelimit import RateLimiter


def ask(h: Harness, **headers: str):
    return h.client.post("/api/ask", json={"question": QUESTION}, headers=headers)


def test_429_after_hourly_limit_per_ip() -> None:
    h = build_harness(rate_limit_per_hour=3, monthly_budget_usd=100.0)
    for _ in range(3):
        assert ask(h).status_code == 200
    response = ask(h)
    assert response.status_code == 429
    answer = Answer.model_validate(response.json())
    assert answer.outcome == Outcome.error
    assert "slow down" in answer.answer_text
    assert len(h.asker.questions) == 3


def test_limit_is_per_ip_on_vercel() -> None:
    h = build_harness(rate_limit_per_hour=2, on_vercel=True, monthly_budget_usd=100.0)
    for _ in range(2):
        assert ask(h, **{"x-forwarded-for": "203.0.113.7"}).status_code == 200
    assert ask(h, **{"x-forwarded-for": "203.0.113.7"}).status_code == 429
    # A different client still gets through.
    assert ask(h, **{"x-forwarded-for": "198.51.100.9"}).status_code == 200
    assert ask(h, **{"x-real-ip": "192.0.2.44"}).status_code == 200


def test_forwarded_headers_ignored_off_vercel() -> None:
    h = build_harness(rate_limit_per_hour=2, on_vercel=False, monthly_budget_usd=100.0)
    # Off Vercel, a client could forge these; rotating them must not help.
    for i in range(2):
        assert ask(h, **{"x-forwarded-for": f"203.0.113.{i}"}).status_code == 200
    assert ask(h, **{"x-forwarded-for": "203.0.113.99"}).status_code == 429


def test_hourly_window_resets_and_daily_limit_holds() -> None:
    clock = FixedClock(datetime(2026, 9, 27, 10, tzinfo=UTC))
    limiter = RateLimiter(InMemoryCounterStore(), per_hour=2, per_day=3, clock=clock)

    async def scenario() -> list[bool]:
        results = [await limiter.hit("ip"), await limiter.hit("ip")]
        results.append(await limiter.hit("ip"))  # 3rd in the hour: blocked
        clock.now += timedelta(hours=1)
        results.append(await limiter.hit("ip"))  # new hour but 4th today
        clock.now += timedelta(days=1)
        results.append(await limiter.hit("ip"))  # new day
        return results

    assert asyncio.run(scenario()) == [True, True, False, False, True]


def test_limiter_keys_do_not_contain_raw_ip() -> None:
    counters = InMemoryCounterStore()
    limiter = RateLimiter(counters, per_hour=10, per_day=50)
    asyncio.run(limiter.hit("203.0.113.7"))
    assert counters._values
    assert all("203.0.113.7" not in key for key in counters._values)


def test_limiter_fails_open_when_store_errors() -> None:
    class Broken(InMemoryCounterStore):
        async def incr(self, key: str, amount: float, ttl_seconds: int) -> float:
            raise ConnectionError("down")

    limiter = RateLimiter(Broken(), per_hour=1, per_day=1)
    assert asyncio.run(limiter.hit("ip")) is True

"""Test doubles and helpers shared by the web tests."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hoa_qa.budget import (
    RESERVE_SCRIPT,
    BudgetStore,
    InMemoryBudgetStore,
    InMemoryCounterStore,
    UpstashBudgetStore,
)
from hoa_qa.models import Answer, Outcome
from hoa_qa.web.app import create_app
from hoa_qa.web.deps import get_asker
from hoa_qa.web.settings import DISCLAIMER, WebSettings

FIXTURE_CORPUS = Path(__file__).parents[1] / "fixtures" / "mini_corpus.json"
QUESTION = "How many pool guests may I bring?"


@dataclass(frozen=True)
class Result:
    answer: Answer
    estimated_cost_usd: float


@dataclass
class RecordingAsker:
    cost_usd: float = 0.01
    error: Exception | None = None
    questions: list[str] = field(default_factory=list)

    async def __call__(self, question: str) -> Result:
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        answer = Answer(
            request_id="req-fake",
            outcome=Outcome.answered,
            answer_text="You may bring two guests.",
            citations=(),
            confidence=0.9,
            conflicts_noted=(),
            disclaimer=DISCLAIMER,
        )
        return Result(answer=answer, estimated_cost_usd=self.cost_usd)


class FixedClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def make_settings(**overrides: object) -> WebSettings:
    values: dict[str, object] = {
        "production": False,
        "on_vercel": False,
        "corpus_path": FIXTURE_CORPUS,
        "documents_url": "https://lakewoodcreekhoa.com/",
        "monthly_budget_usd": 1.0,
        "budget_reserve_per_request_usd": 0.05,
        "rate_limit_per_hour": 10,
        "rate_limit_per_day": 50,
        "fake_asker": False,
    }
    values.update(overrides)
    return WebSettings(**values)  # type: ignore[arg-type]


@dataclass
class Harness:
    app: FastAPI
    client: TestClient
    asker: RecordingAsker
    budget: BudgetStore
    clock: FixedClock


def build_harness(
    *,
    env: dict[str, str] | None = None,
    asker: RecordingAsker | None = None,
    budget: BudgetStore | None = None,
    **s: object,
) -> Harness:
    clock = FixedClock(datetime(2026, 9, 27, 12, tzinfo=UTC))
    counters = InMemoryCounterStore()
    budget = budget or InMemoryBudgetStore(clock=clock, counters=counters)
    app = create_app(
        make_settings(**s), env=env or {}, budget_store=budget, counters=counters
    )
    app.state.rate_limiter.clock = clock
    fake = asker or RecordingAsker()
    app.dependency_overrides[get_asker] = lambda: fake
    return Harness(app, TestClient(app), fake, budget, clock)


UPSTASH_ENV = {
    "UPSTASH_REDIS_REST_URL": "https://example-upstash.io",
    "UPSTASH_REDIS_REST_TOKEN": "test-token",
}


class FakeUpstash:
    """A tiny Redis emulation behind the Upstash REST wire format."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.requests: list[tuple[str, object, str | None]] = []
        self.fail_writes = False

    def run(self, command: list) -> object:
        if command[0] == "EVAL":
            return self.eval_reserve(command)
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

    def eval_reserve(self, command: list) -> object:
        """Python stand-in for RESERVE_SCRIPT (checked against it by payload)."""
        _, script, numkeys, key, amount, limit, ttl, epsilon = command
        assert script == RESERVE_SCRIPT and numkeys == 1
        current = float(self.data.get(key, "0"))
        if current + float(amount) > float(limit) + float(epsilon):
            return [0, repr(current)]
        total = self.run(["INCRBYFLOAT", key, amount])
        self.run(["EXPIRE", key, ttl])
        return [1, total]

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(
            (request.url.path, body, request.headers.get("authorization"))
        )
        is_write = request.url.path == "/multi-exec" or body[0] == "EVAL"
        if self.fail_writes and is_write:
            return httpx.Response(500, json={"error": "ERR write failed"})
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

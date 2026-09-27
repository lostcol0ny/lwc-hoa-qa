"""Test doubles and helpers shared by the web tests."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hoa_qa.budget import InMemoryBudgetStore, InMemoryCounterStore
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
    budget: InMemoryBudgetStore
    clock: FixedClock


def build_harness(
    *,
    env: dict[str, str] | None = None,
    asker: RecordingAsker | None = None,
    **s: object,
) -> Harness:
    clock = FixedClock(datetime(2026, 9, 27, 12, tzinfo=UTC))
    counters = InMemoryCounterStore()
    budget = InMemoryBudgetStore(clock=clock, counters=counters)
    app = create_app(
        make_settings(**s), env=env or {}, budget_store=budget, counters=counters
    )
    app.state.rate_limiter.clock = clock
    fake = asker or RecordingAsker()
    app.dependency_overrides[get_asker] = lambda: fake
    return Harness(app, TestClient(app), fake, budget, clock)

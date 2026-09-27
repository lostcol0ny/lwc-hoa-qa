"""The web app over the real QA core asker, with fake Jev and answer providers."""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from qa_fakes import FINES_DRAFT, FIXTURE, FakeJev, FakeProvider
from web_fakes import QUESTION, FixedClock, build_harness

from hoa_qa import ask as qa
from hoa_qa.ask import QAAsker, QASettings, build_asker
from hoa_qa.budget import InMemoryBudgetStore, Reservation
from hoa_qa.models import Answer, Corpus, Outcome, load_corpus, validate_https
from hoa_qa.web import deps
from hoa_qa.web.app import reservation_amount


@dataclass
class RecordingBudget(InMemoryBudgetStore):
    reserved: list[float] = field(default_factory=list)
    settled: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(clock=FixedClock(datetime(2026, 9, 27, 12, tzinfo=UTC)))

    async def reserve(self, usd: float, budget_usd: float) -> Reservation | None:
        self.reserved.append(usd)
        return await super().reserve(usd, budget_usd)

    async def reconcile(self, reservation: Reservation, actual_usd: float) -> float:
        self.settled.append(actual_usd)
        return await super().reconcile(reservation, actual_usd)


@dataclass
class ClosingJev(FakeJev):
    closed: int = 0

    async def aclose(self) -> None:
        self.closed += 1


@dataclass
class ClosingProvider(FakeProvider):
    closed: int = 0

    async def aclose(self) -> None:
        self.closed += 1


def real_asker() -> tuple[QAAsker, ClosingJev, ClosingProvider]:
    corpus = load_corpus(FIXTURE)
    jev = ClosingJev(
        relevance={"rules-2023-fines": 0.9, "rules-2016-fines": 0.7},
        text_to_id={c.text_clean: c.id for c in corpus.chunks},
    )
    provider = ClosingProvider([FINES_DRAFT])
    asker = build_asker(corpus, QASettings.from_env({}), jev=jev, provider=provider)
    return asker, jev, provider


def test_qa_core_types_satisfy_the_web_protocols() -> None:
    """Typed assignments: pyright fails the build if the contracts drift."""
    asker, _, _ = real_asker()
    web_asker: deps.Asker = asker
    result: deps.AskResult = asyncio.run(web_asker(QUESTION))
    qa_result: qa.AskResult = asyncio.run(asker(QUESTION))
    also: deps.AskResult = qa_result
    assert isinstance(result.answer, Answer) and isinstance(also.answer, Answer)


@pytest.mark.parametrize("env_default", [0.0001, 50.0])
def test_real_asker_end_to_end(env_default: float) -> None:
    asker, jev, provider = real_asker()
    budget = RecordingBudget()
    h = build_harness(
        budget=budget,
        monthly_budget_usd=100.0,
        budget_reserve_per_request_usd=env_default,
    )
    h.app.dependency_overrides.clear()  # the real get_asker
    h.app.state.asker = asker

    with h.client:
        response = h.client.post("/api/ask", json={"question": QUESTION})
    assert response.status_code == 200

    # R = max(env default, asker.max_cost_usd), with the real bound.
    expected_r = max(env_default, asker.max_cost_usd)
    assert budget.reserved == [expected_r]
    assert reservation_amount(h.app.state.settings, asker) == expected_r
    assert 0 < asker.max_cost_usd < 1

    answer = Answer.model_validate(response.json())
    assert answer.outcome == Outcome.answered
    assert "$125" in answer.answer_text
    assert [c.chunk_id for c in answer.citations] == ["rules-2023-fines"]
    for citation in answer.citations:
        assert validate_https(citation.url) == citation.url

    # The real estimated cost was reconciled, and it's within the bound.
    assert len(budget.settled) == 1
    assert 0 < budget.settled[0] <= asker.max_cost_usd
    # Lifespan shutdown closed the asker's clients.
    assert (jev.closed, provider.closed) == (1, 1)


def test_real_get_asker_without_keys_is_503() -> None:
    h = build_harness(env={})
    h.app.dependency_overrides.clear()
    response = h.client.post("/api/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert "isn't set up yet" in response.json()["answer_text"]


def test_real_get_asker_builds_qa_asker_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With keys in the app's env, get_asker builds the real QAAsker once."""
    built: list[QASettings] = []

    def fake_build(corpus: Corpus, settings: QASettings) -> QAAsker:
        built.append(settings)
        asker, _, _ = real_asker()
        return asker

    monkeypatch.setattr(deps, "build_qa_asker", fake_build)
    env = {"TYPESAFE_API_KEY": "ts-test", "ANTHROPIC_API_KEY": "sk-test"}
    h = build_harness(env=env)
    h.app.dependency_overrides.clear()
    for _ in range(2):
        response = h.client.post("/api/ask", json={"question": QUESTION})
        assert response.json()["outcome"] == "answered"
    assert len(built) == 1
    assert built[0].typesafe_api_key is not None
    assert isinstance(h.app.state.asker, QAAsker)

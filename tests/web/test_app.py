import asyncio
import importlib.util
import logging
import os

import pytest
from fastapi.testclient import TestClient
from web_fakes import QUESTION, Harness, RecordingAsker, build_harness, make_settings

from hoa_qa.budget import FailClosedBudgetStore
from hoa_qa.models import Answer, Outcome
from hoa_qa.web import deps as web_deps
from hoa_qa.web.app import create_app
from hoa_qa.web.deps import ServiceUnavailable, build_asker, get_asker
from hoa_qa.web.settings import DISCLAIMER


def ask(h: Harness, question: str = QUESTION, **headers: str):
    return h.client.post("/api/ask", json={"question": question}, headers=headers)


def test_happy_path_returns_answer_json(harness: Harness) -> None:
    response = ask(harness)
    assert response.status_code == 200
    answer = Answer.model_validate(response.json())
    assert answer.outcome == Outcome.answered
    assert answer.answer_text == "You may bring two guests."
    assert harness.asker.questions == [QUESTION]


def test_question_is_stripped_before_the_asker(harness: Harness) -> None:
    ask(harness, f"  {QUESTION}\n")
    assert harness.asker.questions == [QUESTION]


@pytest.mark.parametrize(
    "body",
    [
        {"question": "x" * 501},
        {"question": "   "},
        {"question": ""},
        {},
        {"question": QUESTION, "history": []},
    ],
)
def test_invalid_input_is_422_with_answer_body(harness: Harness, body: dict) -> None:
    response = harness.client.post("/api/ask", json=body)
    assert response.status_code == 422
    answer = Answer.model_validate(response.json())
    assert answer.outcome == Outcome.invalid_input
    assert "x" * 50 not in response.text  # the input isn't echoed back
    assert harness.asker.questions == []


def test_500_chars_is_accepted(harness: Harness) -> None:
    assert ask(harness, "x" * 500).status_code == 200


def test_budget_exhausted_never_calls_asker() -> None:
    h = build_harness(monthly_budget_usd=0.05)
    asyncio.run(h.budget.add_spend(0.05))
    response = ask(h)
    assert response.status_code == 200
    answer = Answer.model_validate(response.json())
    assert answer.outcome == Outcome.budget_exhausted
    assert "https://lakewoodcreekhoa.com/" in answer.answer_text
    assert answer.disclaimer == DISCLAIMER
    assert h.asker.questions == []


def test_production_without_upstash_fails_closed() -> None:
    env = {"VERCEL_ENV": "production", "MONTHLY_BUDGET_USD": "50"}
    app = create_app(env=env)
    assert isinstance(app.state.budget_store, FailClosedBudgetStore)
    fake = RecordingAsker()
    app.dependency_overrides[get_asker] = lambda: fake
    response = TestClient(app).post("/api/ask", json={"question": QUESTION})
    assert response.json()["outcome"] == "budget_exhausted"
    assert fake.questions == []


def test_asker_exception_is_generic_and_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_question = "My neighbor Jane at 12 Elm St keeps parking on the lawn"
    h = build_harness(asker=RecordingAsker(error=ValueError(secret_question)))
    with caplog.at_level(logging.DEBUG):
        response = ask(h, secret_question)
    assert response.status_code == 500
    answer = Answer.model_validate(response.json())
    assert answer.outcome == Outcome.error
    assert secret_question not in response.text
    assert "ValueError" in caplog.text
    assert answer.request_id in caplog.text
    assert secret_question not in caplog.text
    assert "Jane" not in caplog.text


def test_answer_text_is_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    h = build_harness()
    with caplog.at_level(logging.DEBUG):
        ask(h)
    assert "outcome=answered" in caplog.text
    assert QUESTION not in caplog.text
    assert "two guests" not in caplog.text


def test_health_reports_corpus_and_leaks_no_env(
    monkeypatch: pytest.MonkeyPatch, harness: Harness
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "upstash-should-not-leak")
    response = harness.client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "corpus_build_time": "2026-09-27T00:00:00+00:00",
        "chunk_count": 8,
        "documents_url": "https://lakewoodcreekhoa.com/",
    }
    for value in os.environ.values():
        if len(value) >= 8:
            assert value not in response.text


def test_health_without_corpus_is_503(tmp_path) -> None:
    h = build_harness(corpus_path=tmp_path / "missing.json")
    response = h.client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_missing_qa_core_is_503_with_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_qa_core(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(web_deps.importlib, "import_module", no_qa_core)
    h = build_harness()
    h.app.dependency_overrides.clear()  # use the real get_asker
    response = ask(h)
    assert response.status_code == 503
    body = Answer.model_validate(response.json())
    assert body.outcome == Outcome.error
    assert "isn't set up yet" in body.answer_text


def test_qa_core_without_api_keys_is_503(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    if importlib.util.find_spec("hoa_qa.ask") is None:
        pytest.skip("QA core not installed")
    for var in ("TYPESAFE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    h = build_harness()
    h.app.dependency_overrides.clear()  # use the real get_asker
    with caplog.at_level(logging.DEBUG):
        response = ask(h)
    assert response.status_code == 503
    assert "isn't set up yet" in Answer.model_validate(response.json()).answer_text
    assert "building the asker failed" in caplog.text
    assert QUESTION not in caplog.text


def test_fake_asker_flag_serves_canned_answers() -> None:
    asker = build_asker(make_settings(fake_asker=True))
    result = asyncio.run(asker("anything"))
    assert result.answer.outcome == Outcome.answered
    assert result.answer.citations[0].chunk_id == "bylaws-3.4"


def test_missing_corpus_makes_asker_unavailable(tmp_path) -> None:
    with pytest.raises(ServiceUnavailable):
        build_asker(make_settings(fake_asker=True, corpus_path=tmp_path / "x.json"))


def test_shutdown_calls_asker_aclose() -> None:
    closed: list[bool] = []

    class ClosingAsker(RecordingAsker):
        async def aclose(self) -> None:
            closed.append(True)

    h = build_harness()
    with h.client:
        h.app.state.asker = ClosingAsker()
    assert closed == [True]

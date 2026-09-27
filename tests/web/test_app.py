import asyncio
import logging
import os

import pytest
from fastapi.testclient import TestClient
from web_fakes import (
    FIXTURE_CORPUS,
    QUESTION,
    Harness,
    RecordingAsker,
    build_harness,
    make_settings,
)

from hoa_qa.budget import FailClosedBudgetStore
from hoa_qa.models import Answer, Outcome
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
        "budget_config": "ok",
    }
    for value in os.environ.values():
        if len(value) >= 8:
            assert value not in response.text


def test_health_without_corpus_is_503(tmp_path) -> None:
    h = build_harness(corpus_path=tmp_path / "missing.json")
    response = h.client.get("/api/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_qa_core_without_api_keys_is_503(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
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


def test_fake_asker_is_impossible_in_production() -> None:
    env = {
        "VERCEL_ENV": "production",
        "HOA_QA_FAKE_ASKER": "1",
        "MONTHLY_BUDGET_USD": "5",
        "CORPUS_PATH": str(FIXTURE_CORPUS),
    }
    # From the environment: the flag is dropped.
    app = create_app(env=env)
    assert app.state.settings.fake_asker is False
    # Even a hand-built settings object with the flag set can't get the fake:
    # without API keys the real asker can't be built, so it's a 503.
    forced = make_settings(production=True, fake_asker=True)
    with pytest.raises(ServiceUnavailable):
        build_asker(forced, {})
    h = build_harness(env=env, production=True, fake_asker=True)
    h.app.dependency_overrides.clear()  # use the real get_asker
    response = ask(h)
    assert response.status_code == 503
    assert h.app.state.asker is None


def test_health_flags_budget_below_reservation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h = build_harness(monthly_budget_usd=0.01, budget_reserve_per_request_usd=0.05)
    with caplog.at_level(logging.WARNING), h.client:
        body = h.client.get("/api/health").json()
    assert body["budget_config"] == "budget_below_reservation"
    assert "below the per-request reservation" in caplog.text


def test_budget_warning_uses_the_askers_max_cost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Expensive(RecordingAsker):
        max_cost_usd = 0.5

    h = build_harness(asker=Expensive(), monthly_budget_usd=0.2)
    with caplog.at_level(logging.WARNING), h.client:
        assert h.client.get("/api/health").json()["budget_config"] == "ok"
        assert "below the per-request reservation" not in caplog.text
        assert ask(h).json()["outcome"] == "budget_exhausted"
        h.app.state.asker = h.asker  # what get_asker stores in production
        assert (
            h.client.get("/api/health").json()["budget_config"]
            == "budget_below_reservation"
        )
    assert caplog.text.count("below the per-request reservation") == 1


APP_PAIRING_CASES = [
    pytest.param(
        {
            "UPSTASH_REDIS_REST_URL": "https://secret-upstash.example.com",
            "UPSTASH_REDIS_REST_TOKEN": "secret-upstash-token-12345",
        },
        (),
        False,
        "ok",
        id="upstash_pair_only",
    ),
    pytest.param(
        {
            "KV_REST_API_URL": "https://secret-kv.example.com",
            "KV_REST_API_TOKEN": "secret-kv-token-67890",
        },
        (),
        False,
        "ok",
        id="kv_pair_only",
    ),
    pytest.param(
        {
            "UPSTASH_REDIS_REST_URL": "https://secret-upstash.example.com",
            "UPSTASH_REDIS_REST_TOKEN": "secret-upstash-token-12345",
            "KV_REST_API_URL": "https://secret-kv.example.com",
            "KV_REST_API_TOKEN": "secret-kv-token-67890",
        },
        (),
        False,
        "ok",
        id="both_complete_upstash_wins",
    ),
    pytest.param(
        {
            "UPSTASH_REDIS_REST_URL": "https://secret-upstash.example.com",
            "KV_REST_API_TOKEN": "secret-kv-token-67890",
        },
        ("KV_REST_API_URL", "UPSTASH_REDIS_REST_TOKEN"),
        True,
        "redis_config_incomplete: missing KV_REST_API_URL, UPSTASH_REDIS_REST_TOKEN",
        id="upstash_url_and_kv_token_mixed_error",
    ),
    pytest.param(
        {"KV_REST_API_URL": "https://secret-kv.example.com"},
        ("KV_REST_API_TOKEN",),
        True,
        "redis_config_incomplete: missing KV_REST_API_TOKEN",
        id="kv_url_only_error",
    ),
    pytest.param(
        {
            "UPSTASH_REDIS_REST_TOKEN": "secret-upstash-token-12345",
            "KV_REST_API_URL": "https://secret-kv.example.com",
            "KV_REST_API_TOKEN": "secret-kv-token-67890",
        },
        ("UPSTASH_REDIS_REST_URL",),
        True,
        "redis_config_incomplete: missing UPSTASH_REDIS_REST_URL",
        id="upstash_token_only_with_complete_kv_pair_error",
    ),
    pytest.param(
        {"KV_REST_API_READ_ONLY_TOKEN": "secret-ro-token-11111"},
        (),
        False,
        "redis_not_configured",
        id="read_only_token_alone_ignored",
    ),
    pytest.param(
        {
            "KV_REST_API_URL": "https://secret-kv.example.com",
            "KV_REST_API_READ_ONLY_TOKEN": "secret-ro-token-11111",
        },
        ("KV_REST_API_TOKEN",),
        True,
        "redis_config_incomplete: missing KV_REST_API_TOKEN",
        id="kv_url_with_read_only_token_error",
    ),
    pytest.param(
        {
            "UPSTASH_REDIS_REST_URL": "https://secret-upstash.example.com",
            "UPSTASH_REDIS_REST_TOKEN": "secret-upstash-token-12345",
            "KV_REST_API_READ_ONLY_TOKEN": "secret-ro-token-11111",
        },
        (),
        False,
        "ok",
        id="upstash_complete_with_read_only_token_ignored",
    ),
    pytest.param(
        {
            "KV_REST_API_URL": "https://secret-kv.example.com",
            "KV_REST_API_TOKEN": "secret-kv-token-67890",
            "KV_REST_API_READ_ONLY_TOKEN": "secret-ro-token-11111",
        },
        (),
        False,
        "ok",
        id="kv_complete_with_read_only_token_ignored",
    ),
    pytest.param(
        {},
        (),
        False,
        "redis_not_configured",
        id="neither_pair_configured",
    ),
]


@pytest.mark.parametrize(
    ("redis_env", "missing_vars", "is_error", "expected_prod_health"),
    APP_PAIRING_CASES,
)
def test_redis_pairing_startup_health_and_ask_behavior(
    caplog: pytest.LogCaptureFixture,
    redis_env: dict[str, str],
    missing_vars: tuple[str, ...],
    is_error: bool,
    expected_prod_health: str,
) -> None:
    # 1. Production behavior
    prod_env = {
        "VERCEL_ENV": "production",
        "MONTHLY_BUDGET_USD": "10",
        "CORPUS_PATH": str(FIXTURE_CORPUS),
        **redis_env,
    }
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        prod_app = create_app(env=prod_env)
        fake = RecordingAsker()
        prod_app.dependency_overrides[get_asker] = lambda: fake
        client = TestClient(prod_app)
        health_resp = client.get("/api/health")

    assert health_resp.status_code == 200
    health_body = health_resp.json()
    assert health_body["budget_config"] == expected_prod_health

    # Health reason text and logs must name missing variables when incomplete
    for var in missing_vars:
        assert var in health_body["budget_config"]
        assert var in caplog.text

    # MUST NEVER LEAK SECRETS OR URLS IN HEALTH OR LOGS
    for key, val in redis_env.items():
        assert val not in health_resp.text, f"Secret {key} leaked in /api/health"
        assert val not in caplog.text, f"Secret {key} leaked in logs"

    if is_error or expected_prod_health == "redis_not_configured":
        ask_resp = client.post("/api/ask", json={"question": QUESTION})
        assert ask_resp.json()["outcome"] == "budget_exhausted"
        assert fake.questions == []
        expected_msg = (
            "Redis configuration incomplete"
            if is_error
            else "Redis is not configured in production"
        )
        assert any(
            record.levelno == logging.ERROR and expected_msg in record.message
            for record in caplog.records
        )

    # 2. Development behavior
    dev_env = {
        "MONTHLY_BUDGET_USD": "10",
        "CORPUS_PATH": str(FIXTURE_CORPUS),
        **redis_env,
    }
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        dev_app = create_app(env=dev_env)
        dev_fake = RecordingAsker()
        dev_app.dependency_overrides[get_asker] = lambda: dev_fake
        dev_client = TestClient(dev_app)
        dev_health = dev_client.get("/api/health")

    assert dev_health.status_code == 200
    assert dev_health.json()["budget_config"] == "ok"

    for key, val in redis_env.items():
        assert val not in dev_health.text, f"Secret {key} leaked in dev health"
        assert val not in caplog.text, f"Secret {key} leaked in dev logs"

    if is_error:
        # In dev, incomplete pair logs a warning but allows in-memory counter store
        assert any(
            record.levelno == logging.WARNING
            and "Redis configuration incomplete" in record.message
            for record in caplog.records
        )
        for var in missing_vars:
            assert var in caplog.text
        # Questions succeed in dev via in-memory store
        dev_ask = dev_client.post("/api/ask", json={"question": QUESTION})
        assert dev_ask.json()["outcome"] == "answered"
    elif expected_prod_health == "redis_not_configured":
        # Unconfigured in dev: normal in-memory dev mode, no warnings
        assert not any(
            "Redis" in record.message
            for record in caplog.records
            if record.levelno >= logging.WARNING
        )
        dev_ask = dev_client.post("/api/ask", json={"question": QUESTION})
        assert dev_ask.json()["outcome"] == "answered"

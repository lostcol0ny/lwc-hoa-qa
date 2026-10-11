"""The operator monitor: token gate, the full trace, and the metadata-only feed."""

import asyncio
import json
import logging
from datetime import UTC, datetime

import httpx
import pytest
from qa_fakes import FINES_DRAFT, FIXTURE, FakeJev, FakeProvider
from web_fakes import UPSTASH_ENV, FakeUpstash, build_harness

from hoa_qa.ask import QASettings, build_asker
from hoa_qa.budget import InMemoryCounterStore, UpstashCounterStore
from hoa_qa.cache import AnswerCache, CacheEntry, corpus_fingerprint
from hoa_qa.models import Answer, Outcome, load_corpus
from hoa_qa.web import monitor

TOKEN = "operator-token-with-plenty-of-length"
AUTH = {"x-monitor-token": TOKEN}
ENV = {"MONITOR_TOKEN": TOKEN}
SECRET = "What is the fine for a third violation at 12 Quail Run?"
QUIET = "What are the quiet hours in the neighborhood?"


def real_harness(env: dict[str, str] | None = None, *, cache: bool = False):
    corpus = load_corpus(FIXTURE)
    jev = FakeJev(
        relevance={"rules-2023-fines": 0.9, "rules-2016-fines": 0.7},
        text_to_id={c.text_clean: c.id for c in corpus.chunks},
        match={QUIET: 0.97},
    )
    provider = FakeProvider([FINES_DRAFT])
    stored = None
    if cache:
        answer = Answer(
            request_id="built",
            outcome=Outcome.answered,
            answer_text="10 p.m. to 8 a.m.",
            citations=(),
            confidence=1.0,
            conflicts_noted=(),
            disclaimer="d",
        )
        stored = AnswerCache(
            corpus_fingerprint=corpus_fingerprint(corpus),
            built_at=datetime(2026, 10, 9, tzinfo=UTC),
            jev_model="jev-1.13.0",
            answer_model="claude-haiku-4-5",
            entries=(CacheEntry(id="quiet-hours", question=QUIET, answer=answer),),
        )
    asker = build_asker(
        corpus, QASettings.from_env({}), jev=jev, provider=provider, cache=stored
    )
    h = build_harness(env=ENV if env is None else env, monthly_budget_usd=100.0)
    h.app.dependency_overrides.clear()
    h.app.state.asker = asker
    return h, jev, provider


@pytest.mark.parametrize(
    "env",
    [{}, {"MONITOR_TOKEN": ""}, {"MONITOR_TOKEN": "too-short"}],
    ids=["unset", "empty", "short"],
)
def test_monitor_is_off_without_a_long_enough_token(env: dict[str, str]) -> None:
    h, _, _ = real_harness(env)
    for headers in ({}, AUTH, {"x-monitor-token": env.get("MONITOR_TOKEN", "")}):
        assert h.client.get("/api/monitor/config", headers=headers).status_code == 404
        assert h.client.get("/api/monitor/recent", headers=headers).status_code == 404
        response = h.client.post(
            "/api/monitor/ask", json={"question": SECRET}, headers=headers
        )
        assert response.status_code == 404
    # And with the monitor off, ordinary traffic records nothing.
    assert h.client.post("/api/ask", json={"question": SECRET}).status_code == 200
    stored = asyncio.run(h.app.state.recent_store.recent(monitor.RECENT_KEY, 10))
    assert stored == []


@pytest.mark.parametrize(
    "headers",
    [{}, {"x-monitor-token": "wrong"}, {"x-monitor-token": TOKEN + "x"}],
    ids=["missing", "wrong", "prefix"],
)
def test_wrong_token_looks_like_a_missing_route(headers: dict[str, str]) -> None:
    h, jev, _ = real_harness()
    missing = h.client.get("/api/no-such-route")
    for response in (
        h.client.get("/api/monitor/config", headers=headers),
        h.client.get("/api/monitor/recent", headers=headers),
        h.client.post("/api/monitor/ask", json={"question": SECRET}, headers=headers),
    ):
        assert response.status_code == 404
        assert response.json() == missing.json()
    assert jev.calls == [], "nothing ran"


def test_cross_origin_operator_request_is_refused() -> None:
    h, jev, _ = real_harness()
    response = h.client.post(
        "/api/monitor/ask",
        json={"question": SECRET},
        headers={**AUTH, "origin": "https://evil.example"},
    )
    assert response.status_code == 404 and jev.calls == []


def test_config_has_what_the_page_needs_and_no_secrets() -> None:
    h, _, _ = real_harness()
    response = h.client.get("/api/monitor/config", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    config = response.json()
    corpus = load_corpus(FIXTURE)
    assert len(config["chunks"]) == len(corpus.chunks)
    assert config["models"] == {"jev": "jev-1.13.0", "answer": "claude-haiku-4-5"}
    assert config["thresholds"]["cache"] == 0.9
    assert "You answer neighbors' questions" in config["system_prompt"]
    assert TOKEN not in response.text and "api_key" not in response.text


def test_operator_question_returns_the_full_trace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    h, jev, provider = real_harness()
    caplog.set_level(logging.DEBUG)
    # More than the per-IP hourly limit: the token, not the limiter, is the gate.
    responses = [
        h.client.post("/api/monitor/ask", json={"question": SECRET}, headers=AUTH)
        for _ in range(12)
    ]
    assert all(response.status_code == 200 for response in responses)
    trace = responses[-1].json()
    assert trace["question"] == SECRET and trace["outcome"] == "answered"
    assert "$125" in trace["answer_text"]
    assert trace["gate"] == {"score": 0.95, "passed": True}
    assert trace["cache"] is None, "no cache is loaded in this harness"
    corpus = load_corpus(FIXTURE)
    planned = [i for batch in trace["sweep"]["plan"] for i in batch]
    assert sorted(planned) == list(range(len(corpus.chunks)))
    assert trace["sweep"]["top"][0]["selected"] is True
    assert trace["sweep"]["sample"]["state"]["question"] == SECRET
    assert [p["id"] for p in trace["passages"]] == [
        "rules-2023-fines",
        "rules-2016-fines",
    ]
    [attempt] = trace["attempts"]
    assert attempt["valid"] and attempt["feedback"] == ""
    [claim] = attempt["claims"]
    assert claim["kept"] and claim["citations"][0]["quote_ok"]
    assert attempt["support"]["question"]["instructions"]
    assert set(trace["stage_ms"]) == {"gate", "cache", "sweep", "answer", "verify"}
    assert trace["cost_usd"] > 0 and trace["tokens"]["jev"] == sum(jev.billed) / 12
    assert len(provider.prompts) == 12
    # The operator's question is returned to them and nowhere else.
    assert "Quail Run" not in caplog.text
    stored = asyncio.run(h.app.state.recent_store.recent(monitor.RECENT_KEY, 50))
    assert len(stored) == 12 and "Quail Run" not in "".join(stored)


def test_operator_question_shows_a_cache_hit() -> None:
    h, _, provider = real_harness(cache=True)
    response = h.client.post(
        "/api/monitor/ask", json={"question": "When are quiet hours?"}, headers=AUTH
    )
    trace = response.json()
    assert trace["answer_text"] == "10 p.m. to 8 a.m."
    assert trace["cache"]["hit"] == "quiet-hours" and not trace["cache"]["exact"]
    assert trace["cache"]["probability"] == 0.97
    assert trace["cache"]["matched_question"] == QUIET
    assert trace["cache"]["sample"]["state"]["known"] == {"k0": QUIET}
    assert trace["sweep"] is None and trace["attempts"] == []
    assert provider.prompts == []

    exact = h.client.post("/api/monitor/ask", json={"question": QUIET}, headers=AUTH)
    assert exact.json()["cache"]["exact"] is True
    assert exact.json()["gate"] is None, "an exact curated question skips the gate"


def test_traffic_feed_is_numbers_and_codes_only() -> None:
    h, _, _ = real_harness(cache=True)
    assert h.client.post("/api/ask", json={"question": SECRET}).status_code == 200
    assert (
        h.client.post(
            "/api/ask", json={"question": "When are quiet hours?"}
        ).status_code
        == 200
    )
    h.client.post("/api/monitor/ask", json={"question": SECRET}, headers=AUTH)
    response = h.client.get("/api/monitor/recent", headers=AUTH)
    assert response.status_code == 200
    feed = response.json()
    assert [row["operator"] for row in feed["rows"]] == [True, False, False]
    assert [row["cache"] for row in feed["rows"]] == [None, "quiet-hours", None]
    assert feed["summary"]["requests"] == 3 and feed["summary"]["cache_hits"] == 1
    allowed = {
        "t", "outcome", "reason", "latency_ms", "stage_ms", "cache", "cost_usd",
        "jev_tokens", "answer_in", "answer_out", "citations", "operator",
    }  # fmt: skip
    for row in feed["rows"]:
        assert set(row) == allowed
        assert row["outcome"] == "answered" and row["latency_ms"] >= 0
    # No question, answer, quote, or passage text anywhere in the feed.
    for secret in ("Quail Run", "violation", "$125", "quiet hours", "10 p.m."):
        assert secret not in response.text


def test_feed_write_failure_never_fails_the_request() -> None:
    h, _, _ = real_harness()

    class Broken:
        async def push_recent(self, *args: object) -> None:
            raise RuntimeError("store down")

        async def recent(self, *args: object) -> list[str]:
            raise RuntimeError("store down")

    h.app.state.recent_store = Broken()
    assert h.client.post("/api/ask", json={"question": SECRET}).status_code == 200
    response = h.client.get("/api/monitor/recent", headers=AUTH)
    assert response.status_code == 200 and "error" in response.json()


def test_recent_list_is_capped_and_newest_first() -> None:
    async def scenario(store) -> list[str]:
        for n in range(5):
            await store.push_recent("k", json.dumps({"n": n}), 3, 60)
        return await store.recent("k", 10)

    fake = FakeUpstash()
    upstash = UpstashCounterStore(
        UPSTASH_ENV["UPSTASH_REDIS_REST_URL"],
        UPSTASH_ENV["UPSTASH_REDIS_REST_TOKEN"],
        client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)),
    )
    for store in (InMemoryCounterStore(), upstash):
        rows = [json.loads(r)["n"] for r in asyncio.run(scenario(store))]
        assert rows == [4, 3, 2]
    assert fake.ttls["k"] == 60


def test_each_answer_flushes_its_trace_before_responding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serverless function freezes after responding, so traces go out first."""
    flushed: list[str] = []

    async def fake_flush(timeout_ms: int = 0) -> None:
        flushed.append("flush")

    monkeypatch.setattr("hoa_qa.web.app.telemetry.flush_async", fake_flush)
    h, _, _ = real_harness()
    assert h.client.get("/api/health").status_code == 200
    assert h.client.get("/api/monitor/recent", headers=AUTH).status_code == 200
    assert flushed == [], "only question endpoints flush"
    assert h.client.post("/api/ask", json={"question": SECRET}).status_code == 200
    h.client.post("/api/monitor/ask", json={"question": SECRET}, headers=AUTH)
    assert flushed == ["flush", "flush"]

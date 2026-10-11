"""Tracing: the span tree for a question, and what is (not) recorded in it."""

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext, StatusCode
from qa_fakes import FINES_DRAFT, FakeJev, FakeProvider, claim, draft
from test_cache import QUIET, make_cache

from hoa_qa import telemetry
from hoa_qa.ask import QASettings, build_asker
from hoa_qa.models import Corpus, Outcome

QUESTION = "What is the fine for a third violation at 12 Quail Run?"


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    telemetry.use(provider, content=False, environment="test")
    yield exporter
    telemetry.use(None)
    provider.shutdown()


@pytest.fixture
def with_content(spans: InMemorySpanExporter) -> InMemorySpanExporter:
    telemetry._state.content = True
    return spans


def ask(corpus, settings, jev, provider, question=QUESTION, cache=None):
    asker = build_asker(corpus, settings, jev=jev, provider=provider, cache=cache)
    return asyncio.run(asker(question))


def ctx(span: ReadableSpan) -> SpanContext:
    context = span.get_span_context()
    assert context is not None
    return context


def tree(exporter: InMemorySpanExporter) -> dict[str, list[str]]:
    """Span name -> its children's names, in start order."""
    finished = sorted(exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    names = {ctx(s).span_id: s.name for s in finished}
    children: dict[str, list[str]] = {}
    for s in finished:
        parent = names[s.parent.span_id] if s.parent else "(root)"
        children.setdefault(parent, []).append(s.name)
    return children


def by_name(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def attr(span: ReadableSpan, key: str) -> Any:
    assert span.attributes is not None
    return span.attributes.get(key)


def meta(span: ReadableSpan, key: str) -> Any:
    return attr(span, f"langfuse.observation.metadata.{key}")


def everything(exporter: InMemorySpanExporter) -> str:
    return json.dumps(
        [
            {"name": s.name, "attributes": dict(s.attributes or {})}
            for s in exporter.get_finished_spans()
        ],
        default=str,
    )


def test_one_trace_per_question_with_a_span_per_stage_and_model_request(
    corpus: Corpus, settings: QASettings, jev: FakeJev, spans: InMemorySpanExporter
) -> None:
    result = ask(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    assert result.answer.outcome is Outcome.answered
    children = tree(spans)
    assert children["(root)"] == ["ask"]
    assert children["ask"] == ["gate", "sweep", "draft 1", "verify 1"]
    assert children["gate"] == ["jev gate"]
    assert children["sweep"] == ["jev sweep batch"] * 7  # one per document
    assert children["draft 1"] == ["answer model draft"]
    assert children["verify 1"] == ["jev support"]
    assert len({ctx(s).trace_id for s in spans.get_finished_spans()}) == 1

    [root] = by_name(spans, "ask")
    assert attr(root, "langfuse.trace.name") == "ask"
    assert attr(root, "langfuse.trace.tags") == ("answered",)
    assert attr(root, "langfuse.environment") == "test"
    assert meta(root, "outcome") == "answered"
    assert meta(root, "request_id") == result.answer.request_id
    assert meta(root, "cost_usd") == round(result.estimated_cost_usd, 6)
    kinds = {
        s.name: attr(s, "langfuse.observation.type") for s in spans.get_finished_spans()
    }
    assert kinds["gate"] == "guardrail" and kinds["sweep"] == "retriever"
    assert kinds["verify 1"] == "evaluator"
    assert kinds["jev gate"] == kinds["answer model draft"] == "generation"


def test_model_spans_carry_usage_and_cost_that_add_up(
    corpus: Corpus, settings: QASettings, jev: FakeJev, spans: InMemorySpanExporter
) -> None:
    result = ask(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    jev_spans = [s for s in spans.get_finished_spans() if s.name.startswith("jev ")]
    assert all(attr(s, "gen_ai.request.model") == "jev-1.13.0" for s in jev_spans)
    tokens = sum(attr(s, "gen_ai.usage.input_tokens") for s in jev_spans)
    assert tokens == result.jev_input_tokens
    [generation] = by_name(spans, "answer model draft")
    assert attr(generation, "gen_ai.request.model") == "claude-haiku-4-5"
    assert attr(generation, "gen_ai.usage.input_tokens") == result.answer_input_tokens
    assert attr(generation, "gen_ai.usage.output_tokens") == result.answer_output_tokens
    cost = sum(
        json.loads(attr(s, "langfuse.observation.cost_details"))["total"]
        for s in [*jev_spans, generation]
    )
    assert cost == pytest.approx(result.estimated_cost_usd)


def test_without_content_tracing_no_text_is_recorded(
    corpus: Corpus, settings: QASettings, jev: FakeJev, spans: InMemorySpanExporter
) -> None:
    result = ask(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    recorded = everything(spans)
    for text in ("Quail Run", "third violation", "$125", "You answer neighbors"):
        assert text not in recorded
    assert result.answer.answer_text not in recorded
    # Numbers, ids and codes are still there.
    [sweep] = by_name(spans, "sweep")
    selected = json.loads(attr(sweep, "langfuse.observation.output"))["selected"]
    assert selected[0] == {"chunk_id": "rules-2023-fines", "score": 0.9}
    [gate] = by_name(spans, "gate")
    assert json.loads(attr(gate, "langfuse.observation.output")) == {
        "probability": 0.95,
        "passed": True,
    }
    [verify] = by_name(spans, "verify 1")
    assert meta(verify, "kept") == 1 and meta(verify, "failed") == "[]"
    assert attr(verify, "langfuse.observation.output") is None


def test_with_content_tracing_the_data_flow_is_visible(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    with_content: InMemorySpanExporter,
) -> None:
    result = ask(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    [root] = by_name(with_content, "ask")
    assert attr(root, "langfuse.observation.input") == QUESTION
    assert result.answer.answer_text in attr(root, "langfuse.observation.output")
    [generation] = by_name(with_content, "answer model draft")
    messages = json.loads(attr(generation, "langfuse.observation.input"))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "3rd violation: $125." in messages[1]["content"], "the passages it was sent"
    assert "claims" in json.loads(attr(generation, "langfuse.observation.output"))
    [verify] = by_name(with_content, "verify 1")
    [verdict] = json.loads(attr(verify, "langfuse.observation.output"))
    assert verdict["kept"] and verdict["cited"] == ["rules-2023-fines"]
    # A sweep request names its passages and their size, never their text:
    # every question sweeps the whole corpus.
    batch = by_name(with_content, "jev sweep batch")[0]
    state = json.loads(attr(batch, "langfuse.observation.input"))["state"]
    assert state["question"] == QUESTION
    assert all(set(p) == {"heading", "chars"} for p in state["passages"].values())


def test_a_retry_shows_as_a_second_draft_and_verify(
    corpus: Corpus, settings: QASettings, jev: FakeJev, spans: InMemorySpanExporter
) -> None:
    bad = draft(claim("Not in the text.", ("rules-2023-fines", "not in the text")))
    ask(corpus, settings, jev, FakeProvider([bad, FINES_DRAFT]))
    assert tree(spans)["ask"] == [
        "gate",
        "sweep",
        "draft 1",
        "verify 1",
        "draft 2",
        "verify 2",
    ]
    [first] = by_name(spans, "verify 1")
    assert meta(first, "kept") == 0 and meta(first, "failed") == '["no_valid_quote"]'


def test_a_cache_hit_is_a_short_trace(
    corpus: Corpus, settings: QASettings, jev: FakeJev, spans: InMemorySpanExporter
) -> None:
    jev.match = {QUIET: 0.97}
    result = ask(
        corpus,
        settings,
        jev,
        FakeProvider([FINES_DRAFT]),
        "When are quiet hours?",
        cache=make_cache(corpus),
    )
    assert result.cache_entry == "quiet-hours"
    children = tree(spans)
    assert sorted(children["ask"]) == ["cache", "gate"]
    assert children["cache"] == ["jev cache match"]
    [cache] = by_name(spans, "cache")
    assert json.loads(attr(cache, "langfuse.observation.output")) == {
        "hit": "quiet-hours",
        "probability": 0.97,
    }
    [root] = by_name(spans, "ask")
    assert meta(root, "cache_entry") == "quiet-hours"


def test_a_failed_stage_is_marked_by_error_type_only(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    with_content: InMemorySpanExporter,
) -> None:
    jev.fail = lambda state: state.get("document") == "LWC Bylaws"
    result = ask(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    assert result.answer.outcome is Outcome.error
    [sweep] = by_name(with_content, "sweep")
    assert sweep.status.status_code is StatusCode.ERROR
    assert sweep.status.description == "PartialCostError"
    failed = [s for s in by_name(with_content, "jev sweep batch") if not s.status.is_ok]
    assert [s.status.description for s in failed] == ["RuntimeError"]
    assert all(not s.events for s in with_content.get_finished_spans()), "no tracebacks"
    [root] = by_name(with_content, "ask")
    assert root.status.status_code is StatusCode.ERROR
    assert meta(root, "outcome") == "error"


def test_tracing_off_records_nothing_and_changes_nothing(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    assert not telemetry.enabled()
    with telemetry.span("anything") as span:
        span.set(a=1)
        span.input("text")
        span.output("text")
    result = ask(corpus, settings, jev, FakeProvider([FINES_DRAFT]))
    assert result.answer.outcome is Outcome.answered
    telemetry.flush()
    asyncio.run(telemetry.flush_async())


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"LANGFUSE_PUBLIC_KEY": "pk-lf-x"},
        {"LANGFUSE_SECRET_KEY": "sk-lf-x"},
        {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-x",
            "LANGFUSE_SECRET_KEY": "sk-lf-x",
            "LANGFUSE_BASE_URL": "http://insecure.example",
        },
    ],
    ids=["no keys", "public only", "secret only", "not https"],
)
def test_configure_stays_off_without_both_keys_and_https(env: dict[str, str]) -> None:
    assert telemetry.configure(env) is False
    assert not telemetry.enabled()


def test_configure_points_the_exporter_at_langfuse() -> None:
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk-lf-x",
        "LANGFUSE_SECRET_KEY": "sk-lf-y",
        "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com/",
        "TRACE_CONTENT": "1",
        "VERCEL_ENV": "production",
    }
    try:
        assert telemetry.configure(env) is True
        assert telemetry.content_enabled()
        assert telemetry._state.environment == "production"
        provider = telemetry._state.provider
        assert provider is not None
        endpoint, headers = telemetry.langfuse_target(env) or ("", {})
        assert endpoint == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
        # base64("pk-lf-x:sk-lf-y")
        assert headers["Authorization"] == "Basic cGstbGYteDpzay1sZi15"
        assert headers["x-langfuse-ingestion-version"] == "4"
        # A second call keeps the first provider.
        assert telemetry.configure({**env, "TRACE_CONTENT": ""}) is True
        assert telemetry._state.provider is provider and telemetry.content_enabled()
    finally:
        provider = telemetry._state.provider
        telemetry.use(None)
        if provider is not None:
            provider.shutdown()


def test_content_tracing_is_off_unless_asked_for() -> None:
    env = {"LANGFUSE_PUBLIC_KEY": "pk-lf-x", "LANGFUSE_SECRET_KEY": "sk-lf-y"}
    try:
        assert telemetry.configure(env) is True
        assert telemetry.enabled() and not telemetry.content_enabled()
    finally:
        provider = telemetry._state.provider
        telemetry.use(None)
        if provider is not None:
            provider.shutdown()

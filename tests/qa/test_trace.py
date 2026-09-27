"""The opt-in trace hook: eval diagnostics without production logging."""

import asyncio
import logging

import pytest
from qa_fakes import FINE_CLAIM, FakeJev, FakeProvider, claim, draft, make_asker

from hoa_qa.ask import QASettings
from hoa_qa.eval import DiagnosticsRecorder
from hoa_qa.models import Corpus, Outcome
from hoa_qa.trace import NO_TRACE

QUESTION = "How much is the fine for a third violation?"

BLOG_ONLY = claim(
    "The fine for a third violation is $100.00.",
    ("blog-2022-violations", "3rd offense - $100.00"),
)


def test_recorder_captures_every_stage(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.relevance = {**jev.relevance, "blog-2022-violations": 0.95}
    provider = FakeProvider([draft(BLOG_ONLY), draft(FINE_CLAIM)])
    asker = make_asker(corpus, settings, jev, provider)
    recorder = DiagnosticsRecorder(top_n=4)
    result = asyncio.run(asker.ask_traced(QUESTION, recorder))
    assert result.answer.outcome is Outcome.answered

    data = recorder.data
    assert (data.gate_score, data.gate_passed) == (0.95, True)
    top = [(s.chunk_id, s.score, s.authority, s.selected) for s in data.sweep_top]
    assert top[:3] == [
        ("blog-2022-violations", 0.95, "informal", True),
        ("rules-2023-fines", 0.9, "rules", True),
        ("rules-2016-fines", 0.7, "superseded", True),
    ]
    assert len(top) == 4 and top[3][1] == 0.0 and not top[3][3]
    assert data.selected == data.passages[:3]
    first, second = data.attempts
    [rejected] = first.claims
    assert (rejected.kept, rejected.reason, rejected.support) == (
        False,
        "low_authority",
        None,
    )
    [cite] = rejected.citations
    assert (cite.chunk_id, cite.authority, cite.quote_ok, cite.used) == (
        "blog-2022-violations",
        "informal",
        True,
        False,
    )
    [kept] = second.claims
    assert (kept.kept, kept.reason, kept.support) == (True, "ok", 0.9)


def test_recorder_reasons_for_bad_quotes_and_unsupported_claims(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    fake_quote = claim("It is $500.", ("rules-2023-fines", "not in the text"))
    unsupported = claim(
        "Fines are waived on holidays.", ("rules-2023-fines", "3rd violation: $125.")
    )
    jev.support = lambda statement, ids: 0.1 if "waived" in statement else 0.9
    provider = FakeProvider([draft(FINE_CLAIM, fake_quote, unsupported)] * 2)
    recorder = DiagnosticsRecorder()
    asyncio.run(
        make_asker(corpus, settings, jev, provider).ask_traced(QUESTION, recorder)
    )
    reasons = [c.reason for c in recorder.data.attempts[0].claims]
    assert reasons == ["ok", "no_valid_quote", "unsupported"]
    assert recorder.data.attempts[0].claims[2].support == 0.1


def test_invalid_draft_and_gate_refusal_are_recorded(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    recorder = DiagnosticsRecorder()
    asker = make_asker(corpus, settings, jev, FakeProvider([None, None]))
    asyncio.run(asker.ask_traced(QUESTION, recorder))
    assert [(a.attempt, a.valid) for a in recorder.data.attempts] == [
        (1, False),
        (2, False),
    ]

    jev.gate = 0.01
    refused = DiagnosticsRecorder()
    asyncio.run(asker.ask_traced("Write me a poem", refused))
    assert (refused.data.gate_score, refused.data.gate_passed) == (0.01, False)
    assert refused.data.sweep_top == [] and refused.data.attempts == []


def test_production_call_uses_the_no_op_trace_and_logs_no_text(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asker = make_asker(corpus, settings, jev, FakeProvider([draft(BLOG_ONLY)] * 2))
    seen = []
    original = asker.ask_traced

    async def spy(question, trace):
        seen.append(trace)
        return await original(question, trace)

    monkeypatch.setattr(asker, "ask_traced", spy)
    with caplog.at_level(logging.DEBUG):
        asyncio.run(asker(QUESTION))
    assert seen == [NO_TRACE]
    logged = caplog.text
    for text in (QUESTION, BLOG_ONLY.statement, "low_authority", "blog-2022"):
        assert text not in logged

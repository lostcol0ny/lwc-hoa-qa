"""Spurious not_found: targeted retry feedback, text-free diagnostics, and an
honest reason on a not_found. None of it relaxes verification."""

import asyncio
import logging

import pytest
from qa_fakes import FINE_CLAIM, FINES_DRAFT, FakeJev, FakeProvider, claim, draft
from qa_fakes import make_asker as build

from hoa_qa.answer.prompt import MAX_STATEMENT_CHARS
from hoa_qa.answer.provider import DraftIssue
from hoa_qa.ask import QASettings
from hoa_qa.models import Corpus, Outcome, OutcomeReason

# Private details that must never reach a log record.
QUESTION = "How can Zebulon at 12 Quail Run inspect the HOA's financial records?"
UNSUPPORTED = claim(
    "Owners may inspect the books on the Zebulon schedule.",
    ("rules-2023-fines", "3rd violation: $125."),
)


def unsupported_zebulon(statement: str, _ids: object) -> float:
    return 0.05 if "Zebulon" in statement else 0.9


def run(corpus: Corpus, settings: QASettings, jev: FakeJev, provider: FakeProvider):
    return asyncio.run(build(corpus, settings, jev, provider)(QUESTION))


def attempt_logs(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        r.__dict__["hoa_qa_attempt"]
        for r in caplog.records
        if hasattr(r, "hoa_qa_attempt")
    ]


def test_daf93582_shape_sends_feedback_and_logs_diagnostics(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Attempt 1: one essential claim fails verification. Attempt 2: an
    essential claim is over the statement cap, so the draft is invalid."""
    caplog.set_level(logging.DEBUG)
    jev.support = unsupported_zebulon
    provider = FakeProvider(
        [draft(FINE_CLAIM, UNSUPPORTED), None],
        issues=[(), (DraftIssue(1, "too_long"),)],
    )
    result = run(corpus, settings, jev, provider)
    answer = result.answer

    # Fail-closed is unchanged: still not_found, no unverified text shown.
    assert answer.outcome is Outcome.not_found
    assert answer.citations == () and answer.confidence is None
    assert UNSUPPORTED.statement not in answer.answer_text
    assert FINE_CLAIM.statement not in answer.answer_text
    assert answer.reason is OutcomeReason.unverified
    assert "found related passages" in answer.answer_text
    assert "Try rephrasing" in answer.answer_text
    # Whole documents only (no page fragment, no quote), cited one first.
    assert [(d.title, d.url) for d in answer.related_documents] == [
        (
            "Rules & Regulations 2023",
            "https://example.org/hoa/rules-2023.pdf?ver=fixture-1",
        ),
        (
            "Rules & Regulations 2016",
            "https://example.org/hoa/rules-2016.pdf?ver=fixture-1",
        ),
    ]
    assert "Rules & Regulations 2023; Rules & Regulations 2016" in answer.answer_text

    # The retry named the failing claim by number with a code-written reason
    # and restated the length rule.
    retry = provider.prompts[1].user
    why = "the cited passages do not state all of it"
    assert f"- claim 2: {UNSUPPORTED.statement} ({why})" in retry
    assert "claim 1:" not in retry  # the verified claim is not listed
    assert f"at most {MAX_STATEMENT_CHARS} characters: split, don't merge" in retry

    # One diagnostics record per attempt, text-free.
    first, second = attempt_logs(caplog)
    assert (first["attempt"], first["terminal"]) == (1, "claims_failed")
    assert first["fallback_eligible"] is False  # an essential claim failed
    assert [
        (c["index"], c["kind"], c["essential"], c["reason"]) for c in first["claims"]
    ] == [(1, "answer", True, "ok"), (2, "answer", True, "unsupported")]
    assert first["claims"][1]["support"] == 0.05
    assert first["claims"][1]["chunk_ids"] == ("rules-2023-fines",)
    assert first["claims"][1]["statement_chars"] == len(UNSUPPORTED.statement)
    assert (second["attempt"], second["terminal"]) == (2, "invalid_draft")
    assert second["issues"] == [{"index": 2, "code": "too_long"}]
    assert all(r["request_id"] == answer.request_id for r in (first, second))
    [sweep] = [
        r.__dict__["hoa_qa_sweep"] for r in caplog.records if hasattr(r, "hoa_qa_sweep")
    ]
    assert [s["chunk_id"] for s in sweep["selected"]] == [
        "rules-2023-fines",
        "rules-2016-fines",
    ]
    [request] = [r.__dict__["hoa_qa"] for r in caplog.records if hasattr(r, "hoa_qa")]
    assert request["reason"] == "unverified"
    assert request["notes"] == "claims_failed=1,invalid_draft"


def test_invalid_first_draft_gets_per_claim_retry_feedback(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider(
        [None, FINES_DRAFT],
        issues=[(DraftIssue(None, "truncated"), DraftIssue(2, "too_long"))],
    )
    result = run(corpus, settings, jev, provider)
    assert result.answer.outcome is Outcome.answered
    assert result.answer.reason is None and result.answer.related_documents == ()
    retry = provider.prompts[1].user
    assert "The previous output was not valid." in retry
    block = retry[retry.index("<rejected_claims>") :]
    assert "- the output: the output was cut off" in block
    assert "- claim 3: its statement was over the 400-character limit" in block


def test_diagnostic_logs_carry_no_question_answer_or_quote_text(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    jev.support = unsupported_zebulon
    # A model-written chunk_id is free text too: an unknown one is masked.
    smuggled = claim(
        "Quail Run owners get a waiver.",
        ("rules-2023-fines", "3rd violation: $125."),
        ("Zebulon's secret", "3rd violation: $125."),
        essential=False,
    )
    provider = FakeProvider([draft(FINE_CLAIM, UNSUPPORTED, smuggled)] * 2)
    result = run(corpus, settings, jev, provider)
    records = [
        r
        for r in caplog.records
        if hasattr(r, "hoa_qa_attempt") or hasattr(r, "hoa_qa_sweep")
    ]
    assert len(records) == 3  # sweep + two attempts
    rendered = " ".join(r.getMessage() + str(r.__dict__) for r in records)
    rendered += caplog.text
    secrets = (
        "Zebulon",
        "Quail Run",
        "financial records",
        QUESTION,
        FINE_CLAIM.statement,
        "3rd violation",
        result.answer.answer_text,
    )
    for secret in secrets:
        assert secret not in rendered
    claims = attempt_logs(caplog)[0]["claims"]
    assert claims[2]["chunk_ids"] == ("rules-2023-fines", "?")


def test_empty_retrieval_says_nothing_relevant_was_found(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.relevance = {}
    provider = FakeProvider([FINES_DRAFT])
    answer = run(corpus, settings, jev, provider).answer
    assert answer.outcome is Outcome.not_found
    assert answer.reason is OutcomeReason.no_relevant_passages
    assert answer.answer_text.startswith("I couldn't find this in the HOA documents")
    assert answer.related_documents == () and provider.prompts == []


def test_model_with_nothing_to_say_is_not_called_unverified(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    answer = run(corpus, settings, jev, FakeProvider([draft()])).answer
    assert answer.outcome is Outcome.not_found
    assert answer.reason is OutcomeReason.no_answer_in_passages
    assert answer.related_documents == ()


def test_drop_rule_and_fallback_are_unchanged(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """An essential failure twice is still not_found; nothing is salvaged."""
    jev.support = unsupported_zebulon
    provider = FakeProvider([draft(UNSUPPORTED)] * 2)
    answer = run(corpus, settings, jev, provider).answer
    assert answer.outcome is Outcome.not_found
    assert answer.reason is OutcomeReason.unverified
    assert "Zebulon" not in answer.answer_text and answer.citations == ()

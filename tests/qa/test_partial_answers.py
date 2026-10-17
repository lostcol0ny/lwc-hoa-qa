"""A final draft that cannot stand may stand as a partial answer.

The answer model over-applies ``essential``: in the evals of 2026-10-15,
every draft that could not stand had another essential claim verified, and
two questions ended "couldn't verify" although the second draft's verified
claims answered them (docs/specs/2026-10-16-partial-answer-on-final-attempt.md).
On the final attempt only, such a draft stands with its failed claims
dropped and a note that it may be incomplete.
"""

import asyncio
import logging
from typing import Any

import pytest
from qa_fakes import FINE_CLAIM, FakeJev, FakeProvider, claim, draft, make_asker

from hoa_qa.ask import NO_TRACE, OMITTED_NOTE, PARTIAL_NOTE, QASettings
from hoa_qa.cache import CuratedQuestions
from hoa_qa.cache_build import build_entries
from hoa_qa.models import Corpus, Outcome

QUESTION = "How much is the fine for a third violation?"

# Essential, and never supported: it can't be verified on any attempt.
INVENTED = claim(
    "Fines are exempt for first-time owners.",
    ("rules-2023-fines", "2nd violation: $75."),
)
BLOCKED = draft(FINE_CLAIM, INVENTED)


def exempt_unsupported(statement: str, _ids: object) -> float:
    return 0.05 if "exempt" in statement else 0.9


@pytest.fixture
def jev(jev: FakeJev) -> FakeJev:
    jev.support = exempt_unsupported
    return jev


def ask(asker, question: str = QUESTION):
    return asyncio.run(asker(question))


def test_the_final_draft_stands_on_its_verified_essential_claims(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([BLOCKED, BLOCKED])
    result = ask(make_asker(corpus, settings, jev, provider))
    answer = result.answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == f"{FINE_CLAIM.statement} {PARTIAL_NOTE}"
    assert OMITTED_NOTE not in answer.answer_text
    assert "exempt" not in answer.answer_text
    assert {c.chunk_id for c in answer.citations} == {"rules-2023-fines"}
    assert result.tags == ("answered", "retried", "partial")


def test_the_first_draft_is_still_regenerated(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([BLOCKED, draft(FINE_CLAIM)])
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert len(provider.prompts) == 2
    assert INVENTED.statement in provider.prompts[1].user
    assert answer.answer_text == FINE_CLAIM.statement  # the clean retry, no note


def test_no_verified_essential_claim_is_still_not_found(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    side = claim(
        FINE_CLAIM.statement,
        ("rules-2023-fines", "3rd violation: $125."),
        essential=False,
    )
    only_side = draft(side, INVENTED)
    provider = FakeProvider([only_side, only_side])
    answer = ask(make_asker(corpus, settings, jev, provider)).answer
    assert answer.outcome is Outcome.not_found
    assert FINE_CLAIM.statement not in answer.answer_text


def test_the_attempt_log_says_partial(
    corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="hoa_qa.ask")
    ask(make_asker(corpus, settings, jev, FakeProvider([BLOCKED, BLOCKED])))
    attempts = [
        r.__dict__["hoa_qa_attempt"]
        for r in caplog.records
        if hasattr(r, "hoa_qa_attempt")
    ]
    assert [(a["attempt"], a["terminal"]) for a in attempts] == [
        (1, "claims_failed"),
        (2, "partial"),
    ]


def test_a_streamed_partial_answer_is_not_withdrawn(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([BLOCKED, BLOCKED])
    asker = make_asker(corpus, settings, jev, provider)
    events: list[dict[str, Any]] = []
    result = asyncio.run(asker.ask_traced(QUESTION, NO_TRACE, emit=events.append))
    kinds = [e["type"] for e in events if e["type"] != "stage"]
    # Draft 1 is withdrawn before the retry; draft 2's verified claim stays.
    assert kinds == ["claim", "reset", "claim"]
    assert result.answer.outcome is Outcome.answered


def test_cache_build_does_not_store_a_partial_answer(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    curated = CuratedQuestions.model_validate(
        {"questions": [{"id": "fine-third", "question": QUESTION}]}
    )
    asker = make_asker(corpus, settings, jev, FakeProvider([BLOCKED, BLOCKED]))
    entries, [row] = asyncio.run(build_entries(curated, asker))
    assert entries == ()
    assert row.outcome is Outcome.answered and not row.cached
    assert row.claims == ()

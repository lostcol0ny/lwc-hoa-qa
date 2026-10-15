"""Claims about what the documents do not say are rejected by code.

The prompt forbids them, but the support check passed one in the eval of
2026-10-15 ("... without a requirement stated in those documents for a
written collection policy"): a conclusion, close to a legal one, drawn from
what a passage leaves out.
"""

import asyncio

import pytest
from qa_fakes import FINE_CLAIM, FakeJev, FakeProvider, claim, draft, make_asker

from hoa_qa.answer.prompt import REJECTION_REASONS
from hoa_qa.ask import QASettings, _silence_screen
from hoa_qa.models import Corpus, Outcome

EVAL_CLAIM = (
    "Under the current Bylaws and Declaration, the Association may bring legal "
    "action against an Owner to collect unpaid assessments without a requirement "
    "stated in those documents for a written collection policy."
)


@pytest.mark.parametrize(
    ("statement", "rejected"),
    [
        (EVAL_CLAIM, True),
        ("The documents do not mention electric vehicle chargers.", True),
        ("The Declaration does not specify a deadline for repairs.", True),
        ("A fee is not set out in the governing documents.", True),
        ("Guests must be accompanied by an Owner.", False),
        ("A shed not approved by the Board must be removed.", False),
        ("Owners may not install fences taller than four feet.", False),
        ("Trash cans may not be stored in front of the garage.", False),
    ],
)
def test_silence_screen(statement: str, rejected: bool) -> None:
    result = _silence_screen(claim(statement, ("x", "y")))
    assert (result == "absence_claim") is rejected


def test_an_absence_claim_is_dropped_from_the_answer(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    absence = claim(
        EVAL_CLAIM, ("rules-2023-fines", "3rd violation: $125."), essential=False
    )
    provider = FakeProvider([draft(FINE_CLAIM, absence)])
    answer = asyncio.run(
        make_asker(corpus, settings, jev, provider)("What is the third fine?")
    ).answer
    # A non-essential failed claim is dropped; the rest of the answer stands.
    assert answer.outcome is Outcome.answered
    assert FINE_CLAIM.statement in answer.answer_text
    assert "without a requirement" not in answer.answer_text
    # An essential one would fail the draft; the retry then names the reason.
    assert REJECTION_REASONS["absence_claim"].startswith("it says what")

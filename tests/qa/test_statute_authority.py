"""Illinois statutes rank above the governing documents (statutes addendum §5).

A statute passage is authoritative: an informal post can't answer next to it.
When a statute and an HOA document differ, both are kept and the difference
is a conflict note; code never picks a winner.
"""

import asyncio

import pytest
from qa_fakes import (
    STATUTE_MEETINGS,
    FakeJev,
    FakeProvider,
    claim,
    draft,
    make_asker,
    with_statutes,
)

from hoa_qa.answer.prompt import AUTHORITY_FEEDBACK, SYSTEM_PROMPT
from hoa_qa.ask import AUTHORITATIVE, QASettings
from hoa_qa.models import Authority, Corpus, Outcome


@pytest.fixture
def statute_corpus(corpus: Corpus) -> Corpus:
    return with_statutes(corpus)


@pytest.fixture
def jev(statute_corpus: Corpus) -> FakeJev:
    return FakeJev(text_to_id={c.text_clean: c.id for c in statute_corpus.chunks})


def ask(asker, question: str):
    return asyncio.run(asker(question))


def test_statute_is_authoritative() -> None:
    assert Authority.statute in AUTHORITATIVE


def test_prompt_ranks_statute_first_and_never_picks_a_winner() -> None:
    text = " ".join(SYSTEM_PROMPT.split())
    assert "Source authority (highest first): statute > governing >" in text
    assert "Never say which one controls" in text
    assert "never say or imply that the Association is breaking the law" in text
    assert "statute, governing, rules, or board_decision" in AUTHORITY_FEEDBACK


def test_informal_only_answer_is_rejected_next_to_a_statute(
    statute_corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.relevance = {"blog-2022-violations": 0.95, "cicaa-1-30": 0.9}
    blog = claim(
        "An informal 2022 blog post lists a $50 fine for a 2nd offense.",
        ("blog-2022-violations", "2nd offense - $50.00 fine"),
    )
    provider = FakeProvider([draft(blog)] * 2)
    answer = ask(
        make_asker(statute_corpus, settings, jev, provider), "How often must I pay?"
    ).answer
    assert answer.outcome is Outcome.not_found
    assert AUTHORITY_FEEDBACK in provider.prompts[1].user


def test_statute_and_bylaws_difference_keeps_both_and_notes_the_conflict(
    statute_corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    jev.relevance = {"cicaa-1-30": 0.95, "bylaws-5.1": 0.9}
    statute = claim(
        "765 ILCS 160/1-30 states that the board shall meet at least 4 times annually.",
        ("cicaa-1-30", STATUTE_MEETINGS),
    )
    bylaws = claim(
        "The Bylaws say regular Board meetings are held at least twice each year.",
        ("bylaws-5.1", "at least twice each year"),
    )
    conflict = claim(
        "The statute and the Bylaws give different minimum numbers of meetings.",
        ("cicaa-1-30", "at least 4 times annually"),
        ("bylaws-5.1", "at least twice each year"),
        kind="conflict",
        essential=False,
    )
    provider = FakeProvider([draft(statute, bylaws, conflict)])
    answer = ask(
        make_asker(statute_corpus, settings, jev, provider),
        "How often must the Board meet?",
    ).answer
    assert answer.outcome is Outcome.answered
    assert statute.statement in answer.answer_text
    assert bylaws.statement in answer.answer_text
    assert answer.conflicts_noted == (conflict.statement,)
    ids = [c.chunk_id for c in answer.citations]
    assert ids[:2] == ["cicaa-1-30", "bylaws-5.1"]

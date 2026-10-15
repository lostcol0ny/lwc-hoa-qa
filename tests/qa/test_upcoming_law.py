"""A law that has passed but is not yet in effect (ingest.acts).

It may be stated only with the year it takes effect, never as current law,
and it becomes ordinary current law on its effective date with no rebuild.
"""

import asyncio
from datetime import date

import pytest
from qa_fakes import FakeJev, FakeProvider, claim, draft, make_asker, with_statutes

from hoa_qa.answer.prompt import REJECTION_REASONS, SYSTEM_PROMPT, build_prompt
from hoa_qa.ask import QASettings
from hoa_qa.models import Chunk, Corpus, Outcome, not_yet_effective

POLICY = "(j) The board must adopt a written policy for collecting unpaid assessments."
TAKES_EFFECT = date(2099, 1, 1)


@pytest.fixture
def upcoming() -> Chunk:
    body = (
        "Common Interest Community Association Act (765 ILCS 160): 765 ILCS "
        "160/1-45(j) (Finances), as amended by P.A. 199-1, effective 2099-01-01. "
        f"New or changed text: (j).\n{POLICY}"
    )
    return Chunk(
        id="pa-199-0001-1-45-j",
        doc_id="pa-199-0001",
        doc_title="Public Act 199-0001",
        source_url="https://www.ilga.gov/Legislation/PublicActs/View/199-0001",
        page_start=None,
        page_end=None,
        citation_label="765 ILCS 160/1-45(j) (Finances), as amended by P.A. 199-1",
        heading_path=("Common Interest Community Association Act (765 ILCS 160)",),
        text_clean=body,
        text_raw=body,
        authority="statute",  # pyright: ignore[reportArgumentType]
        effective_date=TAKES_EFFECT,
        published_date=None,
        superseded_by=None,
        token_estimate=len(body) // 4,
    )


@pytest.fixture
def law_corpus(corpus: Corpus, upcoming: Chunk) -> Corpus:
    statutes = with_statutes(corpus)
    return statutes.model_copy(update={"chunks": (*statutes.chunks, upcoming)})


@pytest.fixture
def jev(law_corpus: Corpus, upcoming: Chunk) -> FakeJev:
    return FakeJev(
        text_to_id={c.text_clean: c.id for c in law_corpus.chunks},
        relevance={upcoming.id: 0.95},
    )


def ask(asker, question: str):
    return asyncio.run(asker(question))


def test_upcoming_until_its_effective_date(upcoming: Chunk) -> None:
    assert not_yet_effective(upcoming, date(2098, 12, 31))
    assert not not_yet_effective(upcoming, TAKES_EFFECT)


def test_prompt_marks_a_law_not_yet_in_effect(upcoming: Chunk) -> None:
    before = build_prompt("q", [upcoming], today=date(2098, 12, 31)).user
    note = 'note="not yet in effect; takes effect 2099-01-01; Illinois statute'
    assert note in before
    after = build_prompt("q", [upcoming], today=TAKES_EFFECT).user
    assert "not yet in effect" not in after
    text = " ".join(SYSTEM_PROMPT.split())
    assert 'note says "not yet in effect"' in text
    assert "It is a scheduled change, not a conflict" in text


@pytest.mark.parametrize(
    ("statement", "kept"),
    [
        (
            "Starting January 1, 2099, 765 ILCS 160/1-45(j) will require the board "
            "to adopt a written policy for collecting unpaid assessments.",
            True,
        ),
        # No year: in 2026 this reads as the law today.
        (
            "765 ILCS 160/1-45(j) states that the board must adopt a written policy "
            "for collecting unpaid assessments.",
            False,
        ),
        # The year, but called current.
        (
            "765 ILCS 160/1-45(j), as amended effective 2099, currently requires a "
            "written collection policy.",
            False,
        ),
    ],
)
def test_a_claim_citing_it_must_say_when_it_takes_effect(
    law_corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    upcoming: Chunk,
    statement: str,
    kept: bool,
) -> None:
    provider = FakeProvider([draft(claim(statement, (upcoming.id, POLICY)))])
    answer = ask(
        make_asker(law_corpus, settings, jev, provider),
        "Does the HOA need a collection policy?",
    ).answer
    assert (answer.outcome is Outcome.answered) is kept
    if not kept:
        retry = provider.prompts[1].user
        assert REJECTION_REASONS["not_yet_in_effect"] in retry


def test_once_in_effect_it_is_ordinary_law(
    law_corpus: Corpus,
    settings: QASettings,
    jev: FakeJev,
    upcoming: Chunk,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hoa_qa.ask.hoa_today", lambda: TAKES_EFFECT)
    statement = "765 ILCS 160/1-45(j) states that the board must adopt a policy."
    provider = FakeProvider([draft(claim(statement, (upcoming.id, POLICY)))])
    answer = ask(
        make_asker(law_corpus, settings, jev, provider),
        "Does the HOA need a collection policy?",
    ).answer
    assert answer.outcome is Outcome.answered

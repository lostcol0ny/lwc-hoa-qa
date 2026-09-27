"""Fixed statute text added by code, never by the model (addendum §6)."""

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

from hoa_qa.answer.prompt import REJECTION_REASONS, SYSTEM_PROMPT, build_prompt
from hoa_qa.answer.statute_notes import (
    DUES,
    HOMES,
    STATUTE_DISCLAIMER,
    applicability_note,
)
from hoa_qa.ask import QASettings, gives_advice
from hoa_qa.models import Corpus, Outcome

APPLICABILITY = (
    "CICAA exempts associations with 10 or fewer units or annual budgeted "
    "assessments of $100,000 or less (765 ILCS 160/1-75). Lakewood Creek's "
    "website lists 735 homes and annual dues of $452, which suggests assessments "
    "of roughly $332,000, above that threshold. Confirm with the Board or an "
    "attorney."
)
STATUTE_CLAIM = claim(
    "765 ILCS 160/1-30 states that the board shall meet at least 4 times annually.",
    ("cicaa-1-30", STATUTE_MEETINGS),
)
RULES_CLAIM = claim(
    "Under the 2023 Rules, a second violation is a $75 fine.",
    ("rules-2023-fines", "2nd violation: $75."),
)


@pytest.fixture
def statute_corpus(corpus: Corpus) -> Corpus:
    return with_statutes(corpus)


@pytest.fixture
def jev(statute_corpus: Corpus) -> FakeJev:
    return FakeJev(
        relevance={"cicaa-1-30": 0.9, "rules-2023-fines": 0.8},
        text_to_id={c.text_clean: c.id for c in statute_corpus.chunks},
    )


def ask(asker, question: str = "How often must the Board meet?"):
    return asyncio.run(asker(question)).answer


def test_the_note_is_computed_from_cited_corpus_figures(
    statute_corpus: Corpus,
) -> None:
    note = applicability_note(statute_corpus.chunks)
    assert note.text == APPLICABILITY
    assert [(c.chunk_id, c.quote) for c in note.citations] == [
        ("cicaa-1-75", "annual budgeted assessments of $100,000 or less"),
        ("amendments-committee-intro", "492 of 735 homes"),
        ("home-1", "$452 year"),
    ]


@pytest.mark.parametrize("evidence", [HOMES, DUES])
def test_missing_or_changed_evidence_fails(statute_corpus: Corpus, evidence) -> None:
    missing = [c for c in statute_corpus.chunks if c.id != evidence.chunk_id]
    with pytest.raises(ValueError, match="is missing"):
        applicability_note(missing)
    # The dues or the home count changed on the site: 452 -> 460, 735 -> 760.
    edit = {"$452": "$460", "735": "760"}
    changed = []
    for chunk in statute_corpus.chunks:
        if chunk.id == evidence.chunk_id:
            text = chunk.text_clean
            for old, new in edit.items():
                text = text.replace(old, new)
            chunk = chunk.model_copy(update={"text_clean": text})
        changed.append(chunk)
    with pytest.raises(ValueError, match="no longer says"):
        applicability_note(changed)


def test_asker_refuses_a_corpus_with_stale_evidence(
    statute_corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    chunks = tuple(c for c in statute_corpus.chunks if c.id != "home-1")
    stale = statute_corpus.model_copy(
        update={
            "chunks": chunks,
            "manifest": statute_corpus.manifest.model_copy(
                update={"chunk_count": len(chunks)}
            ),
        }
    )
    with pytest.raises(ValueError, match="home-1 is missing"):
        make_asker(stale, settings, jev, FakeProvider([]))


def test_statute_answers_get_the_disclaimer_and_cicaa_note(
    statute_corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft(STATUTE_CLAIM)])
    answer = ask(make_asker(statute_corpus, settings, jev, provider))
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == " ".join(
        (STATUTE_CLAIM.statement, STATUTE_DISCLAIMER, APPLICABILITY)
    )
    assert [c.chunk_id for c in answer.citations] == [
        "cicaa-1-30",
        "cicaa-1-75",
        "amendments-committee-intro",
        "home-1",
    ]
    # The model is never asked to write either text.
    prompt = provider.prompts[0]
    assert STATUTE_DISCLAIMER not in prompt.system + prompt.user
    assert "roughly" not in prompt.system + prompt.user


def test_hoa_only_answers_get_neither(
    statute_corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    provider = FakeProvider([draft(RULES_CLAIM)])
    answer = ask(make_asker(statute_corpus, settings, jev, provider))
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == RULES_CLAIM.statement
    assert STATUTE_DISCLAIMER not in answer.answer_text


def test_dropped_statute_claims_add_nothing(
    statute_corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    """Only kept claims count: a statute claim that failed adds no note."""
    bad = claim(
        "765 ILCS 160/1-30 states that the board shall meet monthly.",
        ("cicaa-1-30", "meet monthly"),
        essential=False,
    )
    provider = FakeProvider([draft(RULES_CLAIM, bad)] * 2)
    answer = ask(make_asker(statute_corpus, settings, jev, provider))
    assert answer.outcome is Outcome.answered
    assert STATUTE_DISCLAIMER not in answer.answer_text


@pytest.mark.parametrize(
    "statement",
    [
        "You have the right to inspect the association's records.",
        "Under 765 ILCS 160/1-30 you are entitled to see the books.",
        "In your case, the board must meet 4 times a year.",
        "The Association is violating 765 ILCS 160/1-30.",
        "Your rights under the Act include inspecting records.",
    ],
)
def test_advice_phrasing_on_statute_claims_is_rejected(
    statute_corpus: Corpus, settings: QASettings, jev: FakeJev, statement: str
) -> None:
    advice = claim(statement, ("cicaa-1-30", STATUTE_MEETINGS))
    provider = FakeProvider([draft(advice), draft(STATUTE_CLAIM)])
    answer = ask(make_asker(statute_corpus, settings, jev, provider))
    assert statement not in answer.answer_text
    assert answer.answer_text.startswith(STATUTE_CLAIM.statement)
    assert REJECTION_REASONS["advice_phrasing"] in provider.prompts[1].user
    # Rejected by code before a support check was spent on it.
    assert len(jev.calls_of("c")) == 1


def test_advice_screen_is_narrow() -> None:
    assert not gives_advice(STATUTE_CLAIM.statement)
    assert not gives_advice(
        "765 ILCS 160/1-30 states that the board may levy fines for violations."
    )
    assert gives_advice("You have a right to vote by proxy.")


def test_prompt_rules_for_statutes(statute_corpus: Corpus) -> None:
    text = " ".join(SYSTEM_PROMPT.split())
    assert '"765 ILCS 160/... states that ..."' in text
    assert "Never tell the reader what their rights are" in text
    assert 'include at least one "answer" claim stating what that statute' in text
    assert "never state or imply either" in text
    statute = next(c for c in statute_corpus.chunks if c.id == "cicaa-1-30")
    rendered = build_prompt("q", [statute]).user
    assert 'authority="statute"' in rendered
    assert 'note="Illinois statute; state what it says, not how it applies"' in rendered

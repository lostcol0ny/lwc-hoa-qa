"""Illinois statutes rank above the governing documents (statutes addendum §5).

A statute passage is authoritative: an informal post can't answer next to it.
When a statute and an HOA document differ, both are kept and the difference
is a conflict note; code never picks a winner.
"""

import asyncio

import pytest
from qa_fakes import FakeJev, FakeProvider, claim, draft, make_asker

from hoa_qa.answer.prompt import AUTHORITY_FEEDBACK, SYSTEM_PROMPT
from hoa_qa.ask import AUTHORITATIVE, QASettings
from hoa_qa.models import Authority, Chunk, Corpus, Outcome

MEETINGS = "(a) The board shall meet at least 4 times annually."
BYLAWS_MEETINGS = (
    "Regular meetings of the Board of Directors shall be held at least twice each year."
)


def statute_chunk(chunk_id: str, label: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id="cicaa",
        doc_title="Common Interest Community Association Act",
        source_url=(
            "https://ftp.ilga.gov/ILCS/Ch%200765/Act%200160/076501600K1-30.html"
        ),
        page_start=None,
        page_end=None,
        citation_label=label,
        heading_path=("Common Interest Community Association Act (765 ILCS 160)",),
        text_clean=text,
        text_raw=text,
        authority=Authority.statute,
        effective_date="2024-01-01",  # pyright: ignore[reportArgumentType]
        published_date=None,
        superseded_by=None,
        token_estimate=len(text) // 4,
    )


@pytest.fixture
def statute_corpus(corpus: Corpus) -> Corpus:
    """The mini corpus plus a CICAA section and a differing Bylaws clause."""
    bylaws = next(c for c in corpus.chunks if c.doc_id == "bylaws")
    extra = (
        statute_chunk(
            "cicaa-1-30",
            "765 ILCS 160/1-30 (Board duties and obligations; records)",
            f"(765 ILCS 160/1-30)\nSec. 1-30. Board duties and obligations; "
            f"records.\n{MEETINGS}",
        ),
        bylaws.model_copy(
            update={
                "id": "bylaws-5.1",
                "citation_label": "Bylaws, Art. 5, §5.1, p. 4",
                "text_clean": BYLAWS_MEETINGS,
                "text_raw": BYLAWS_MEETINGS,
            }
        ),
    )
    hashes = {**corpus.manifest.source_hashes, "cicaa": "0" * 64}
    manifest = corpus.manifest.model_copy(
        update={"chunk_count": len(corpus.chunks) + len(extra), "source_hashes": hashes}
    )
    return Corpus.model_validate(
        {"manifest": manifest, "chunks": (*corpus.chunks, *extra)}
    )


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
        ("cicaa-1-30", MEETINGS),
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
    assert {c.chunk_id for c in answer.citations} == {"cicaa-1-30", "bylaws-5.1"}

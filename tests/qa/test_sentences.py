"""Numbered sentences: the splitter, and citing by number end to end."""

import asyncio

import pytest
from qa_fakes import FakeJev, FakeProvider, make_asker

from hoa_qa.answer.prompt import build_prompt
from hoa_qa.answer.provider import (
    ANSWER_SCHEMA,
    AnswerDraft,
    DraftCitation,
    DraftClaim,
    parse_draft,
)
from hoa_qa.ask import OMITTED_NOTE, QASettings
from hoa_qa.models import Corpus, Outcome
from hoa_qa.sentences import MAX_SENTENCE_CHARS, number_sentences, split_sentences
from hoa_qa.verify.quotes import check_quotes, resolve_sentences

RULE = (
    "2.9 Holiday Decorations. Exterior items of a decorative nature shall be "
    "displayed no earlier than two (2) weeks prior to the event. Provided, "
    "however, that decorative lights may only be displayed between December 1st "
    "and January 10th. Gutter light hooks may be left attached to gutters."
)


def test_splits_at_sentence_ends_and_keeps_section_numbers_attached() -> None:
    assert split_sentences(RULE) == (
        "2.9 Holiday Decorations. Exterior items of a decorative nature shall be "
        "displayed no earlier than two (2) weeks prior to the event.",
        "Provided, however, that decorative lights may only be displayed between "
        "December 1st and January 10th.",
        "Gutter light hooks may be left attached to gutters.",
    )


@pytest.mark.parametrize(
    "text",
    [
        "See Sec. 3 of the Rules for the fining schedule that applies here today.",
        "Quiet hours begin at 10 p.m. Monday through Friday for every resident.",
        "The fee is set by No. 12 of the Board resolutions adopted last spring.",
        "B. Fines The Board of Directors has set up a fining structure for all.",
        "Payment is due Jan. 1. Late fees apply.",
    ],
)
def test_abbreviations_and_short_pieces_do_not_split(text: str) -> None:
    assert split_sentences(text) == (text,)


def test_a_long_run_is_cut_at_a_clause_break() -> None:
    schedule = "FINING SCHEDULE " + "; ".join(
        f"offense {n} is fined at ${n * 25} per day" for n in range(1, 30)
    )
    pieces = split_sentences(schedule)
    assert len(pieces) > 1
    assert all(0 < len(piece) <= MAX_SENTENCE_CHARS for piece in pieces)
    assert " ".join(pieces) == schedule
    assert all(piece.endswith(";") for piece in pieces[:-1]), "cut after a clause"


def test_every_sentence_of_every_passage_is_a_valid_quote(corpus: Corpus) -> None:
    by_id = {chunk.id: chunk for chunk in corpus.chunks}
    for chunk in corpus.chunks:
        sentences = split_sentences(chunk.text_clean)
        assert sentences and " ".join(sentences) == " ".join(chunk.text_clean.split())
        cited = [
            DraftCitation(chunk_id=chunk.id, sentence=n)
            for n in range(1, len(sentences) + 1)
        ]
        checked = check_quotes(resolve_sentences(cited, by_id), by_id)
        assert [c.quote for c in checked] == list(sentences)


def test_empty_and_whitespace_only_text() -> None:
    assert split_sentences("") == () and split_sentences("  \n ") == ()
    assert number_sentences("One short rule.") == "[1] One short rule."


def test_resolving_a_bad_number_or_passage_leaves_no_quote(corpus: Corpus) -> None:
    by_id = {chunk.id: chunk for chunk in corpus.chunks}
    cited = [
        DraftCitation(chunk_id="rules-2023-fines", sentence=99),
        DraftCitation(chunk_id="no-such-chunk", sentence=1),
        DraftCitation(chunk_id="rules-2023-fines", quote="3rd violation: $125."),
    ]
    resolved = resolve_sentences(cited, by_id)
    assert [c.quote for c in resolved] == ["", "", "3rd violation: $125."]
    assert [c.quote for c in check_quotes(resolved, by_id)] == ["3rd violation: $125."]


def test_the_model_is_asked_for_sentence_numbers_not_quotes(corpus: Corpus) -> None:
    citation = ANSWER_SCHEMA["properties"]["claims"]["items"]["properties"][
        "citations"
    ]["items"]
    assert set(citation["properties"]) == {"chunk_id", "sentence"}
    assert citation["required"] == ["chunk_id", "sentence"]
    prompt = build_prompt("What are the fines?", corpus.chunks[:2])
    assert "Do not copy the sentence's text" in prompt.system
    assert "copied EXACTLY" not in prompt.system
    assert "\n[1] " in prompt.user
    parsed = parse_draft(
        '{"claims": [{"statement": "A fine.", "kind": "answer", "essential": true,'
        ' "citations": [{"chunk_id": "rules-2023-fines", "sentence": 2}]}],'
        ' "confidence": 0.9, "refer_to_board": false}'
    )
    assert parsed is not None
    assert parsed.claims[0].citations[0].sentence == 2
    assert (
        parse_draft(
            '{"claims": [{"statement": "A fine.", "kind": "answer", "essential": true,'
            ' "citations": [{"chunk_id": "x", "sentence": 0}]}],'
            ' "confidence": 0.9, "refer_to_board": false}'
        )
        is None
    ), "sentence numbers start at 1"


def cited(statement: str, sentence: int, *, essential: bool = True) -> DraftClaim:
    return DraftClaim(
        statement=statement,
        kind="answer",
        essential=essential,
        citations=[DraftCitation(chunk_id="rules-2023-fines", sentence=sentence)],
    )


def test_a_cited_sentence_becomes_the_quote_residents_see(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    fines = next(c for c in corpus.chunks if c.id == "rules-2023-fines")
    sentences = split_sentences(fines.text_clean)
    number = next(n for n, s in enumerate(sentences, 1) if "$125" in s)
    answer_draft = AnswerDraft(
        claims=[cited("A third violation is a $125 fine.", number)],
        confidence=0.9,
        refer_to_board=False,
    )
    provider = FakeProvider([answer_draft])
    answer = asyncio.run(make_asker(corpus, settings, jev, provider)("fines?")).answer
    assert answer.outcome is Outcome.answered
    [citation] = answer.citations
    assert citation.quote == sentences[number - 1]
    assert citation.quote in fines.text_clean


def test_a_sentence_number_that_does_not_exist_fails_the_claim(
    corpus: Corpus, settings: QASettings, jev: FakeJev
) -> None:
    answer_draft = AnswerDraft(
        claims=[
            cited("A third violation is a $125 fine.", 1),
            cited("There is a tenth rule.", 99, essential=False),
        ],
        confidence=0.9,
        refer_to_board=False,
    )
    provider = FakeProvider([answer_draft])
    answer = asyncio.run(make_asker(corpus, settings, jev, provider)("fines?")).answer
    assert answer.outcome is Outcome.answered
    assert answer.answer_text == f"A third violation is a $125 fine. {OMITTED_NOTE}"
    assert len(provider.prompts) == 1 and len(answer.citations) == 1

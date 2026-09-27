"""Quote normalization and the code-only citation check."""

import pytest

from hoa_qa.answer.provider import DraftCitation
from hoa_qa.models import Corpus
from hoa_qa.verify.quotes import check_quotes, normalize


def test_normalize_whitespace_case_and_curly_quotes() -> None:
    assert normalize("  The “Board”\n\tisn’t  ") == 'the "board" isn\'t'


def by_id(corpus: Corpus) -> dict:
    return {c.id: c for c in corpus.chunks}


def test_quote_matches_after_normalization(corpus: Corpus) -> None:
    kept = check_quotes(
        [DraftCitation(chunk_id="rules-2023-fines", quote="3RD  violation:\n$125")],
        by_id(corpus),
    )
    assert [k.chunk.id for k in kept] == ["rules-2023-fines"]


def test_rejections(corpus: Corpus) -> None:
    kept = check_quotes(
        [
            DraftCitation(chunk_id="rules-2023-fines", quote="3rd violation: $100"),
            DraftCitation(chunk_id="missing", quote="1st violation"),
            DraftCitation(chunk_id="rules-2023-fines", quote="   "),
            DraftCitation(chunk_id="rules-2023-fines", quote="2nd violation: $75."),
            DraftCitation(chunk_id="rules-2023-fines", quote="first violation: $75."),
        ],
        by_id(corpus),
    )
    assert [k.quote for k in kept] == ["2nd violation: $75."]


def test_normalize_folds_ellipses_and_dashes() -> None:
    assert normalize("HOA DUES… This — that – other") == normalize(
        "HOA DUES... This - that - other"
    )


@pytest.mark.parametrize("punctuation", list(";,:()[]{}/"))
def test_normalize_punctuation_spacing(punctuation: str) -> None:
    assert normalize(f"one \t{punctuation}\n two") == normalize(f"one{punctuation}two")


@pytest.mark.parametrize(
    ("quote", "source", "matches"),
    [
        ("color changes;3) fencing", "color changes; 3) fencing", True),
        ("shall not", "shallnot", False),
        ("$75", "$7 5", False),
        ("color changes; fencing", "color changes; 3) other work; fencing", False),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_quote_spacing_preserves_contiguous_text(
    corpus: Corpus, quote: str, source: str, matches: bool, reverse: bool
) -> None:
    if reverse:
        quote, source = source, quote
    chunk = corpus.chunks[0].model_copy(update={"text_clean": source})
    kept = check_quotes(
        [DraftCitation(chunk_id=chunk.id, quote=quote)], {chunk.id: chunk}
    )
    assert bool(kept) is matches

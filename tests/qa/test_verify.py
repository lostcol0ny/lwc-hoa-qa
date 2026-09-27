"""Quote normalization and the code-only citation check."""

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

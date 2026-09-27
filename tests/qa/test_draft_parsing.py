"""Parsing the answer model's JSON: strict first, then a safe salvage."""

import json

from hoa_qa.answer.prompt import (
    MAX_CITATIONS_PER_CLAIM,
    MAX_CLAIMS,
    MAX_STATEMENT_CHARS,
)
from hoa_qa.answer.provider import parse_draft, parse_draft_checked, parse_draft_noted


def claim(statement: str = "Fact.", *, essential: bool = True, cites: int = 1) -> dict:
    return {
        "statement": statement,
        "kind": "answer",
        "essential": essential,
        "citations": [{"chunk_id": f"c{i}", "quote": "q"} for i in range(cites)],
    }


def output(*claims: dict) -> str:
    return json.dumps(
        {"claims": list(claims), "confidence": 0.7, "refer_to_board": False}
    )


def test_valid_output_has_no_note() -> None:
    draft, note = parse_draft_noted(output(claim()))
    assert draft is not None and note is None


def test_extra_citations_are_trimmed() -> None:
    draft, note = parse_draft_noted(output(claim(cites=MAX_CITATIONS_PER_CLAIM + 2)))
    assert draft is not None
    assert len(draft.claims[0].citations) == MAX_CITATIONS_PER_CLAIM
    assert note is not None and note.startswith("salvaged: ")
    assert "citations:too_long" in note


def test_non_essential_claims_past_the_cap_or_too_long_are_dropped() -> None:
    long = claim("x" * (MAX_STATEMENT_CHARS + 1), essential=False)
    extra = [claim(f"Fact {i}.", essential=False) for i in range(MAX_CLAIMS + 1)]
    draft, note = parse_draft_noted(output(claim("Main."), long, *extra))
    assert draft is not None and note is not None
    assert len(draft.claims) == MAX_CLAIMS
    assert draft.claims[0].statement == "Main."
    assert all(len(c.statement) <= MAX_STATEMENT_CHARS for c in draft.claims)


def test_losing_an_essential_claim_invalidates_the_draft() -> None:
    long = claim("x" * (MAX_STATEMENT_CHARS + 1), essential=True)
    draft, note = parse_draft_noted(output(claim(), long))
    assert draft is None
    assert note is not None and note.startswith("invalid: ")
    # Notes carry field paths and error types, never the model's text.
    assert "xxxx" not in note
    past_cap = [claim(f"Fact {i}.") for i in range(MAX_CLAIMS + 1)]
    assert parse_draft(output(*past_cap)) is None


def test_unparseable_output_is_invalid() -> None:
    draft, note = parse_draft_noted("{not json")
    assert draft is None and note is not None and note.startswith("invalid: ")


def test_salvage_reports_dropped_claims_but_not_trimmed_citations() -> None:
    cut = parse_draft_checked(output(claim(cites=MAX_CITATIONS_PER_CLAIM + 1)))
    assert cut.draft is not None and not cut.claims_trimmed
    long = claim("x" * (MAX_STATEMENT_CHARS + 1), essential=False)
    dropped = parse_draft_checked(output(claim(), long))
    assert dropped.draft is not None and dropped.claims_trimmed
    assert not parse_draft_checked(output(claim())).claims_trimmed

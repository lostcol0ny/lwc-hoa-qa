"""Tests for the contract consumed by ingest, QA, and web units."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from hoa_qa.models import (
    Answer,
    Authority,
    Chunk,
    CorpusManifest,
    Outcome,
    authority_rank,
    citation_url,
    load_corpus,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini_corpus.json"


@pytest.fixture
def chunk_data() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())["chunks"][0]


def test_valid_chunk(chunk_data: dict[str, Any]) -> None:
    chunk = Chunk.model_validate(chunk_data)
    assert chunk.id == "bylaws-3.4"
    assert chunk.authority is Authority.governing
    assert chunk.effective_date == date(2001, 8, 9)
    assert chunk.published_date is None
    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk
    with pytest.raises(ValidationError, match="frozen"):
        chunk.text_clean = "replacement"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("page_end", 1),
        ("page_start", 0),
        ("page_end", 0),
        ("page_start", "2"),
        ("page_end", True),
        ("text_clean", ""),
        ("text_clean", " \n\t"),
        ("source_url", "http://example.org/bylaws.pdf"),
        ("source_url", "not-a-url"),
        ("source_url", "https://"),
        ("source_url", " https://example.org/bylaws.pdf"),
        ("authority", "unknown"),
        ("effective_date", "2026-02-30"),
        ("published_date", "yesterday"),
        ("token_estimate", -1),
        ("token_estimate", "10"),
        ("token_estimate", True),
        ("unexpected", "value"),
    ],
)
def test_chunk_validation_failures(
    chunk_data: dict[str, Any], field: str, value: Any
) -> None:
    chunk_data[field] = value
    with pytest.raises(ValidationError):
        Chunk.model_validate(chunk_data)


def test_missing_field(chunk_data: dict[str, Any]) -> None:
    del chunk_data["text_raw"]
    with pytest.raises(ValidationError, match="text_raw"):
        Chunk.model_validate(chunk_data)


@pytest.mark.parametrize(("start", "end"), [(None, None), (2, None), (None, 2)])
def test_optional_pages(
    chunk_data: dict[str, Any], start: int | None, end: int | None
) -> None:
    Chunk.model_validate({**chunk_data, "page_start": start, "page_end": end})


def test_authority_order() -> None:
    assert (
        authority_rank(Authority.governing)
        > authority_rank(Authority.rules)
        > authority_rank(Authority.board_decision)
        > authority_rank(Authority.website)
        > authority_rank(Authority.form)
        == authority_rank(Authority.informal)
        > authority_rank(Authority.superseded)
    )


@pytest.mark.parametrize(
    ("url", "page", "expected"),
    [
        ("https://example.org/a.pdf", 2, "https://example.org/a.pdf#page=2"),
        (
            "https://example.org/a.pdf?ver=a%2Fb&x=1",
            2,
            "https://example.org/a.pdf?ver=a%2Fb&x=1#page=2",
        ),
        (
            "https://example.org/a.PDF?ver=1#old",
            2,
            "https://example.org/a.PDF?ver=1#page=2",
        ),
        (
            "https://example.org/post?file=a.pdf",
            2,
            "https://example.org/post?file=a.pdf",
        ),
        ("https://example.org/a.pdf?ver=1", None, "https://example.org/a.pdf?ver=1"),
    ],
)
def test_citation_url(
    chunk_data: dict[str, Any], url: str, page: int | None, expected: str
) -> None:
    chunk = Chunk.model_validate({**chunk_data, "source_url": url, "page_start": page})
    assert citation_url(chunk) == expected


def test_load_fixture() -> None:
    corpus = load_corpus(FIXTURE)
    assert corpus.manifest.chunk_count == len(corpus.chunks) == 6
    assert corpus.chunks[3].superseded_by == "rules-2023"
    assert corpus.manifest.build_time.tzinfo is not None


def test_duplicate_ids(tmp_path: Path) -> None:
    data = json.loads(FIXTURE.read_text())
    data["chunks"].append(data["chunks"][0])
    data["manifest"]["chunk_count"] += 1
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="duplicate chunk ids"):
        load_corpus(path)


def test_load_validates_chunks(tmp_path: Path) -> None:
    data = json.loads(FIXTURE.read_text())
    data["chunks"][0]["text_clean"] = ""
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="text_clean"):
        load_corpus(path)


@pytest.mark.parametrize("count", [-1, "6", True])
def test_manifest_invalid_count(count: Any) -> None:
    manifest = json.loads(FIXTURE.read_text())["manifest"]
    manifest["chunk_count"] = count
    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(manifest)


@pytest.mark.parametrize("outcome", list(Outcome))
def test_response_outcomes(outcome: Outcome) -> None:
    answer = Answer(
        request_id="test-request",
        outcome=outcome,
        answer_text="Test response",
        citations=[],
        confidence=None,
        conflicts_noted=[],
        disclaimer="Unofficial",
    )
    assert Answer.model_validate_json(answer.model_dump_json()) == answer


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), "0.5", True])
def test_invalid_confidence(confidence: Any) -> None:
    with pytest.raises(ValidationError):
        Answer(
            request_id="test",
            outcome=Outcome.answered,
            answer_text="Test",
            citations=[],
            confidence=confidence,
            conflicts_noted=[],
            disclaimer="",
        )


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_cited_answer(confidence: float) -> None:
    answer = Answer.model_validate(
        dict(
            request_id="test",
            outcome="answered",
            answer_text="A cited answer",
            citations=[
                dict(
                    chunk_id="bylaws-3.4",
                    citation_label="Bylaws §3.4",
                    url="https://example.org/bylaws.pdf#page=2",
                    quote="Board",
                )
            ],
            confidence=confidence,
            conflicts_noted=["Sources differ"],
            disclaimer="Unofficial",
        )
    )
    assert answer.citations[0].chunk_id == "bylaws-3.4"
    assert answer.confidence == confidence

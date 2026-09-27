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
    Citation,
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
    assert corpus.manifest.chunk_count == len(corpus.chunks) == 8
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
        citations=(),
        confidence=None,
        conflicts_noted=(),
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
            citations=(),
            confidence=confidence,
            conflicts_noted=(),
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


@pytest.mark.parametrize("authority", list(Authority))
@pytest.mark.parametrize("replacement", [None, "rules-2023"])
def test_supersession_contract(
    chunk_data: dict[str, Any], authority: Authority, replacement: str | None
) -> None:
    data = {**chunk_data, "authority": authority, "superseded_by": replacement}
    if (authority == Authority.superseded) == (replacement is not None):
        assert Chunk.model_validate(data).superseded_by == replacement
    else:
        with pytest.raises(ValidationError, match="superseded"):
            Chunk.model_validate(data)


@pytest.mark.parametrize("problem", ["replacement", "count", "hash"])
def test_corpus_integrity(tmp_path: Path, problem: str) -> None:
    data = json.loads(FIXTURE.read_text())
    if problem == "replacement":
        # A chunk ID is not a document ID.
        data["chunks"][3]["superseded_by"] = "rules-2023-fines"
        message = "superseded_by"
    elif problem == "count":
        data["manifest"]["chunk_count"] = 7
        message = "chunk_count"
    else:
        del data["manifest"]["source_hashes"]["bylaws"]
        message = "source_hashes"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match=message):
        load_corpus(path)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,test",
        "http://example.org/a.pdf",
        "https://user:pass@example.org/a.pdf",
        "https://user@example.org/",
        "https://@example.org/",
    ],
)
def test_unsafe_urls(chunk_data: dict[str, Any], url: str) -> None:
    with pytest.raises(ValidationError):
        Chunk.model_validate({**chunk_data, "source_url": url})
    with pytest.raises(ValidationError):
        Citation(chunk_id="test", citation_label="Test", url=url, quote="Test")


def test_https_citation_fragment() -> None:
    url = "https://example.org/a.pdf?ver=123#page=2"
    assert (
        Citation(chunk_id="test", citation_label="Test", url=url, quote="").url == url
    )


@pytest.mark.parametrize("field", ["id", "doc_id", "doc_title", "citation_label"])
def test_empty_chunk_identifiers(chunk_data: dict[str, Any], field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Chunk.model_validate({**chunk_data, field: ""})


@pytest.mark.parametrize("field", ["chunk_id", "citation_label"])
def test_empty_citation_identifiers(field: str) -> None:
    data = dict(
        chunk_id="test", citation_label="Test", url="https://example.org/", quote=""
    )
    data[field] = ""
    with pytest.raises(ValidationError, match=field):
        Citation.model_validate(data)


def test_empty_request_id() -> None:
    with pytest.raises(ValidationError, match="request_id"):
        Answer(
            request_id="",
            outcome=Outcome.not_found,
            answer_text="",
            citations=(),
            confidence=None,
            conflicts_noted=(),
            disclaimer="",
        )


@pytest.mark.parametrize("timestamp", ["2026-09-27T00:00:00", "2026-09-27"])
def test_naive_build_time(timestamp: str) -> None:
    data = json.loads(FIXTURE.read_text())["manifest"]
    data["build_time"] = timestamp
    with pytest.raises(ValidationError):
        CorpusManifest.model_validate(data)


@pytest.mark.parametrize("field", ["effective_date", "published_date"])
@pytest.mark.parametrize(
    "value",
    [
        0,
        86400,
        0.0,
        True,
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00Z",
        "2026",
        "20260101",
    ],
)
def test_strict_dates(chunk_data: dict[str, Any], field: str, value: Any) -> None:
    data = {**chunk_data, field: value}
    with pytest.raises(ValidationError):
        Chunk.model_validate(data)
    with pytest.raises(ValidationError):
        Chunk.model_validate_json(json.dumps(data))


def test_python_date_objects(chunk_data: dict[str, Any]) -> None:
    from datetime import datetime

    assert Chunk.model_validate({**chunk_data, "effective_date": date(2001, 1, 1)})
    with pytest.raises(ValidationError):
        Chunk.model_validate({**chunk_data, "effective_date": datetime(2001, 1, 1)})


def test_immutable_sequences() -> None:
    corpus = load_corpus(FIXTURE)
    answer = Answer.model_validate(
        dict(
            request_id="test",
            outcome="not_found",
            answer_text="",
            citations=[],
            confidence=None,
            conflicts_noted=["Test"],
            disclaimer="",
        )
    )
    for sequence in (
        corpus.chunks,
        corpus.chunks[0].heading_path,
        corpus.manifest.ocr_fallbacks,
        answer.citations,
        answer.conflicts_noted,
    ):
        assert isinstance(sequence, tuple)
        with pytest.raises(AttributeError):
            sequence.append("bad")  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(json.loads(answer.model_dump_json())["citations"], list)


def test_fixture_history() -> None:
    corpus = load_corpus(FIXTURE)
    blog, declaration = corpus.chunks[-2:]
    assert blog.authority == Authority.informal
    assert blog.published_date == blog.effective_date == date(2022, 10, 20)
    assert blog.text_clean == (
        "1st offense - Written warning 2nd offense - $50.00 fine "
        "3rd offense - $100.00 4th and subsequent offense - $50.00 per day"
    )
    assert declaration.authority == Authority.governing
    assert "($326.00) per Unit" in declaration.text_clean
    assert declaration.source_url.endswith(".pdf?ver=1749305476488")
    assert declaration.page_start == declaration.page_end == 28
    assert citation_url(declaration) == declaration.source_url + "#page=28"
    assert corpus.chunks[4].text_clean.endswith("Motion approved.")
    assert (
        corpus.chunks[5].text_clean
        == "2026 Assessment Prices: $452 year or $113 per quarter"
    )
    assert all(chunk.text_clean == chunk.text_raw for chunk in corpus.chunks)


def test_empty_query_delimiter(chunk_data: dict[str, Any]) -> None:
    chunk = Chunk.model_validate(
        {**chunk_data, "source_url": "https://example.org/a.pdf?"}
    )
    assert citation_url(chunk) == "https://example.org/a.pdf#page=2"


def test_self_supersession(chunk_data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="same doc_id"):
        Chunk.model_validate(
            {
                **chunk_data,
                "authority": "superseded",
                "superseded_by": chunk_data["doc_id"],
            }
        )

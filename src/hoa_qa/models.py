"""Shared corpus and response contracts. Larger authority ranks take precedence."""

import re
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)

_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


def validate_https(value: str) -> str:
    """Validate public HTTPS links without rewriting their query or fragment."""
    url = _HTTP_URL_ADAPTER.validate_python(value)
    if url.scheme != "https" or value != value.strip():
        raise ValueError("URL must be HTTPS without outer whitespace")
    if "@" in urlsplit(value).netloc or url.username or url.password:
        raise ValueError("URL must not contain userinfo")
    return value


HttpsUrl = Annotated[str, AfterValidator(validate_https)]


def validate_date(value: object) -> date | None:
    """Accept dates or exact ISO calendar dates, never timestamp coercion."""
    if value is None or type(value) is date:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return date.fromisoformat(value)
    raise ValueError("date must be YYYY-MM-DD or a date object")


CalendarDate = Annotated[date | None, BeforeValidator(validate_date)]


class Model(BaseModel):
    """Reject unknown fields and accidental reassignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Authority(StrEnum):
    governing = "governing"
    rules = "rules"
    board_decision = "board_decision"
    website = "website"
    form = "form"
    informal = "informal"
    superseded = "superseded"


def authority_rank(a: Authority) -> int:
    """Return 0 (superseded) through 5 (governing), with form/informal tied."""
    return {
        Authority.governing: 5,
        Authority.rules: 4,
        Authority.board_decision: 3,
        Authority.website: 2,
        Authority.form: 1,
        Authority.informal: 1,
        Authority.superseded: 0,
    }[a]


class Chunk(Model):
    """A source passage; superseded_by names the superseding document's doc_id.

    A year-only source date is stored as YYYY-01-01; citation_label carries
    the plain year so consumers can see the original date precision.
    effective_date is the date used for 'newer wins' ordering and must be set
    for every non-superseded authority except form when unknown;
    published_date is informational.
    """

    id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    doc_title: str = Field(min_length=1)
    source_url: HttpsUrl
    page_start: int | None = Field(ge=1, strict=True)
    page_end: int | None = Field(ge=1, strict=True)
    citation_label: str = Field(min_length=1)
    heading_path: tuple[str, ...]
    text_clean: str
    text_raw: str
    authority: Authority
    effective_date: CalendarDate
    published_date: CalendarDate
    superseded_by: str | None
    token_estimate: int = Field(ge=0, strict=True)

    @field_validator("text_clean")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text_clean must not be empty")
        return value

    @model_validator(mode="after")
    def consistent_metadata(self) -> Self:
        if self.superseded_by == self.doc_id:
            raise ValueError("superseded_by must not name the same doc_id")
        if (self.authority == Authority.superseded) != (self.superseded_by is not None):
            raise ValueError(
                "superseded authority requires superseded_by and vice versa"
            )
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must be >= page_start")
        return self


def citation_url(chunk: Chunk) -> str:
    """Add a PDF page fragment, preserving the original path and query string."""
    parts = urlsplit(chunk.source_url)
    if parts.path.lower().endswith(".pdf") and chunk.page_start is not None:
        # urlunsplit drops an empty query delimiter (a trailing ?).
        return urlunsplit(parts._replace(fragment=f"page={chunk.page_start}"))
    return chunk.source_url


class CorpusManifest(Model):
    build_time: AwareDatetime
    source_hashes: dict[str, str]
    chunk_count: int = Field(ge=0, strict=True)
    ocr_fallbacks: tuple[str, ...]


class Corpus(Model):
    manifest: CorpusManifest
    chunks: tuple[Chunk, ...]

    @model_validator(mode="after")
    def consistent_manifest_and_references(self) -> Self:
        ids = [chunk.id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk ids")
        if self.manifest.chunk_count != len(self.chunks):
            raise ValueError("manifest chunk_count does not match chunks")
        doc_ids = {chunk.doc_id for chunk in self.chunks}
        if not doc_ids <= self.manifest.source_hashes.keys():
            raise ValueError("manifest source_hashes missing chunk doc_id")
        for chunk in self.chunks:
            if chunk.superseded_by is not None and chunk.superseded_by not in doc_ids:
                raise ValueError(
                    "superseded_by must name a doc_id present in the corpus"
                )
        return self


def load_corpus(path: str | Path) -> Corpus:
    """Load the manifest + chunks JSON envelope, validating every record."""
    return Corpus.model_validate_json(Path(path).read_text(encoding="utf-8"))


class Citation(Model):
    chunk_id: str = Field(min_length=1)
    citation_label: str = Field(min_length=1)
    url: HttpsUrl
    quote: str


class Outcome(StrEnum):
    answered = "answered"
    not_found = "not_found"
    refused_off_topic = "refused_off_topic"
    budget_exhausted = "budget_exhausted"
    invalid_input = "invalid_input"
    error = "error"


class Answer(Model):
    request_id: str = Field(min_length=1)
    outcome: Outcome
    answer_text: str
    citations: tuple[Citation, ...]
    confidence: float | None = Field(ge=0, le=1, strict=True)
    conflicts_noted: tuple[str, ...]
    disclaimer: str

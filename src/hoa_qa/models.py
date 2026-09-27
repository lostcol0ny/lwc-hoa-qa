"""Shared corpus and response contracts. Larger authority ranks take precedence."""

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    TypeAdapter,
    field_validator,
    model_validator,
)


class Model(BaseModel):
    """Reject unknown fields and accidental reassignment; nested lists remain lists."""

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
    id: str
    doc_id: str
    doc_title: str
    source_url: str
    page_start: int | None = Field(ge=1, strict=True)
    page_end: int | None = Field(ge=1, strict=True)
    citation_label: str
    heading_path: list[str]
    text_clean: str
    text_raw: str
    authority: Authority
    effective_date: date | None
    published_date: date | None
    superseded_by: str | None
    token_estimate: int = Field(ge=0, strict=True)

    @field_validator("source_url")
    @classmethod
    def https_url(cls, value: str) -> str:
        url = TypeAdapter(HttpUrl).validate_python(value)
        if url.scheme != "https" or value != value.strip():
            raise ValueError("source_url must be an HTTPS URL without outer whitespace")
        return value

    @field_validator("text_clean")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text_clean must not be empty")
        return value

    @model_validator(mode="after")
    def page_order(self) -> Self:
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
        return urlunsplit(parts._replace(fragment=f"page={chunk.page_start}"))
    return chunk.source_url


class CorpusManifest(Model):
    build_time: datetime
    source_hashes: dict[str, str]
    chunk_count: int = Field(ge=0, strict=True)
    ocr_fallbacks: list[str]


class Corpus(Model):
    manifest: CorpusManifest
    chunks: list[Chunk]

    @model_validator(mode="after")
    def unique_chunk_ids(self) -> Self:
        ids = [chunk.id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate chunk ids")
        return self


def load_corpus(path: str | Path) -> Corpus:
    """Load the manifest + chunks JSON envelope, validating every record."""
    return Corpus.model_validate_json(Path(path).read_text(encoding="utf-8"))


class Citation(Model):
    chunk_id: str
    citation_label: str
    url: str
    quote: str


class Outcome(StrEnum):
    answered = "answered"
    not_found = "not_found"
    refused_off_topic = "refused_off_topic"
    budget_exhausted = "budget_exhausted"
    invalid_input = "invalid_input"
    error = "error"


class Answer(Model):
    request_id: str
    outcome: Outcome
    answer_text: str
    citations: list[Citation]
    confidence: float | None = Field(ge=0, le=1, strict=True)
    conflicts_noted: list[str]
    disclaimer: str

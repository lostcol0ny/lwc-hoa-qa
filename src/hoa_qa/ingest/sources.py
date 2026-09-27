"""Validated, reviewable source registry."""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, TypeAdapter, model_validator

from hoa_qa.models import Authority, CalendarDate, HttpsUrl, Model


class Source(Model):
    doc_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    url: HttpsUrl
    kind: Literal["pdf", "blog_post", "web_page"]
    authority: Authority
    # "crawl": website pages are dated by the build that fetched them.
    effective_date: Literal["crawl"] | CalendarDate
    # Displayed post date, when it differs in meaning from the content date.
    published_date: CalendarDate = None
    superseded_by: str | None = None
    exclude_pages: tuple[int, ...] = ()
    exclude: bool = False
    exclude_reason: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.exclude and not self.exclude_reason:
            raise ValueError("excluded sources need exclude_reason")
        if any(page < 1 for page in self.exclude_pages):
            raise ValueError("exclude_pages are 1-based PDF indices")
        if self.kind != "pdf" and self.exclude_pages:
            raise ValueError("only PDFs have exclude_pages")
        if (self.authority == Authority.superseded) != bool(self.superseded_by):
            raise ValueError("superseded sources must name their replacement")
        if self.effective_date == "crawl" and self.kind != "web_page":
            raise ValueError("only web pages are dated by crawl")
        if self.effective_date is None and self.authority not in (
            Authority.form,
            Authority.superseded,
        ):
            raise ValueError("effective_date is required for this authority")
        return self

    def effective(self, crawl_date: date) -> date | None:
        if self.effective_date == "crawl":
            return crawl_date
        return self.effective_date


def load_sources(path: Path) -> tuple[Source, ...]:
    sources = TypeAdapter(tuple[Source, ...]).validate_python(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    ids = [source.doc_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source doc_id")
    included = {s.doc_id for s in sources if not s.exclude}
    for source in sources:
        if source.superseded_by and (
            source.superseded_by not in included
            or source.superseded_by == source.doc_id
        ):
            raise ValueError("superseded_by must name another included source")
    return sources

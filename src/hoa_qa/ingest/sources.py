"""Validated, reviewable source registry."""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, TypeAdapter, model_validator

from hoa_qa.models import Authority, CalendarDate, HttpsUrl, Model

ILGA_FTP_ILCS = "https://ftp.ilga.gov/ILCS/"
ILGA_PUBLIC_ACT = "https://www.ilga.gov/Legislation/PublicActs/View/"


class Ilcs(Model):
    """Where an Act lives in the Illinois Compiled Statutes."""

    chapter: int = Field(ge=1, le=9999)
    # The Act number as cited ("805 ILCS 105" is chapter 805, act 105).
    act: int = Field(ge=1, le=9999)
    # Articles to ingest; empty means the whole Act.
    articles: tuple[str, ...] = ()

    @property
    def citation(self) -> str:
        return f"{self.chapter} ILCS {self.act}"

    @property
    def prefix(self) -> str:
        """ILGA document-name prefix: 4-digit chapter, 5-digit act (``aaaa0``)."""
        return f"{self.chapter:04d}{self.act:04d}0"

    @property
    def listing_url(self) -> str:
        return f"{ILGA_FTP_ILCS}Ch%20{self.chapter:04d}/Act%20{self.act:04d}/"


class Source(Model):
    doc_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    url: HttpsUrl
    kind: Literal["pdf", "blog_post", "web_page", "statute", "public_act"]
    authority: Authority
    # "crawl": website pages are dated by the build that fetched them.
    # "section": each statute section is dated by its own source note.
    effective_date: Literal["crawl", "section"] | CalendarDate
    # Displayed post date, when it differs in meaning from the content date.
    published_date: CalendarDate = None
    superseded_by: str | None = None
    exclude_pages: tuple[int, ...] = ()
    exclude: bool = False
    exclude_reason: str | None = None
    ilcs: Ilcs | None = None
    # public_act only: its number ("104-0734", as ILGA's URL writes it) and
    # the doc_id of the ingested statute it amends (see ingest.acts).
    public_act: str | None = Field(default=None, pattern=r"^\d{2,3}-\d{4}$")
    amends: str | None = None
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
        statute = self.kind == "statute"
        law = statute or self.kind == "public_act"
        if law != (self.authority == Authority.statute):
            raise ValueError(
                "statutes and public acts, and only they, have statute authority"
            )
        act = self.kind == "public_act"
        if act != (self.public_act is not None) or act != (self.amends is not None):
            raise ValueError("public acts, and only they, name public_act and amends")
        expected = f"{ILGA_PUBLIC_ACT}{self.public_act}"
        if act and self.url != expected:
            raise ValueError(f"public act url must be {expected}")
        if statute != (self.effective_date == "section"):
            raise ValueError("statutes, and only statutes, are dated per section")
        if statute != (self.ilcs is not None):
            raise ValueError("statutes, and only statutes, name their ilcs location")
        if self.ilcs is not None and self.url != self.ilcs.listing_url:
            raise ValueError(f"statute url must be {self.ilcs.listing_url}")
        if self.effective_date is None and self.authority not in (
            Authority.form,
            Authority.superseded,
        ):
            raise ValueError("effective_date is required for this authority")
        return self

    def effective(self, crawl_date: date) -> date | None:
        if self.effective_date == "crawl":
            return crawl_date
        if self.effective_date == "section":
            raise ValueError(f"{self.doc_id}: statute sections carry their own dates")
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
    statutes = {s.doc_id for s in sources if s.kind == "statute" and not s.exclude}
    for source in sources:
        if source.amends is not None and source.amends not in statutes:
            raise ValueError(f"{source.doc_id}: amends must name an included statute")
    return sources

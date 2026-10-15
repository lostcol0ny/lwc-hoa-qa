"""Public Acts that have passed but are not yet in ILGA's compiled statutes.

ILGA compiles an amendment into the ILCS text (``statutes/``) only after it
takes effect, sometimes months later. A ``public_act`` source bridges that
gap: the Act's own text, as enacted, for the sections it amends in an
ingested statute source. Only the subsections holding new text are kept
(an amended section is reprinted whole, mostly unchanged), and each chunk
is dated by the Act's effective date, so the answer pipeline can tell a
scheduled change from current law (see ``models.not_yet_effective``).

The Act page lives on ILGA's interactive site; its file repository has only
the bills as introduced. An enacted Act never changes, so the page is fetched
once into a reviewed, hash-checked snapshot under ``statutes/<doc_id>/``,
like the ILCS files. Once the amended Act's snapshot is compiled through
this Public Act, the build skips the source and says it can be removed.

On the page, inserted text is underlined (``<u>``) and deleted text struck
through (``<s>``/``<strike>``); each printed line is one ``td.xsl`` cell, and
a cell opening with an indent starts a new paragraph.
"""

import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from pydantic import Field

from hoa_qa.ingest.section import _suffix
from hoa_qa.ingest.sources import Source
from hoa_qa.ingest.statutes import (
    MANIFEST,
    STATUTE_MAX_CHARS,
    Snapshot,
    _label_style,
    _pack,
    snapshot_dir,
)
from hoa_qa.models import Chunk, Model

PAGE = "act.html"

_CITATION = re.compile(r"^\((\d+) ILCS (\d+)/([0-9A-Za-z.-]+)\)$")
_SECTION = re.compile(r"^Sec\. ([0-9A-Za-z.-]+)\.\s*(.*)$")
_SOURCE_NOTE = re.compile(r"^\(Source: P\.A\. ")
_ACT_SECTION = re.compile(r"^Section \d+\. ")


class ActSnapshot(Model):
    """The one page fetched for a Public Act, checked on every build."""

    doc_id: str
    url: str
    fetched_on: date
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_act(root: Path, source: Source) -> bytes:
    """Read and hash-check one Public Act's snapshot."""
    folder = snapshot_dir(root, source)
    snapshot = ActSnapshot.model_validate_json((folder / MANIFEST).read_bytes())
    if snapshot.doc_id != source.doc_id or snapshot.url != source.url:
        raise ValueError(f"{source.doc_id}: snapshot is for another source")
    page = (folder / PAGE).read_bytes()
    if hashlib.sha256(page).hexdigest() != snapshot.sha256:
        raise ValueError(f"{source.doc_id}: {PAGE} does not match its hash")
    return page


def snapshot_act(
    root: Path, source: Source, fetch: Callable[[str], bytes], today: date
) -> ActSnapshot:
    """Fetch a Public Act's page and write its snapshot."""
    page = fetch(source.url)
    paragraphs(page)  # Fail now, not at the next build, if it doesn't parse.
    snapshot = ActSnapshot(
        doc_id=source.doc_id,
        url=source.url,
        fetched_on=today,
        sha256=hashlib.sha256(page).hexdigest(),
    )
    folder = snapshot_dir(root, source)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / PAGE).write_bytes(page)
    (folder / MANIFEST).write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot


def public_act_key(number: str) -> tuple[int, int]:
    """'104-734' or '104-0734' -> (104, 734), for ordering Public Acts."""
    ga, act = number.split("-")
    return int(ga), int(act)


def compiled(source: Source, amended: Snapshot) -> bool:
    """True once ILGA's compiled copy of the amended Act includes this one."""
    assert source.public_act is not None
    return public_act_key(amended.through_public_act) >= public_act_key(
        source.public_act
    )


@dataclass(frozen=True)
class Paragraph:
    """One printed paragraph of the Act, with deleted text already removed."""

    text: str
    inserted: bool  # Any of it is new (underlined).
    all_inserted: bool  # All of it is new.


def paragraphs(page: bytes) -> list[Paragraph]:
    """The Act's paragraphs in order, as it will read once in effect."""
    soup = BeautifulSoup(page, "html.parser")
    cells = soup.select("td.xsl")
    if not cells:
        raise ValueError("no Public Act text cells (td.xsl) on the page")
    found: list[Paragraph] = []
    kept: list[str] = []
    new: list[bool] = []

    def flush() -> None:
        text = " ".join("".join(kept).split())
        if text:
            marks = [n for n, piece in zip(new, kept, strict=True) if piece.strip()]
            found.append(Paragraph(text, any(marks), all(marks)))
        kept.clear()
        new.clear()

    for cell in cells:
        if _opens_paragraph(cell):
            flush()
        for node in cell.descendants:
            if type(node) is not NavigableString:
                continue
            names = {parent.name for parent in node.parents}
            if names & {"s", "strike", "del"}:
                continue
            if str(node).replace("\xa0", " ").lstrip().startswith("(Source: P.A."):
                # Printed on the section's last line: it ends the section, and
                # is the old note, never part of the new text.
                flush()
            kept.append(str(node).replace("\xa0", " ") + " ")
            new.append(bool(names & {"u", "ins"}))
    flush()
    return found


def _opens_paragraph(cell: Tag) -> bool:
    """A cell whose first element is an indent-only <code> starts a paragraph."""
    for child in cell.children:
        if isinstance(child, NavigableString):
            if child.strip():
                return False
            continue
        if isinstance(child, Tag):
            return (
                child.name == "code"
                and not child.get_text().replace("\xa0", " ").strip()
            )
    return False


@dataclass(frozen=True)
class Amendment:
    """One amended subsection that holds new text."""

    section: str  # "1-45"
    caption: str  # "Finances."
    subsection: str  # "j"
    lines: tuple[str, ...]
    changed: str  # Where the new text is: "(j)", "(d)(8)".


def amendments(
    page: bytes, chapter: int, act: int, warnings: list[str], label: str
) -> list[Amendment]:
    """The subsections of ``chapter`` ILCS ``act`` this Public Act changes."""
    found: list[Amendment] = []
    section: str | None = None
    caption = ""
    body: list[Paragraph] = []

    def close() -> None:
        if section is not None:
            found.extend(_changed_subsections(section, caption, body))
            if not any(p.inserted for p in body):
                warnings.append(f"{label}: Sec. {section} has no new text")

    for paragraph in paragraphs(page):
        text = paragraph.text
        citation = _CITATION.match(text)
        if citation:
            close()
            ours = (int(citation[1]), int(citation[2])) == (chapter, act)
            section, caption, body = (citation[3] if ours else None), "", []
            continue
        if section is None:
            continue
        heading = _SECTION.match(text)
        if heading and not caption and heading[1] == section:
            caption = heading[2]
        elif _SOURCE_NOTE.match(text) or _ACT_SECTION.match(text):
            close()
            section = None
        else:
            body.append(paragraph)
    close()
    if not found:
        raise ValueError(f"{label}: no new text for {chapter} ILCS {act}")
    return found


def _changed_subsections(
    section: str, caption: str, body: list[Paragraph]
) -> list[Amendment]:
    """Group a section's paragraphs into lettered subsections; keep changed ones.

    Unlabeled paragraphs ("The principal officer ... shall furnish the above
    information") belong to the subsection they follow.
    """
    groups: list[tuple[str, list[Paragraph]]] = []
    last_letter = ""
    previous = ""
    for paragraph in body:
        style = _label_style(
            paragraph.text, last_letter, after_lead_in=previous.endswith(":")
        )
        if style == "lower" and (not groups or not previous.endswith(":")):
            last_letter = paragraph.text[1]
            groups.append((last_letter, []))
        if not groups:
            groups.append(("", []))  # Text before (a): a section without letters.
        groups[-1][1].append(paragraph)
        previous = paragraph.text
    kept = []
    for letter, items in groups:
        if not any(p.inserted for p in items):
            continue
        where = f"({letter})" if letter else ""
        if not all(p.all_inserted for p in items):
            places = []
            for p in items[1:] if letter else items:
                if p.inserted:
                    match = re.match(r"^\(([0-9A-Za-z]{1,5})\)", p.text)
                    places.append(f"{where}({match[1]})" if match else where)
            where = ", ".join(dict.fromkeys(places)) or where
        kept.append(
            Amendment(
                section=section,
                caption=caption,
                subsection=letter,
                lines=tuple(p.text for p in items),
                changed=where or f"Sec. {section}",
            )
        )
    return kept


def act_chunks(
    source: Source,
    amended: Source,
    page: bytes,
    warnings: list[str],
) -> list[Chunk]:
    """Chunks for the changed subsections, dated by the Act's effective date."""
    assert amended.ilcs is not None and source.public_act is not None
    assert isinstance(source.effective_date, date)
    ilcs = amended.ilcs
    act = f"{amended.title} ({ilcs.citation})"
    effective = source.effective_date
    number = "-".join(str(n) for n in public_act_key(source.public_act))
    chunks: list[Chunk] = []
    for change in amendments(
        page, ilcs.chapter, ilcs.act, warnings, f"{source.doc_id}"
    ):
        caption = change.caption.rstrip(".").strip()
        where = f"({change.subsection})" if change.subsection else ""
        label = (
            f"{ilcs.citation}/{change.section}{where} ({caption}), as amended by "
            f"P.A. {number}, effective {effective.isoformat()}"
        )
        context = (
            f"{act}: {ilcs.citation}/{change.section}{where} ({caption}), as "
            f"amended by P.A. {number}, effective {effective.isoformat()}. New or "
            f"changed text: {change.changed}."
        )
        opening = [
            f"({ilcs.citation}/{change.section})",
            f"Sec. {change.section}. {change.caption}",
        ]
        groups = _pack([*opening, *change.lines], STATUTE_MAX_CHARS - len(context) - 20)
        key = f"{change.section}-{change.subsection}".rstrip("-")
        for index, group in enumerate(groups):
            text = "\n".join(
                [context if index == 0 else f"{context} (continued)", *group]
            )
            chunks.append(
                Chunk(
                    id=f"{source.doc_id}-{key}"
                    + ("" if len(groups) == 1 else f"-{_suffix(index)}"),
                    doc_id=source.doc_id,
                    doc_title=source.title,
                    source_url=source.url,
                    page_start=None,
                    page_end=None,
                    citation_label=label,
                    heading_path=(
                        act,
                        source.title,
                        f"Sec. {change.section}. {change.caption}",
                    ),
                    text_clean=text,
                    text_raw=text,
                    authority=source.authority,
                    effective_date=effective,
                    published_date=None,
                    superseded_by=None,
                    token_estimate=math.ceil(len(text) / 4),
                )
            )
    return chunks

"""Illinois statutes from ILGA's public file repository (spec addendum §2-§4).

ILGA asks automated clients to use ``https://ftp.ilga.gov/`` (not the
interactive site) and to wait ``Crawl-delay: 10`` seconds between requests.
That repository's ILCS tree is a static copy that ILGA regenerates about once a
year (see its ``aReadMe.txt``), one HTML file per section. Fetching every
section at the crawl delay takes about half an hour, so the files are kept as a
reviewed, hash-checked snapshot under ``statutes/<doc_id>/`` and refreshed with
``python -m hoa_qa.ingest refresh-statutes``. Every build re-reads ILGA's
directory listings (one request per Act) and fails if ILGA has published files
the snapshot does not match, then selects the versions in force on the build
date.
"""

import hashlib
import html
import json
import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote, unquote

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from pydantic import Field

from hoa_qa.ingest.section import _suffix
from hoa_qa.ingest.sources import ILGA_FTP_ILCS, Source
from hoa_qa.models import Chunk, Model, StatuteCompilation

README_URL = f"{ILGA_FTP_ILCS}aReadMe/aReadMe.txt"
SEQUENCE_URL = f"{ILGA_FTP_ILCS}aReadMe/Section%20Sequence.txt"
MANIFEST = "manifest.json"

_LISTING = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2}\s+[AP]M)\s+(\d+)\s+"
    r'<a\s+href="([^"]+)">([^<]+)</a>',
    re.IGNORECASE,
)
_README = re.compile(
    r"updated on (\d{1,2}/\d{1,2}/\d{4}) with all Public Acts through "
    r"Public Act (\d+-\d+)",
    re.IGNORECASE,
)


class ListingEntry(Model):
    name: str = Field(min_length=1)
    size: int = Field(ge=0)
    stamp: str = Field(min_length=1)


class Snapshot(Model):
    """What ``refresh-statutes`` fetched for one Act, checked on every build."""

    doc_id: str
    listing_url: str
    fetched_on: date
    # ILGA's own statement of the copy's currency, from aReadMe.txt.
    ilga_updated_on: str
    through_public_act: str
    listing: tuple[ListingEntry, ...]
    # Document names in ILGA's official order (Section Sequence.txt).
    sequence: tuple[str, ...]
    sha256: dict[str, str]


def compilation(snapshots: Iterable[Snapshot]) -> StatuteCompilation | None:
    """The least current ILGA copy among the snapshots (they normally agree)."""
    found = [
        StatuteCompilation(
            through_public_act=s.through_public_act,
            updated_on=datetime.strptime(s.ilga_updated_on, "%m/%d/%Y").date(),
        )
        for s in snapshots
    ]
    return min(found, key=lambda c: c.updated_on, default=None)


def parse_listing(page: bytes, prefix: str) -> tuple[ListingEntry, ...]:
    """The Act's documents in an IIS directory listing, in listing order."""
    entries = []
    for day, clock, size, href, name in _LISTING.findall(page.decode("utf-8")):
        name = html.unescape(name)
        if not name.startswith(prefix) or not name.endswith(".html"):
            continue
        if unquote(href.rsplit("/", 1)[-1]) != name:
            raise ValueError(f"listing link does not match its name: {name!r}")
        entries.append(
            ListingEntry(name=name, size=int(size), stamp=f"{day} {clock.upper()}")
        )
    if not entries:
        raise ValueError(f"no {prefix} documents in the ILGA listing")
    return tuple(entries)


def parse_readme(text: bytes) -> tuple[str, str]:
    """(updated-on date, last Public Act included) from ILGA's aReadMe.txt."""
    match = _README.search(text.decode("utf-8", errors="replace"))
    if match is None:
        raise ValueError("ILGA aReadMe.txt no longer states its update date")
    return match[1], match[2]


def parse_sequence(text: bytes, prefix: str) -> tuple[str, ...]:
    """The Act's document names, in ILCS order, from Section Sequence.txt."""
    names = []
    for line in text.decode("utf-8", errors="replace").splitlines():
        name = line.strip()
        if name.startswith(prefix):
            names.append(name if name.endswith(".html") else name + ".html")
    return tuple(names)


def file_url(source: Source, name: str) -> str:
    """The public https URL of one snapshot document on ILGA's repository."""
    return source.url + quote(name)


def snapshot_dir(root: Path, source: Source) -> Path:
    return root / source.doc_id


def load_snapshot(root: Path, source: Source) -> tuple[Snapshot, dict[str, bytes]]:
    """Read and hash-check one Act's snapshot."""
    folder = snapshot_dir(root, source)
    snapshot = Snapshot.model_validate_json((folder / MANIFEST).read_bytes())
    if snapshot.doc_id != source.doc_id or snapshot.listing_url != source.url:
        raise ValueError(f"{source.doc_id}: snapshot is for another source")
    documents: dict[str, bytes] = {}
    for entry in snapshot.listing:
        data = (folder / entry.name).read_bytes()
        if hashlib.sha256(data).hexdigest() != snapshot.sha256.get(entry.name):
            raise ValueError(f"{source.doc_id}: {entry.name} does not match its hash")
        documents[entry.name] = data
    return snapshot, documents


def check_current(
    snapshot: Snapshot, source: Source, fetch: Callable[[str], bytes]
) -> None:
    """Fail if ILGA's listing no longer matches the snapshot (new publication)."""
    prefix = source.ilcs.prefix if source.ilcs else ""
    live = parse_listing(fetch(source.url), prefix)
    if live != snapshot.listing:
        raise ValueError(
            f"{source.doc_id}: ILGA has published files that differ from the "
            f"statutes/ snapshot ({snapshot.ilga_updated_on}). Review them with "
            "`python -m hoa_qa.ingest refresh-statutes` and commit the result."
        )


def refresh(
    root: Path,
    sources: Iterable[Source],
    fetch: Callable[[str], bytes],
    today: date,
) -> list[Snapshot]:
    """Fetch every statute source's documents and rewrite its snapshot."""
    statutes = [s for s in sources if s.kind == "statute" and not s.exclude]
    if not statutes:
        return []
    updated_on, through = parse_readme(fetch(README_URL))
    sequence = fetch(SEQUENCE_URL)
    written = []
    for source in statutes:
        assert source.ilcs is not None
        prefix = source.ilcs.prefix
        listing = parse_listing(fetch(source.url), prefix)
        order = parse_sequence(sequence, prefix)
        missing = {e.name for e in listing} - set(order)
        if missing:
            raise ValueError(f"{source.doc_id}: not in Section Sequence: {missing}")
        documents = {e.name: fetch(file_url(source, e.name)) for e in listing}
        snapshot = Snapshot(
            doc_id=source.doc_id,
            listing_url=source.url,
            fetched_on=today,
            ilga_updated_on=updated_on,
            through_public_act=through,
            listing=listing,
            sequence=tuple(name for name in order if name in documents),
            sha256={
                name: hashlib.sha256(data).hexdigest()
                for name, data in documents.items()
            },
        )
        folder = snapshot_dir(root, source)
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob("*.html"):
            if stale.name not in documents:
                stale.unlink()
        for name, data in documents.items():
            (folder / name).write_bytes(data)
        (folder / MANIFEST).write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(snapshot)
    return written


# Parsing: every ILGA section document (and every section on ILGA's full-Act
# pages) is one <table> container. A container can hold several versions of
# the section, each opened by a "(Text of Section ...)" marker and closed by
# its "(Source: P.A. ...)" note.

_CITATION = re.compile(r"^\((\d+) ILCS (\d+)/([0-9A-Za-z.-]+)\)")
_MARKER = re.compile(
    r"^\(Text of Section (before amendment by|after amendment by|from) "
    r"P\.A\. (\d+-\d+)\s*\)$",
    re.IGNORECASE,
)
_SECTION = re.compile(r"^Sec\. ([0-9A-Za-z.-]+)\.\s*(.*)$")
_SOURCE = re.compile(r"\s*\(Source: (P\.A\. [^()]*?)\s*\.?\s*\)\s*$")
_PUBLIC_ACT = re.compile(r"(\d{2,3})-(\d+)(?:,\s*eff\.\s*(\d{1,2})-(\d{1,2})-(\d{2}))?")
# A list item run onto the previous one: "...; (ii) the ...", "...: (1) ...".
# Statute parts are half the size of other chunks (~400 tokens): a long
# section is a dense run of numbered provisions, and the support check judges
# a claim against its whole cited part (eval run 4: a correct claim citing a
# ~2,900-character list of records scored 0.25).
STATUTE_MAX_CHARS = 1600
_ITEM = re.compile(r"(?<=[.;:])\s+(?=\((?:[a-z]{1,4}|\d{1,3})\)\s)")
_ARTICLE = re.compile(r"^\(\d+ ILCS \d+/Art\. ([^ )]+) heading\)\s*(.*?)\s*(?:\(|$)")
_KINDS = {
    "before amendment by": "before",
    "after amendment by": "after",
    "from": "from",
}


@dataclass(frozen=True)
class Version:
    """One version of one statute section, as ILGA compiles it."""

    section: str
    caption: str
    lines: tuple[str, ...]
    source_note: str
    # "only" (no marker), or "before"/"after"/"from" a named Public Act.
    kind: str = "only"
    public_act: str | None = None

    @property
    def effective(self) -> date | None:
        """Effective date of the latest Public Act in the source note."""
        return source_effective(self.source_note)

    @property
    def latest_act(self) -> str:
        """The latest Public Act in the source note, e.g. "103-486"."""
        ga, number = max(public_act_dates(self.source_note))
        return f"{ga}-{number}"

    @property
    def repealed(self) -> bool:
        caption = self.caption.lower()
        return caption.startswith(("(repealed", "(blank")) or (
            not self.lines and "repealed" in caption
        )


def _pa_key(ga: str, number: str) -> tuple[int, int]:
    return int(ga), int(number)


def public_act_dates(note: str) -> dict[tuple[int, int], date | None]:
    """Public Acts in a source note, each with its effective date if stated."""
    acts: dict[tuple[int, int], date | None] = {}
    for ga, number, month, day, year in _PUBLIC_ACT.findall(note):
        when = None
        if year:
            # Two-digit years: P.A. numbering starts in the 1970s (77th GA).
            full = int(year) + (1900 if int(year) >= 70 else 2000)
            when = date(full, int(month), int(day))
        acts[_pa_key(ga, number)] = when
    return acts


def source_effective(note: str) -> date | None:
    acts = public_act_dates(note)
    if not acts:
        raise ValueError(f"no Public Act in source note {note!r}")
    return acts[max(acts)]


def container_lines(container: Tag) -> list[str]:
    """Visible lines of one section container: <br> breaks, other space folded."""
    pieces: list[str] = []
    for node in container.descendants:
        # Exactly NavigableString: comments and the doctype are subclasses.
        if type(node) is NavigableString:
            pieces.append(re.sub(r"\s+", " ", str(node)))
        elif isinstance(node, Tag) and node.name == "br":
            pieces.append("\n")
    lines: list[str] = []
    for line in "".join(pieces).split("\n"):
        # Nested layout tables run list items together without breaks.
        lines += [" ".join(item.split()) for item in _ITEM.split(line)]
    return [line for line in lines if line]


def versions(page: bytes) -> list[Version]:
    """Every section version in an ILGA section document or full-Act page."""
    found: list[Version] = []
    for container in _containers(page):
        found += _container_versions(container_lines(container))
    return found


def _containers(page: bytes) -> list[Tag]:
    """Top-level tables; nested ones indent paragraphs inside a section."""
    soup = BeautifulSoup(page, "html.parser")
    return [
        table for table in soup.find_all("table") if table.find_parent("table") is None
    ]


def _container_versions(lines: list[str]) -> list[Version]:
    result: list[Version] = []
    kind, act = "only", None
    section = caption = None
    body: list[str] = []
    cited = None
    for line in lines:
        citation = _CITATION.match(line)
        if citation and section is None:
            cited = citation[3]
            continue
        marker = _MARKER.match(line)
        if marker and section is None:
            kind, act = _KINDS[marker[1].lower()], marker[2]
            continue
        if section is None:
            heading = _SECTION.match(line)
            if heading is None:
                continue  # "(This Section may contain text ...)" and the like.
            section, caption = heading[1], heading[2]
            if cited is not None and cited != section:
                raise ValueError(f"section {section} filed under {cited}")
            line = ""
        note = _SOURCE.search(line) if line else _SOURCE.search(caption or "")
        if note is None:
            if line:
                body.append(line)
            continue
        if line:
            rest = line[: note.start()].strip()
            if rest:
                body.append(rest)
        else:
            caption = (caption or "")[: note.start()].strip()
        caption, body = _split_caption(caption or "", body)
        result.append(Version(section, caption, tuple(body), note[1], kind, act))
        kind, act, section, caption, body = "only", None, None, None, []
    if section is not None:
        raise ValueError(f"section {section} has no source note")
    return result


def _split_caption(caption: str, body: list[str]) -> tuple[str, list[str]]:
    """Separate a caption from body text that shares its line.

    ILGA usually breaks the line after the caption ("Finances."), but some
    sections run straight on ("Law enforcement ... vehicles. An association
    may not ..."). The caption is the text up to the first sentence end.
    """
    match = re.match(r"^(.*?[.;:])(?:\s+(?=[A-Z(\"“])(.*))?$", caption)
    if match is None or not match[2]:
        return caption, body
    return match[1], [match[2], *body]


def in_force(
    found: Iterable[Version], build_date: date, warnings: list[str], label: str
) -> list[Version]:
    """The versions of one section that are law on ``build_date``.

    - "before amendment by P.A. N" is law until its "after" twin takes effect;
      the twin must be present (its source note dates P.A. N).
    - Every other version is law once its latest Public Act is effective.
    - Of several in-force "from P.A. N" versions (parallel amendments), only
      those with the latest effective date remain; older ones were replaced.
    - A section whose every version is future-effective is left out: ILGA may
      already have removed the text currently in force.
    """
    found = [v for v in found if not v.repealed]
    after = {v.public_act: v for v in found if v.kind == "after"}
    kept: list[Version] = []
    for version in found:
        if version.kind == "before":
            twin = after.get(version.public_act)
            if twin is None:
                raise ValueError(f"{label}: no text after P.A. {version.public_act}")
            if not _effective_by(twin, build_date):
                kept.append(version)
        elif _effective_by(version, build_date):
            kept.append(version)
    parallel = [v for v in kept if v.kind == "from"]
    if len(parallel) > 1:
        latest = max(_date(v) for v in parallel)
        kept = [v for v in kept if v.kind != "from" or _date(v) == latest]
    if found and not kept:
        dates = ", ".join(
            sorted({str(v.effective) for v in found if v.effective is not None})
        )
        warnings.append(f"{label}: only future-effective text ({dates}); not ingested")
    return kept


def _date(version: Version) -> date:
    return version.effective or date.min


def _effective_by(version: Version, build_date: date) -> bool:
    return _date(version) <= build_date


@dataclass(frozen=True)
class Part:
    """One chunk's worth of an in-force section version."""

    key: str
    label: str
    heading: tuple[str, ...]
    text: str
    effective: date | None
    url: str


def article_title(page: bytes) -> tuple[str, str]:
    """(article id, display title) from an ILGA article heading document."""
    for container in _containers(page):
        for line in container_lines(container):
            match = _ARTICLE.match(line)
            if match:
                return match[1], match[2] or f"Article {match[1]}"
    raise ValueError("article heading document without a heading")


def statute_parts(
    source: Source,
    snapshot: Snapshot,
    documents: dict[str, bytes],
    build_date: date,
    warnings: list[str],
) -> list[Part]:
    """The in-force sections of one Act, in ILCS order, split into parts."""
    assert source.ilcs is not None
    ilcs = source.ilcs
    act = f"{source.title} ({ilcs.citation})"
    article: tuple[str, str] | None = None
    parts: list[Part] = []
    for name in snapshot.sequence:
        kind = name[len(ilcs.prefix)]
        if kind == "H":
            article = article_title(documents[name])
            continue
        if kind != "K":
            continue
        if article is None:
            raise ValueError(f"{source.doc_id}: {name} precedes every article")
        if ilcs.articles and article[0] not in ilcs.articles:
            continue
        label = f"{ilcs.citation}/{name[len(ilcs.prefix) + 1 : -len('.html')]}"
        kept = in_force(versions(documents[name]), build_date, warnings, label)
        for version in kept:
            parts += _parts(
                source,
                act,
                article[1],
                version,
                parallel=len(kept) > 1,
                url=file_url(source, name),
            )
    for part in parts:
        if part.effective is not None and part.effective > build_date:
            raise ValueError(f"{part.label}: effective {part.effective} is future")
    return parts


def _parts(
    source: Source,
    act: str,
    article: str,
    version: Version,
    *,
    parallel: bool,
    url: str,
) -> list[Part]:
    assert source.ilcs is not None
    caption = version.caption.rstrip(".").strip()
    label = f"{source.ilcs.citation}/{version.section} ({caption})"
    key = version.section
    if parallel and version.public_act:
        label += f", text from P.A. {version.public_act}"
        key += f"-pa{version.public_act}"
    effective = version.effective
    # Old Public Acts carry no date in ILGA's source notes.
    when = f", effective {effective.isoformat()}" if effective else ""
    context = (
        f"{act}, {article}: {label}, as amended by P.A. {version.latest_act}{when}."
    )
    opening = [f"({source.ilcs.citation}/{version.section})"]
    opening.append(f"Sec. {version.section}. {version.caption}")
    groups = _pack([*opening, *version.lines], STATUTE_MAX_CHARS - len(context) - 20)
    heading = (act, article, f"Sec. {version.section}. {version.caption}")
    return [
        Part(
            key=key if len(groups) == 1 else f"{key}-{_suffix(index)}",
            label=label,
            heading=heading,
            text="\n".join(
                [context if index == 0 else f"{context} (continued)", *group]
            ),
            effective=effective,
            url=url,
        )
        for index, group in enumerate(groups)
    ]


def _pack(lines: list[str], limit: int) -> list[list[str]]:
    """Greedy groups of whole lines; an oversized line splits at sentences."""
    units: list[str] = []
    for line in lines:
        if len(line) <= limit:
            units.append(line)
            continue
        sentence = ""
        for piece in re.split(r"(?<=[.;:])\s+", line):
            while len(piece) > limit:  # A single run-on "sentence".
                cut = piece.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                if sentence:
                    units.append(sentence)
                    sentence = ""
                units.append(piece[:cut])
                piece = piece[cut:].strip()
            if sentence and len(sentence) + 1 + len(piece) > limit:
                units.append(sentence)
                sentence = piece
            else:
                sentence = f"{sentence} {piece}".strip()
        if sentence:
            units.append(sentence)
    groups: list[list[str]] = []
    size = 0
    for unit in units:
        if groups and size + 1 + len(unit) <= limit:
            groups[-1].append(unit)
            size += 1 + len(unit)
            continue
        # Keep a list's lead-in ("... shall maintain the following records:")
        # with its first item, so one part states the whole requirement.
        carried: list[str] = []
        if groups and len(groups[-1]) > 1 and groups[-1][-1].endswith(":"):
            lead = groups[-1][-1]
            if len(lead) + 1 + len(unit) <= limit:
                carried = [groups[-1].pop()]
        groups.append([*carried, unit])
        size = sum(len(u) + 1 for u in groups[-1]) - 1
    return groups


def statute_chunks(
    source: Source,
    snapshot: Snapshot,
    documents: dict[str, bytes],
    build_date: date,
    warnings: list[str],
) -> list[Chunk]:
    return [
        Chunk(
            id=f"{source.doc_id}-{part.key}",
            doc_id=source.doc_id,
            doc_title=source.title,
            source_url=part.url,
            page_start=None,
            page_end=None,
            citation_label=part.label,
            heading_path=part.heading,
            text_clean=part.text,
            text_raw=part.text,
            authority=source.authority,
            effective_date=part.effective,
            published_date=None,
            superseded_by=None,
            token_estimate=math.ceil(len(part.text) / 4),
        )
        for part in statute_parts(source, snapshot, documents, build_date, warnings)
    ]

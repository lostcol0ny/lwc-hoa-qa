"""Source-structure splitting; continuations never cross section boundaries."""

import re
from collections import Counter
from dataclasses import dataclass

from hoa_qa.ingest.cleanup import normalize
from hoa_qa.ingest.extract import Page
from hoa_qa.ingest.sources import Source

ARTICLE = re.compile(
    r"^(?:Art(?:icle|icie|ic1e|vicle)|Anicle|Acricle|Arvicle|Atrticle)"
    r"\s+(\d+)(?:[.;:]?$|[:;]\s+)",
    re.I,
)
SECTION = re.compile(r'^(\d+\.\d+(?:\.\d+)*)(?:[.:]?\s+(?=[A-Z"(“])|\s*$)')
EXHIBIT = re.compile(r"^(?:EXHIBIT|EXHItsIT|EXIIIBIl)[\s\']*([A-Z])\b", re.I)
AGENDA = re.compile(r"^(?:[IVXl][IVXl ]*|\d+)[.:]\s+\S", re.I)
TOPIC = re.compile(r"^(?:\d+[.)]\s+|[A-Z][A-Z /&-]{3,}:)")


@dataclass(frozen=True)
class Section:
    key: str
    heading: tuple[str, ...]
    lines: tuple[tuple[str, Page], ...]

    @property
    def text(self) -> str:
        return "\n".join(line for line, _ in self.lines)


def sections(source: Source, pages: list[Page]) -> list[Section]:
    if source.doc_id == "arch-form":
        return [
            Section("application", (source.title,), tuple((p.text, p) for p in pages))
        ]
    result: list[Section] = []
    lines: list[tuple[str, Page]] = []
    key = "intro"
    heading = (source.title,)
    article: str | None = None
    sequence = 0
    in_exhibit = False
    structural = source.doc_id in {
        "declaration",
        "bylaws",
        "articles",
        "rules-2023",
        "rules-2016",
    }

    def flush() -> None:
        if lines:
            result.append(Section(key, heading, tuple(lines)))
            lines.clear()

    for page in pages:
        if page.heading:
            flush()
            sequence += 1
            key, heading = str(sequence), (page.heading,)
        for original in page.text.splitlines():
            line = normalize(original)
            if not line:
                continue
            new_key: str | None = None
            new_heading = heading
            if structural:
                major = ARTICLE.match(line)
                if source.doc_id.startswith("rules-"):
                    major = re.match(r"^SECTION\s+(\d+)(?:\s*[-—–.:]|$)", line, re.I)
                minor = SECTION.match(line)
                exhibit = EXHIBIT.match(line)
                if major and not in_exhibit:
                    article = major[1]
                    new_key = f"article-{article}"
                    new_heading = (f"Article {article}",)
                elif minor and not in_exhibit and int(minor[1].split(".")[0]) <= 30:
                    new_key = minor[1]
                    article = minor[1].split(".")[0]
                    new_heading = (f"Article {article}", f"§{new_key}")
                elif exhibit:
                    article = None
                    in_exhibit = True
                    new_key = "exhibit-" + exhibit[1].lower()
                    new_heading = (line,)
                # Declaration OCR occasionally drops the dot (e.g. 84 = 8.4).
                # Require the active article and a short heading, never rewrite text.
                elif (
                    source.doc_id in {"declaration", "bylaws"}
                    and article
                    and not in_exhibit
                ):
                    compact = re.fullmatch(r"(\d{2,3})\s+([A-Z].*)", line)
                    if compact and compact[1].startswith(article):
                        suffix = compact[1][len(article) :]
                        if suffix:
                            new_key = article + "." + suffix
                            new_heading = (f"Article {article}", f"§{new_key}")
            elif source.authority.value == "board_decision":
                if AGENDA.match(line) or line.lower().startswith("open forum:"):
                    sequence += 1
                    new_key, new_heading = str(sequence), (line,)
            elif not page.heading:
                if (
                    TOPIC.match(line)
                    or SECTION.match(line)
                    or re.match(r"^[A-Z][A-Z /&-]{3,}[.…:]", line)
                    or (len(line) < 100 and line.endswith("?"))
                ):
                    sequence += 1
                    new_key, new_heading = str(sequence), (line,)
            if new_key:
                flush()
                key, heading = new_key, new_heading
            lines.append((original, page))
    flush()
    # Repeated headings (e.g. TOC vs body or reused exhibit letters) get explicit
    # occurrence ids, not collisions or merging of unrelated sections.
    counts: Counter[str] = Counter()
    output = []
    for section in result:
        counts[section.key] += 1
        suffix = "" if counts[section.key] == 1 else f"-occ{counts[section.key]}"
        output.append(Section(section.key + suffix, section.heading, section.lines))
    return output


def parts(section: Section, *, keep_together: bool = False) -> list[Section]:
    if keep_together or len(normalize(section.text)) <= 3200:
        return [section]
    # Group source lines into sentences first, so a page/line break does not split
    # a sentence. Every part retains source page provenance.
    units: list[list[tuple[str, Page]]] = []
    pending: list[tuple[str, Page]] = []
    for line, page in section.lines:
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            pending.append((sentence, page))
            if re.search(r"[.!?][\"')]*$", sentence.strip()):
                units.append(pending)
                pending = []
    if pending:
        units.append(pending)
    groups: list[list[tuple[str, Page]]] = []
    current: list[tuple[str, Page]] = []
    for unit in units:
        if current and len(normalize(" ".join(t for t, _ in current + unit))) > 3200:
            groups.append(current)
            current = []
        current.extend(unit)
    if current:
        groups.append(current)
    # Extremely long OCR sentences have no punctuation: split at source lines.
    bounded: list[list[tuple[str, Page]]] = []
    for group in groups:
        current = []
        for line, page in group:
            words = line.split()
            buffer = ""
            for word in words:
                if len(buffer) + len(word) > 3000:
                    if current:
                        bounded.append(current)
                        current = []
                    bounded.append([(buffer, page)])
                    buffer = ""
                buffer = (buffer + " " + word).strip()
            if (
                current
                and len(normalize(" ".join(t for t, _ in current))) + len(buffer) > 3200
            ):
                bounded.append(current)
                current = []
            if buffer:
                current.append((buffer, page))
        if current:
            bounded.append(current)

    def suffix(index: int) -> str:
        value = ""
        while True:
            value = chr(97 + index % 26) + value
            index = index // 26 - 1
            if index < 0:
                return value

    return [
        Section(f"{section.key}-{suffix(i)}", section.heading, tuple(group))
        for i, group in enumerate(bounded)
    ]


def citation(source: Source, section: Section) -> str:
    first = section.lines[0][1]
    last = section.lines[-1][1]
    label = source.title + ", " + ", ".join(section.heading)
    if source.doc_id in {"articles", "bylaws"}:
        label += " (2001)"
    if first.number is not None:
        start = first.printed or str(first.number)
        end = last.printed or str(last.number)
        label += (
            f", p. {start}" if first.number == last.number else f", pp. {start}–{end}"
        )
    return label

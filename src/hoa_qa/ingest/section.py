"""Source-structure splitting; continuations never cross section boundaries.

Structured documents (Declaration, Bylaws, Articles, Rules) are split on their own
numbering. Every numbered heading must be a plausible *next* key after the previous
one, which keeps OCR noise (survey distances, dotless numbers such as ``838`` for
§8.8) from inventing or mis-keying sections. Rejected explicit headings are logged
in ``warnings`` and stay in the enclosing section's text.
"""

import re
from collections import Counter
from dataclasses import dataclass

from hoa_qa.ingest.cleanup import normalize
from hoa_qa.ingest.extract import Page
from hoa_qa.ingest.sources import Source

ARTICLE_WORD = (
    r"(?:Art(?:icle|icie|ic1e|vicle)|Aticle|Anicle|Acricle|Arvicle|Atrticle|Axticle)"
)
# "Article 8" alone on a line, or "Article 5: Powers." style.
ARTICLE = re.compile(rf"^{ARTICLE_WORD}\s+(\d+)(?:[.;:]?$|[:;]\s+)", re.I)
# "Article 13 Assessments ..." with the title on the same line; only accepted as the
# immediately following article number (see _major).
ARTICLE_TITLED = re.compile(rf"^{ARTICLE_WORD}\s+(\d+)\s+(?=[A-Z])", re.I)
# A heading OCR merged into the end of the previous paragraph's line.
ARTICLE_MIDLINE = re.compile(rf"(?<=[.;:!?])\s+(?={ARTICLE_WORD}\s+\d+\s+[A-Z])", re.I)
# Upper-case only, and not "Section 2.4 of the Declaration" wrapped to a line start.
RULES_SECTION = re.compile(r"^SECTION\s+(\d+)(?!\.\d)(?:\s*[-—–.:]|$)")
SECTION = re.compile(r'^(\d+\.\d+(?:\.\d+)*)(?:[.:]?\s+(?=[A-Z"(“])|\s*$)')
COMPACT = re.compile(r"^(\d{2,4})\s+(?=[A-Z])")
LETTERED = re.compile(r"^([A-H])\.\s+([A-Z].*)$")
EXHIBIT_WORD = r"(?:EXHIBIT|EXHItsIT|EXIIIBIl)"
EXHIBIT_ALONE = re.compile(rf"^{EXHIBIT_WORD}[\s'\"]*([A-Z])[\s'\"]*$", re.I)
EXHIBIT_CAPS = re.compile(rf"^{EXHIBIT_WORD}[\s'\"]*([A-Z])\b")
AGENDA = re.compile(r"^(?:[IVXl][IVXl ]*|\d+)[.:]\s*\S", re.I)
NUMBERED = re.compile(r"^\d+[.)]\s+")
CAPS_HEADING = re.compile(r"^[A-Z][A-Z /&-]{3,}[.…:]")
TERMINAL = re.compile(r"[.!?:…][\"')”’]*$")
FORM_BLANK = re.compile(r"_{4,}")

STRUCTURAL = {"declaration", "bylaws", "articles", "rules-2023", "rules-2016"}
MAX_CHARS = 3200  # ~800 estimated tokens at 4 characters per token.
LABEL_CHARS = 80

Key = tuple[int, ...]


@dataclass(frozen=True)
class Section:
    key: str
    heading: tuple[str, ...]
    lines: tuple[tuple[str, Page], ...]
    # Citation parts after the document title; heading_path may be more verbose.
    label: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(line for line, _ in self.lines)


def shorten(text: str, limit: int = LABEL_CHARS) -> str:
    text = normalize(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def dotted(key: Key) -> str:
    return ".".join(map(str, key))


@dataclass
class _Numbering:
    """Monotonic article/section state for one structured document."""

    doc_id: str
    warnings: list[str]
    article: int | None = None
    last: Key | None = None
    letter: str | None = None

    def expected(self) -> list[Key]:
        """Plausible next keys, most likely first (small gaps tolerate lost lines)."""
        keys: list[Key] = []
        article = self.article
        last = self.last
        if article is not None:
            if last is not None and last[0] == article:
                for depth in range(len(last), 1, -1):
                    for step in (1, 2, 3):
                        keys.append(last[: depth - 1] + (last[depth - 1] + step,))
                    if depth == len(last):
                        keys.append(last + (1,))
            else:
                # An excerpt or a lost first heading may start mid-article.
                keys += [(article, n) for n in range(1, 10)]
            keys += [(article + 1, 1), (article + 1, 2)]
        elif last is None:
            keys += [(1, 1), (1, 2)]
        return keys

    def accept_minor(self, candidates: list[Key], explicit: str | None) -> Key | None:
        expected = self.expected()
        ranked = [k for k in candidates if k in expected]
        if not ranked:
            if explicit is not None:
                after = dotted(self.last) if self.last else f"Article {self.article}"
                self.warnings.append(
                    f"{self.doc_id}: non-monotonic heading {explicit!r} after "
                    f"{after}; kept as text"
                )
            return None
        key = min(ranked, key=expected.index)
        self.article, self.last, self.letter = key[0], key, None
        return key

    def accept_major(self, number: int, *, titled: bool) -> bool:
        if self.article is None:
            ok = not titled or number == 1
        elif titled:
            ok = number == self.article + 1
        else:
            ok = self.article < number <= self.article + 3
        if ok:
            self.article, self.last, self.letter = number, None, None
        return ok


def _split_last(prefix: Key, rest: str) -> list[Key]:
    """Readings of a trailing digit run whose dot OCR may have dropped or misread.

    ``74`` after ``7.`` may be 7.74, 7.7.4 (dot dropped), or 7.4 (dot read as a
    digit, as in dotless ``838`` for 8.8); the caller keeps the expected-next one.
    """
    candidates: list[Key] = [(*prefix, int(rest))] if rest else []
    if len(rest) >= 2:
        candidates.append((*prefix, int(rest[0]), int(rest[1:])))
        candidates.append((*prefix, int(rest[1:])))
    return candidates


def _compact_candidates(digits: str, article: int | None) -> list[Key]:
    """Dotless OCR headings: 84 -> 8.4, 722 -> 7.2.2/7.22, 838 -> 8.8 (dot as 3)."""
    if article is None or not digits.startswith(str(article)):
        return []
    return _split_last((article,), digits[len(str(article)) :])


def _dotted_candidates(heading: str) -> list[Key]:
    """Explicit ``7.74`` may be 7.74 or 7.7.4 (inner dot dropped).

    Unlike a dotless run, an explicit dot is trusted, so ``2.23`` is never read
    as 2.3.
    """
    *prefix, rest = heading.split(".")
    key = tuple(int(p) for p in prefix)
    return _split_last(key, rest)[:2]


def _explode(line: str) -> list[str]:
    """Split a line where an Article heading follows sentence-ending punctuation."""
    return [part for part in ARTICLE_MIDLINE.split(line) if part.strip()]


def _structural(
    source: Source, pages: list[Page], warnings: list[str]
) -> list[Section]:
    rules = source.doc_id.startswith("rules-")
    major_word = "Section" if rules else "Article"
    state = _Numbering(source.doc_id, warnings)
    result: list[Section] = []
    lines: list[tuple[str, Page]] = []
    key, heading, label = "intro", (source.title,), ()
    in_exhibit = False

    def flush() -> None:
        if lines:
            result.append(Section(key, heading, tuple(lines), label))
            lines.clear()

    for page in pages:
        for original in page.text.splitlines():
            for piece in _explode(original) if not rules else [original]:
                line = normalize(piece)
                if not line:
                    continue
                new: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
                exhibit = EXHIBIT_ALONE.match(line) or (
                    EXHIBIT_CAPS.match(line) if line[:7].isupper() else None
                )
                if exhibit:
                    in_exhibit = True
                    letter = exhibit[1].upper()
                    name = f"Exhibit {letter}"
                    new = (f"exhibit-{letter.lower()}", (name,), (name,))
                elif not in_exhibit:
                    new = _heading(line, rules, major_word, state)
                if new:
                    flush()
                    key, heading, label = new
                lines.append((piece, page))
    flush()
    return _fold(_titled(result, major_word))


def _heading(
    line: str, rules: bool, major_word: str, state: _Numbering
) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    if rules:
        major = RULES_SECTION.match(line)
        if major and state.accept_major(int(major[1]), titled=False):
            n = state.article
            return f"section-{n}", (f"Section {n}",), (f"Section {n}",)
    else:
        major = ARTICLE.match(line)
        titled = False
        if not major:
            major, titled = ARTICLE_TITLED.match(line), True
        if major and state.accept_major(int(major[1]), titled=titled):
            n = state.article
            return f"article-{n}", (f"Article {n}",), (f"Article {n}",)
    minor = SECTION.match(line)
    key: Key | None = None
    if minor:
        key = state.accept_minor(_dotted_candidates(minor[1]), explicit=minor[1])
    else:
        compact = COMPACT.match(line)
        if compact:
            key = state.accept_minor(
                _compact_candidates(compact[1], state.article), explicit=None
            )
    if key is not None:
        name = dotted(key)
        return (
            name,
            (f"{major_word} {key[0]}", f"§{name}"),
            (f"{major_word} {key[0]}", f"§{name}"),
        )
    lettered = LETTERED.match(line)
    if (
        rules
        and lettered
        and state.article is not None
        and state.last is None
        and lettered[1] == chr(ord(state.letter or "@") + 1)
    ):
        state.letter = lettered[1]
        name = f"{state.article}.{lettered[1]}"
        title = shorten(lettered[2], 40)
        return (
            name,
            (f"Section {state.article}", f"§{name} {title}"),
            (f"Section {name} ({title})",),
        )
    return None


def _titled(sections: list[Section], major_word: str) -> list[Section]:
    """Put each article's own title (e.g. 'Article 8 Covenant for ...') in paths."""
    titles: dict[str, str] = {}
    for section in sections:
        if section.key.startswith(("article-", "section-")):
            number = section.key.split("-", 1)[1]
            titles[number] = shorten(section.lines[0][0] + " " + _second(section))
    output = []
    for section in sections:
        first = section.heading[0]
        number = first.removeprefix(f"{major_word} ")
        if first.startswith(major_word) and number in titles:
            heading = (titles[number], *section.heading[1:])
            section = Section(section.key, heading, section.lines, section.label)
        output.append(section)
    return output


def _second(section: Section) -> str:
    """The title line after a bare 'Article N' line, if the heading is split."""
    first = normalize(section.lines[0][0])
    if ARTICLE.match(first) and first.split()[-1].rstrip(".:;").isdigit():
        return section.lines[1][0] if len(section.lines) > 1 else ""
    return ""


def _is_child(parent: Section, child: Section) -> bool:
    if parent.key.startswith(("article-", "section-")):
        number = parent.key.split("-", 1)[1]
        return child.key.startswith(number + ".")
    return child.key.startswith(parent.key + ".")


def _heading_only(section: Section) -> bool:
    text = normalize(section.text)
    if section.key.startswith(("article-", "section-")):
        # The title, or the title plus a one-line lead-in ("... have the following
        # meanings:"). Unnumbered substantive text (a lost N.1 heading) stays put.
        lead_in = len(section.lines) <= 3 and text.endswith(":")
        return len(text) <= 100 or lead_in
    return len(text) <= 60


def _fold(sections: list[Section]) -> list[Section]:
    """Merge heading-only sections (e.g. 'Article 2 ...', '5.9 Fences.') forward."""
    output: list[Section] = []
    index = 0
    while index < len(sections):
        section = sections[index]
        following = sections[index + 1] if index + 1 < len(sections) else None
        if (
            following is not None
            and _heading_only(section)
            and _is_child(section, following)
        ):
            heading = following.heading
            folded = shorten(section.text)
            if folded not in heading and not section.key.startswith(
                ("article-", "section-")
            ):
                heading = (*heading[:-1], folded, heading[-1])
            sections[index + 1] = Section(
                following.key,
                heading,
                section.lines + following.lines,
                following.label,
            )
        else:
            output.append(section)
        index += 1
    return output


def _meeting(source: Source, pages: list[Page]) -> list[Section]:
    meeting = f"Meeting {source.effective_date}"
    return [
        Section(
            "meeting",
            (source.title, meeting),
            tuple((line, page) for page in pages for line in page.text.splitlines()),
            (meeting,),
        )
    ]


def _topics(source: Source, pages: list[Page]) -> list[Section]:
    result: list[Section] = []
    lines: list[tuple[str, Page]] = []
    key, heading, label = "intro", (source.title,), ()
    sequence = 0
    previous = ""
    in_form = False

    def flush() -> None:
        if lines:
            result.append(Section(key, heading, tuple(lines), label))
            lines.clear()

    for page in pages:
        if page.heading:
            flush()
            sequence += 1
            key, heading = str(sequence), (page.heading,)
            label = (shorten(page.heading),)
            previous, in_form = "", False
        for original in page.text.splitlines():
            line = normalize(original)
            if not line:
                continue
            boundary = False
            form = bool(FORM_BLANK.search(line))
            if not page.heading:
                # A wrapped ALL-CAPS sentence continues unless the previous line
                # ended a sentence; a run of form blanks stays one chunk.
                after_sentence = not previous or bool(TERMINAL.search(previous))
                if form:
                    boundary = not in_form and after_sentence
                elif (
                    NUMBERED.match(line)
                    or SECTION.match(line)
                    or (len(line) < 100 and line.endswith("?"))
                ):
                    boundary = True
                elif CAPS_HEADING.match(line):
                    boundary = after_sentence and not in_form
            if boundary:
                flush()
                sequence += 1
                key = str(sequence)
                heading = (shorten(line),)
                label = heading
            in_form = form or (in_form and not boundary and bool(line.isupper()))
            lines.append((original, page))
            previous = line
    flush()
    return result


def sections(
    source: Source, pages: list[Page], warnings: list[str] | None = None
) -> list[Section]:
    warnings = warnings if warnings is not None else []
    if source.doc_id == "arch-form":
        result = [
            Section("application", (source.title,), tuple((p.text, p) for p in pages))
        ]
    elif source.authority.value == "board_decision":
        result = _meeting(source, pages)
    elif source.doc_id in STRUCTURAL:
        result = _structural(source, pages, warnings)
    else:
        result = _topics(source, pages)
    # Repeated keys (e.g. reused exhibit letters) get explicit occurrence ids.
    counts: Counter[str] = Counter()
    output = []
    for section in result:
        counts[section.key] += 1
        suffix = "" if counts[section.key] == 1 else f"-occ{counts[section.key]}"
        output.append(
            Section(section.key + suffix, section.heading, section.lines, section.label)
        )
    return output


def _pack(
    units: list[list[tuple[str, Page]]], limit: int = MAX_CHARS
) -> list[list[tuple[str, Page]]]:
    groups: list[list[tuple[str, Page]]] = []
    current: list[tuple[str, Page]] = []
    for unit in units:
        if current and len(normalize(" ".join(t for t, _ in current + unit))) > limit:
            groups.append(current)
            current = []
        current.extend(unit)
    if current:
        groups.append(current)
    return groups


def _suffix(index: int) -> str:
    value = ""
    while True:
        value = chr(97 + index % 26) + value
        index = index // 26 - 1
        if index < 0:
            return value


def _split_text(lines: tuple[tuple[str, Page], ...]) -> list[list[tuple[str, Page]]]:
    # Group source lines into sentences first, so a page/line break does not split
    # a sentence. Every part retains source page provenance.
    units: list[list[tuple[str, Page]]] = []
    pending: list[tuple[str, Page]] = []
    for line, page in lines:
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            pending.append((sentence, page))
            if re.search(r"[.!?][\"')]*$", sentence.strip()):
                units.append(pending)
                pending = []
    if pending:
        units.append(pending)
    groups = _pack(units)
    # Extremely long OCR sentences have no punctuation: split at source lines.
    bounded: list[list[tuple[str, Page]]] = []
    for group in groups:
        current: list[tuple[str, Page]] = []
        for line, page in group:
            buffer = ""
            for word in line.split():
                if len(buffer) + len(word) > MAX_CHARS - 200:
                    if current:
                        bounded.append(current)
                        current = []
                    bounded.append([(buffer, page)])
                    buffer = ""
                buffer = (buffer + " " + word).strip()
            if (
                current
                and len(normalize(" ".join(t for t, _ in current))) + len(buffer)
                > MAX_CHARS
            ):
                bounded.append(current)
                current = []
            if buffer:
                current.append((buffer, page))
        if current:
            bounded.append(current)
    return bounded


def parts(
    section: Section, *, keep_together: bool = False, header: str | None = None
) -> list[Section]:
    """Split sections over ~800 tokens into stable -a/-b parts.

    With ``header`` (minutes), split only between top-level agenda items and repeat
    the header line at the start of every later part so each carries its date.
    """
    if keep_together or len(normalize(section.text)) <= MAX_CHARS:
        return [section]
    if header is not None:
        blocks: list[list[tuple[str, Page]]] = []
        for line, page in section.lines:
            if not blocks or AGENDA.match(normalize(line)):
                blocks.append([])
            blocks[-1].append((line, page))
        groups: list[list[tuple[str, Page]]] = []
        for group in _pack(blocks):
            if len(normalize(" ".join(t for t, _ in group))) > MAX_CHARS:
                groups += _split_text(tuple(group))
            else:
                groups.append(group)
        groups = [
            group if i == 0 else [(header, group[0][1]), *group]
            for i, group in enumerate(groups)
        ]
    else:
        groups = _split_text(section.lines)
    return [
        Section(f"{section.key}-{_suffix(i)}", section.heading, tuple(g), section.label)
        for i, g in enumerate(groups)
    ]


def citation(source: Source, section: Section) -> str:
    first = section.lines[0][1]
    last = section.lines[-1][1]
    label = ", ".join((source.title, *section.label))
    if source.doc_id in {"articles", "bylaws"}:
        label += " (2001)"
    if first.number is not None and last.number is not None:
        # Never mix printed and PDF numbering in one label.
        if first.printed and last.printed:
            start, end, prefix = first.printed, last.printed, ""
        else:
            start, end, prefix = str(first.number), str(last.number), "PDF "
        label += (
            f", {prefix}p. {start}"
            if first.number == last.number
            else f", {prefix}pp. {start}–{end}"
        )
    return label

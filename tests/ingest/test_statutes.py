"""Illinois statutes: ILGA snapshot, in-force versions, chunks, and fetching."""

import hashlib
import html
import json
import ssl
from datetime import date
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError

from hoa_qa.ingest import build
from hoa_qa.ingest.fetch import Fetcher, tls_context
from hoa_qa.ingest.section import MAX_CHARS
from hoa_qa.ingest.sources import Source, load_sources
from hoa_qa.ingest.statutes import (
    README_URL,
    SEQUENCE_URL,
    Snapshot,
    _pack,
    check_current,
    file_url,
    in_force,
    load_snapshot,
    parse_listing,
    public_act_dates,
    refresh,
    source_effective,
    statute_chunks,
    versions,
)
from hoa_qa.models import Authority, citation_url

ROOT = Path(__file__).parents[2]
PREFIX = "076501600"
LISTING_URL = "https://ftp.ilga.gov/ILCS/Ch%200765/Act%200160/"
BUILD_DATE = date(2026, 9, 27)


def statute_source(**overrides: object) -> Source:
    return Source.model_validate(
        {
            "doc_id": "cicaa",
            "title": "Common Interest Community Association Act",
            "url": LISTING_URL,
            "kind": "statute",
            "authority": "statute",
            "effective_date": "section",
            "ilcs": {"chapter": 765, "act": 160, "articles": ["1"]},
            **overrides,
        }
    )


def ilga(*lines: str, nested: str = "") -> bytes:
    """One ILGA section container in the repository's markup."""
    cells = "<br>".join(
        f'<code>&nbsp;&nbsp;</code><code><font size="2">{html.escape(line)}'
        "</font></code>"
        for line in lines
    )
    return (
        '<!DOCTYPE html><html><body><table width="600"><tr><td><div>'
        f"{cells}{nested}</div></td></tr></table></body></html>"
    ).encode()


def listing(entries: dict[str, int]) -> bytes:
    rows = "".join(
        f'11/24/2025  1:21 PM {size:>12} <A HREF="/ILCS/Ch%200765/Act%200160/'
        f'{name.replace(" ", "%20")}">{name}</A><br>'
        for name, size in entries.items()
    )
    return f"<html><body><pre>{rows}</pre></body></html>".encode()


SEC_1_30 = ilga(
    "(765 ILCS 160/1-30)",
    "Sec. 1-30. Board duties and obligations; records.",
    "(a) The board shall meet at least 4 times annually.",
    "(Source: P.A. 102-921, eff. 5-27-22; 103-486, eff. 1-1-24.)",
)
TWO_VERSIONS = ilga(
    "(765 ILCS 160/1-30)",
    "(Text of Section before amendment by P.A. 104-797 )",
    "Sec. 1-30. Board duties and obligations; records.",
    "(a) The board shall meet at least 4 times annually.",
    "(Source: P.A. 102-921, eff. 5-27-22; 103-486, eff. 1-1-24.)",
    "(Text of Section after amendment by P.A. 104-797 )",
    "Sec. 1-30. Board duties and obligations; records.",
    "(a) The board shall meet at least 6 times annually.",
    "(Source: P.A. 103-486, eff. 1-1-24; 104-797, eff. 1-1-27.)",
)
PARALLEL = ilga(
    "(765 ILCS 160/1-45)",
    "(Text of Section from P.A. 100-292)",
    "Sec. 1-45. Finances.",
    "(a) Current text.",
    "(Source: P.A. 100-292, eff. 1-1-18.)",
    "(Text of Section from P.A. 104-734)",
    "Sec. 1-45. Finances.",
    "(a) One future text. (Source: P.A. 104-734, eff. 1-1-27.)",
    "(Text of Section from P.A. 104-797)",
    "Sec. 1-45. Finances.",
    "(a) Another future text.",
    "(Source: P.A. 104-797, eff. 1-1-27.)",
)
FUTURE_ONLY = ilga(
    "(765 ILCS 160/1-73)",
    "(This Section may contain text from a Public Act with a delayed effective date)",
    "Sec. 1-73. Law enforcement and firefighter vehicles. An association may "
    "not define a marked law enforcement vehicle as a commercial vehicle.",
    "(Source: P.A. 104-580, eff. 1-1-27.)",
)


def test_single_version_section() -> None:
    [version] = versions(SEC_1_30)
    assert version.section == "1-30"
    assert version.caption == "Board duties and obligations; records."
    assert version.lines == ("(a) The board shall meet at least 4 times annually.",)
    assert version.kind == "only"
    assert version.effective == date(2024, 1, 1)


def test_nested_layout_tables_and_run_on_items_split_into_lines() -> None:
    page = ilga(
        "(765 ILCS 160/1-30)",
        "Sec. 1-30. Board duties and obligations; records.",
        "(i) Board records. (1) The board shall maintain these records:",
        nested=(
            '<table width="100%"><tr><td><code>(i) the declaration; (ii) the '
            "bylaws.</code></td></tr></table><br><code>(Source: P.A. 103-486, "
            "eff. 1-1-24.)</code>"
        ),
    )
    [version] = versions(page)
    assert version.lines == (
        "(i) Board records.",
        "(1) The board shall maintain these records:(i) the declaration;",
        "(ii) the bylaws.",
    )


def test_before_version_is_law_until_the_amendment_takes_effect() -> None:
    found = versions(TWO_VERSIONS)
    assert [(v.kind, v.public_act) for v in found] == [
        ("before", "104-797"),
        ("after", "104-797"),
    ]
    warnings: list[str] = []
    [now] = in_force(found, BUILD_DATE, warnings, "1-30")
    assert "4 times" in now.lines[0] and now.effective == date(2024, 1, 1)
    [later] = in_force(found, date(2027, 1, 1), warnings, "1-30")
    assert "6 times" in later.lines[0] and later.effective == date(2027, 1, 1)
    assert warnings == []


def test_before_version_without_its_twin_fails() -> None:
    before = versions(TWO_VERSIONS)[:1]
    with pytest.raises(ValueError, match="no text after P.A. 104-797"):
        in_force(before, BUILD_DATE, [], "1-30")


def test_parallel_versions_keep_only_the_latest_in_force() -> None:
    found = versions(PARALLEL)
    assert [v.public_act for v in found] == ["100-292", "104-734", "104-797"]
    # The source note can trail the last paragraph on the same line.
    assert found[1].lines == ("(a) One future text.",)
    [now] = in_force(found, BUILD_DATE, [], "1-45")
    assert now.public_act == "100-292"
    later = in_force(found, date(2027, 1, 1), [], "1-45")
    assert [v.public_act for v in later] == ["104-734", "104-797"]


def test_future_only_section_is_left_out_with_a_warning() -> None:
    [version] = versions(FUTURE_ONLY)
    # A caption that runs straight into the body is split at its sentence end.
    assert version.caption == "Law enforcement and firefighter vehicles."
    assert version.lines[0].startswith("An association may not define")
    warnings: list[str] = []
    assert in_force([version], BUILD_DATE, warnings, "765 ILCS 160/1-73") == []
    assert warnings == [
        "765 ILCS 160/1-73: only future-effective text (2027-01-01); not ingested"
    ]


def test_repealed_sections_are_skipped() -> None:
    page = ilga(
        "(805 ILCS 105/101.30)",
        "Sec. 101.30. (Repealed).",
        "(Source: P.A. 92-33, eff. 7-1-01.)",
    )
    assert in_force(versions(page), BUILD_DATE, [], "101.30") == []


def test_source_note_dates() -> None:
    note = "P.A. 84-1423; 96-649, eff. 1-1-10; 103-486, eff. 1-1-24"
    assert public_act_dates(note) == {
        (84, 1423): None,
        (96, 649): date(2010, 1, 1),
        (103, 486): date(2024, 1, 1),
    }
    # The latest Public Act dates the section, not the latest date listed.
    assert source_effective("P.A. 97-1090, eff. 8-24-12; 98-1, eff. 1-1-10") == (
        date(2010, 1, 1)
    )
    assert source_effective("P.A. 84-1423") is None
    with pytest.raises(ValueError, match="no Public Act"):
        source_effective("n/a")


def snapshot_files(tmp_path: Path, documents: dict[str, bytes]) -> Snapshot:
    """Write a snapshot the way refresh does, from ``documents``."""
    sequence = "\n".join(documents)
    pages = {
        README_URL: b"It was updated on 11/21/2025 with all Public Acts through "
        b"Public Act 104-433.",
        SEQUENCE_URL: sequence.encode(),
        LISTING_URL: listing({n: len(d) for n, d in documents.items()}),
    }
    source = statute_source()
    pages |= {file_url(source, name): data for name, data in documents.items()}
    [snapshot] = refresh(tmp_path, [source], pages.__getitem__, BUILD_DATE)
    return snapshot


ARTICLE = ilga("(765 ILCS 160/Art. 1 heading) Article 1", "(Source: P.A. 96-1400.)")


def test_refresh_writes_a_hash_checked_snapshot(tmp_path: Path) -> None:
    (tmp_path / "cicaa").mkdir()
    (tmp_path / "cicaa/stale.html").write_text("old")
    documents = {f"{PREFIX}HArt. 1.html": ARTICLE, f"{PREFIX}K1-30.html": SEC_1_30}
    snapshot = snapshot_files(tmp_path, documents)
    assert snapshot.ilga_updated_on == "11/21/2025"
    assert snapshot.through_public_act == "104-433"
    assert snapshot.sequence == tuple(documents)
    assert not (tmp_path / "cicaa/stale.html").exists()
    loaded, read = load_snapshot(tmp_path, statute_source())
    assert loaded == snapshot and read == documents

    (tmp_path / "cicaa" / f"{PREFIX}K1-30.html").write_bytes(TWO_VERSIONS)
    with pytest.raises(ValueError, match="does not match its hash"):
        load_snapshot(tmp_path, statute_source())


def test_a_changed_ilga_listing_fails_the_build(tmp_path: Path) -> None:
    documents = {f"{PREFIX}HArt. 1.html": ARTICLE, f"{PREFIX}K1-30.html": SEC_1_30}
    snapshot = snapshot_files(tmp_path, documents)
    same = listing({n: len(d) for n, d in documents.items()})
    check_current(snapshot, statute_source(), lambda url: same)
    grown = listing({**{n: len(d) for n, d in documents.items()}, "x": 1})
    check_current(snapshot, statute_source(), lambda url: grown)  # not ours
    changed = listing({f"{PREFIX}HArt. 1.html": 1, f"{PREFIX}K1-30.html": 2})
    with pytest.raises(ValueError, match="refresh-statutes"):
        check_current(snapshot, statute_source(), lambda url: changed)


def test_listing_parsing() -> None:
    entries = parse_listing(listing({f"{PREFIX}HArt. 1.html": 982}), PREFIX)
    assert [(e.name, e.size, e.stamp) for e in entries] == [
        (f"{PREFIX}HArt. 1.html", 982, "11/24/2025 1:21 PM")
    ]
    with pytest.raises(ValueError, match="no 076501600 documents"):
        parse_listing(b"<pre></pre>", PREFIX)


def test_chunks_carry_citation_url_and_continuation_context(tmp_path: Path) -> None:
    long_lines = [f"({chr(97 + i)}) " + "The board shall act. " * 40 for i in range(8)]
    long = ilga(
        "(765 ILCS 160/1-25)",
        "Sec. 1-25. Board of managers.",
        *long_lines,
        "(Source: P.A. 98-1042, eff. 1-1-15; 99-41, eff. 7-14-15.)",
    )
    documents = {
        f"{PREFIX}HArt. 1.html": ARTICLE,
        f"{PREFIX}K1-25.html": long,
        f"{PREFIX}K1-30.html": TWO_VERSIONS,
        f"{PREFIX}K1-73.html": FUTURE_ONLY,
    }
    snapshot = snapshot_files(tmp_path, documents)
    warnings: list[str] = []
    chunks = statute_chunks(statute_source(), snapshot, documents, BUILD_DATE, warnings)
    assert [c.id for c in chunks] == [
        "cicaa-1-25-a",
        "cicaa-1-25-b",
        "cicaa-1-25-c",
        "cicaa-1-30",
    ]
    assert warnings == [
        "765 ILCS 160/1-73: only future-effective text (2027-01-01); not ingested"
    ]
    first, second, _, records = chunks
    assert first.citation_label == "765 ILCS 160/1-25 (Board of managers)"
    assert first.text_clean.startswith(
        "Common Interest Community Association Act (765 ILCS 160), Article 1: "
        "765 ILCS 160/1-25 (Board of managers), as amended by P.A. 99-41, "
        "effective 2015-07-14.\n"
        "(765 ILCS 160/1-25)\nSec. 1-25. Board of managers."
    )
    # Every continuation names the Act, Article, section and date.
    assert second.text_clean.startswith(
        "Common Interest Community Association Act (765 ILCS 160), Article 1: "
        "765 ILCS 160/1-25 (Board of managers), as amended by P.A. 99-41, "
        "effective 2015-07-14. "
        "(continued)\n"
    )
    assert all(len(c.text_clean) <= MAX_CHARS for c in chunks)
    assert records.effective_date == date(2024, 1, 1)
    assert "4 times" in records.text_clean and "6 times" not in records.text_clean
    assert records.authority is Authority.statute
    assert records.page_start is None and records.heading_path[1] == "Article 1"
    assert citation_url(records) == (
        "https://ftp.ilga.gov/ILCS/Ch%200765/Act%200160/076501600K1-30.html"
    )
    # On 2027-01-01 the amended text replaces it; nothing future is ingested.
    [amended] = [
        c
        for c in statute_chunks(
            statute_source(), snapshot, documents, date(2027, 1, 1), []
        )
        if c.id.startswith("cicaa-1-30")
    ]
    assert "6 times" in amended.text_clean


def test_articles_outside_scope_are_skipped(tmp_path: Path) -> None:
    article_99 = ilga(
        "(765 ILCS 160/Art. 99 heading) Article 99", "(Source: P.A. 96-1400.)"
    )
    effective = ilga(
        "(765 ILCS 160/99-5)",
        "Sec. 99-5. Effective date.",
        "This Act takes effect upon becoming law.",
        "(Source: P.A. 96-1400, eff. 7-29-10.)",
    )
    documents = {
        f"{PREFIX}HArt. 1.html": ARTICLE,
        f"{PREFIX}K1-30.html": SEC_1_30,
        f"{PREFIX}HArt. 99.html": article_99,
        f"{PREFIX}K99-5.html": effective,
    }
    snapshot = snapshot_files(tmp_path, documents)
    chunks = statute_chunks(statute_source(), snapshot, documents, BUILD_DATE, [])
    assert [c.id for c in chunks] == ["cicaa-1-30"]
    whole = statute_source(ilcs={"chapter": 765, "act": 160})
    chunks = statute_chunks(whole, snapshot, documents, BUILD_DATE, [])
    assert [c.id for c in chunks] == ["cicaa-1-30", "cicaa-99-5"]


def test_build_reads_the_snapshot_and_fetches_only_the_listing(
    tmp_path: Path,
) -> None:
    documents = {f"{PREFIX}HArt. 1.html": ARTICLE, f"{PREFIX}K1-30.html": SEC_1_30}
    snapshot_files(tmp_path / "statutes", documents)
    registry = tmp_path / "sources.yaml"
    registry.write_text(yaml.safe_dump([statute_source().model_dump(mode="json")]))
    calls: list[str] = []
    page = listing({n: len(d) for n, d in documents.items()})

    def fetch(url: str) -> bytes:
        calls.append(url)
        return page

    # CICAA without the applicability note's evidence chunks fails the build.
    with pytest.raises(ValueError, match="amendments-committee-intro is missing"):
        build(tmp_path / "out", registry, fetcher=fetch, crawl_date=BUILD_DATE)
    assert not (tmp_path / "out/corpus.json").exists()

    # Another Act has no note, so the same snapshot builds.
    other = tmp_path / "statutes/other-act"
    (tmp_path / "statutes/cicaa").rename(other)
    manifest = json.loads((other / "manifest.json").read_text())
    (other / "manifest.json").write_text(
        json.dumps({**manifest, "doc_id": "other-act"})
    )
    registry.write_text(
        yaml.safe_dump([statute_source(doc_id="other-act").model_dump(mode="json")])
    )
    calls.clear()
    corpus = build(tmp_path / "out", registry, fetcher=fetch, crawl_date=BUILD_DATE)
    assert calls == [LISTING_URL]
    assert [c.id for c in corpus.chunks] == ["other-act-1-30"]
    manifest_bytes = (other / "manifest.json").read_bytes()
    assert corpus.manifest.source_hashes == {
        "other-act": hashlib.sha256(manifest_bytes).hexdigest()
    }


def test_statute_source_validation() -> None:
    assert statute_source().ilcs is not None
    with pytest.raises(ValidationError, match="statute authority"):
        statute_source(authority="governing")
    with pytest.raises(ValidationError, match="dated per section"):
        statute_source(effective_date="2024-01-01")
    with pytest.raises(ValidationError, match="url must be"):
        statute_source(url="https://ftp.ilga.gov/ILCS/")
    with pytest.raises(ValidationError, match="name their ilcs location"):
        statute_source(ilcs=None)
    with pytest.raises(ValidationError, match="statute authority"):
        Source.model_validate(
            {
                "doc_id": "bylaws",
                "title": "Bylaws",
                "url": "https://example.com/b.pdf",
                "kind": "pdf",
                "authority": "statute",
                "effective_date": "2001-01-01",
            }
        )
    with pytest.raises(ValueError, match="carry their own dates"):
        statute_source().effective(BUILD_DATE)


def test_registry_statutes() -> None:
    statutes = {
        s.doc_id: s for s in load_sources(ROOT / "sources.yaml") if s.kind == "statute"
    }
    assert statutes.keys() == {"cicaa", "nfp-act"}
    assert statutes["cicaa"].ilcs is not None
    assert statutes["cicaa"].ilcs.articles == ("1",)
    assert statutes["nfp-act"].ilcs is not None
    assert statutes["nfp-act"].ilcs.articles == ()
    # The outdated 2022 IDFPR PDF is never a source.
    urls = [s.url for s in load_sources(ROOT / "sources.yaml")]
    assert not any("idfpr" in url.lower() for url in urls)


def test_committed_snapshot_is_current_law() -> None:
    """The committed ILGA snapshot builds, and nothing in it is future law."""
    for source in load_sources(ROOT / "sources.yaml"):
        if source.kind != "statute":
            continue
        snapshot, documents = load_snapshot(ROOT / "statutes", source)
        warnings: list[str] = []
        chunks = statute_chunks(source, snapshot, documents, BUILD_DATE, warnings)
        assert chunks and warnings == []
        assert all(
            c.effective_date is None or c.effective_date <= BUILD_DATE for c in chunks
        )
        assert all(len(c.text_clean) <= MAX_CHARS for c in chunks)


def test_crawl_delay_spaces_requests_per_host(tmp_path: Path) -> None:
    now = [100.0]
    pauses: list[float] = []

    def pause(seconds: float) -> None:
        pauses.append(seconds)
        now[0] += seconds

    fetch = Fetcher(
        tmp_path,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x")),
        clock=lambda: now[0],
        pause=pause,
    )
    fetch.fresh("https://ftp.ilga.gov/a")
    now[0] += 3
    fetch.fresh("https://ftp.ilga.gov/b")
    fetch.fresh("https://example.com/c")
    assert pauses == [7.0]
    # Cached reads are not requests.
    fetch("https://ftp.ilga.gov/d")
    fetch("https://ftp.ilga.gov/d")
    assert pauses == [7.0, 10.0]
    assert not list(tmp_path.glob("*.tmp"))


def test_tls_context_adds_the_pinned_intermediate_without_partial_chains() -> None:
    context = tls_context()
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname
    assert not context.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    # get_ca_certs() lists subjects as nested (name, value) pairs.
    subjects = str([cert["subject"] for cert in context.get_ca_certs()])
    assert "Sectigo Public Server Authentication CA OV R40" in subjects
    assert "Sectigo Public Server Authentication Root R46" in subjects


def test_snapshot_manifest_is_reviewable_json() -> None:
    manifest = json.loads((ROOT / "statutes/cicaa/manifest.json").read_text())
    assert manifest["listing_url"] == LISTING_URL
    assert manifest["through_public_act"]


def test_a_list_lead_in_stays_with_its_first_item() -> None:
    body = "(a) " + "x" * 20
    lead = "(b) Keep these:"
    item = "(i) first item"
    # Without the rule the lead-in would end the first part (40 of 40 chars).
    assert _pack([body, lead, item], 40) == [[body], [lead, item]]
    # A lead-in that is the whole part, or can't fit with its item, stays.
    assert _pack([lead, item], 20) == [[lead], [item]]
    assert _pack([body, lead, "(i) " + "y" * 30], 40) == [
        [body, lead],
        ["(i) " + "y" * 30],
    ]

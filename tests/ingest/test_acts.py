"""Public Acts not yet compiled into ILCS (ingest.acts)."""

import hashlib
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from hoa_qa.ingest.acts import (
    ActSnapshot,
    act_chunks,
    amendments,
    compiled,
    load_act,
    paragraphs,
    snapshot_act,
)
from hoa_qa.ingest.sources import Source, load_sources
from hoa_qa.ingest.statutes import Snapshot

ROOT = Path(__file__).parents[2]
REGISTRY = {s.doc_id: s for s in load_sources(ROOT / "sources.yaml")}
INDENT = "<code>&nbsp;&nbsp;&nbsp;&nbsp;</code>"


def line(html: str, *, opens: bool = False) -> str:
    """One printed line, as ILGA's Public Act pages lay it out."""
    return f'<tr><td class="xsl">{INDENT if opens else ""}{html}</td></tr>'


PAGE = "".join(
    [
        "<table>",
        line("<code>AN ACT concerning civil law.</code>", opens=True),
        line("<code>Section 5. The Common Interest Community Act</code>", opens=True),
        line("<code>is amended by changing Sections 1-35 and 1-45:</code>"),
        line("<code>(765 ILCS 160/1-35)</code>", opens=True),
        line("<code>Sec. 1-35. </code><code>Member powers. </code>", opens=True),
        line("<code>(a) Leases are covered. </code>", opens=True),
        line("<code>(d) On resale, the board shall provide: </code>", opens=True),
        line("<code>(1) A copy of the declaration. </code>", opens=True),
        line(
            "<code>(2) A statement of </code><s><code>all </code></s>"
            "<code>liens. </code>",
            opens=True,
        ),
        line("<u><code>(3) A copy of the collection </code></u>", opens=True),
        line("<u><code>policy.</code></u><code> </code>"),
        line("<code>The officer shall furnish it in 30 days. </code>", opens=True),
        line("<code>(Source: P.A. 98-842, eff. 1-1-15</code><code>.)</code>"),
        line("<code>(765 ILCS 160/1-45)</code>", opens=True),
        line("<code>Sec. 1-45. </code><code>Finances. </code>", opens=True),
        line("<code>(h) Fees are capped unless: </code>", opens=True),
        line("<code>(i) a contract allows them. </code>", opens=True),
        line("<u><code>(i) The board must adopt a collection </code></u>", opens=True),
        line("<u><code>policy.</code></u>"),
        line("<code>(Source: P.A. 100-292, eff. 1-1-18.)</code>"),
        line("<code>Section 10. The Condominium Property Act is </code>", opens=True),
        line("<code>(765 ILCS 605/18.4)</code>", opens=True),
        line("<code>Sec. 18.4. </code><code>Powers. </code>", opens=True),
        line("<u><code>(t) A condominium policy.</code></u>", opens=True),
        line("<code>(Source: P.A. 102-976.)</code>"),
        "</table>",
    ]
).encode()


def test_paragraphs_drop_deleted_text_and_mark_new_text() -> None:
    found = {p.text: p for p in paragraphs(PAGE)}
    assert "(2) A statement of liens." in found  # "all" was struck.
    assert found["(3) A copy of the collection policy."].all_inserted
    assert not found["(1) A copy of the declaration."].inserted
    # The source note is its own paragraph, never part of the last one.
    assert "The officer shall furnish it in 30 days." in found


def test_only_changed_subsections_of_the_amended_act_are_kept() -> None:
    warnings: list[str] = []
    changes = amendments(PAGE, 765, 160, warnings, "test")
    assert [(c.section, c.subsection, c.changed) for c in changes] == [
        ("1-35", "d", "(d)(3)"),
        # "(i)" after "(h) ... unless:" is (h)'s roman item; the next "(i)",
        # after a sibling item, is the letter (i): a new subsection.
        ("1-45", "i", "(i)"),
    ]
    resale = changes[0].lines
    assert resale[0].startswith("(d) On resale")  # The lead-in stays.
    assert resale[-1] == "The officer shall furnish it in 30 days."
    assert not warnings
    # 765 ILCS 605 (the Condominium Property Act) is another Act: skipped.
    assert all(c.section != "18.4" for c in changes)


def test_act_chunks_are_dated_by_the_act_and_say_what_is_new() -> None:
    source = REGISTRY["pa-104-0734"]
    chunks = act_chunks(source, REGISTRY["cicaa"], PAGE, [])
    assert [c.id for c in chunks] == ["pa-104-0734-1-35-d", "pa-104-0734-1-45-i"]
    first = chunks[0]
    assert first.effective_date == date(2027, 1, 1)
    assert first.authority.value == "statute"
    assert first.citation_label == (
        "765 ILCS 160/1-35(d) (Member powers), as amended by P.A. 104-734, "
        "effective 2027-01-01"
    )
    assert first.text_clean.splitlines()[0].endswith(
        "effective 2027-01-01. New or changed text: (d)(3)."
    )
    assert "Source:" not in first.text_clean


def test_the_committed_snapshot_parses() -> None:
    source = REGISTRY["pa-104-0734"]
    page = load_act(ROOT / "statutes", source)
    warnings: list[str] = []
    chunks = act_chunks(source, REGISTRY["cicaa"], page, warnings)
    assert not warnings
    assert {c.id.rsplit("-", 1)[0] for c in chunks} == {
        "pa-104-0734-1-35-d",
        "pa-104-0734-1-45-j",
    }
    text = " ".join(c.text_clean for c in chunks)
    assert "(8) A copy of the collection policy adopted under subsection (j)" in text
    assert "shall not take legal action to collect common expenses" in text
    assert "New or changed text: (j)." in text
    assert "Condominium" not in text  # Section 10 amends another Act.


def test_snapshot_is_hash_checked(tmp_path: Path) -> None:
    source = REGISTRY["pa-104-0734"]
    snapshot_act(tmp_path, source, lambda _: PAGE, date(2026, 10, 15))
    assert load_act(tmp_path, source) == PAGE
    manifest = ActSnapshot.model_validate_json(
        (tmp_path / source.doc_id / "manifest.json").read_text()
    )
    assert manifest.sha256 == hashlib.sha256(PAGE).hexdigest()
    (tmp_path / source.doc_id / "act.html").write_bytes(PAGE + b" ")
    with pytest.raises(ValueError, match="does not match its hash"):
        load_act(tmp_path, source)


def _compiled_through(act: str) -> Snapshot:
    return Snapshot(
        doc_id="cicaa",
        listing_url=REGISTRY["cicaa"].url,
        fetched_on=date(2027, 3, 1),
        ilga_updated_on="2/1/2027",
        through_public_act=act,
        listing=(),
        sequence=(),
        sha256={},
    )


def test_a_compiled_act_is_no_longer_needed() -> None:
    source = REGISTRY["pa-104-0734"]
    assert not compiled(source, _compiled_through("104-433"))
    assert compiled(source, _compiled_through("104-734"))
    assert compiled(source, _compiled_through("105-12"))


def test_public_act_sources_are_validated() -> None:
    base = REGISTRY["pa-104-0734"].model_dump()
    with pytest.raises(ValidationError, match="public act url"):
        Source.model_validate(base | {"url": "https://example.com/act"})
    with pytest.raises(ValidationError, match="name public_act and amends"):
        Source.model_validate(base | {"amends": None})
    with pytest.raises(ValidationError, match="statute authority"):
        Source.model_validate(base | {"authority": "rules"})

"""Regressions for the PR #9 review: real-corpus sectioning, labels, and guards."""

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hoa_qa.ingest import build
from hoa_qa.ingest.cleanup import clean, is_ocr_noise
from hoa_qa.ingest.extract import Page, blog_text, printed_pages
from hoa_qa.ingest.pii import check_pii
from hoa_qa.ingest.section import citation, sections
from hoa_qa.ingest.sources import Source, load_sources

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[2]
REGISTRY = {s.doc_id: s for s in load_sources(ROOT / "sources.yaml")}


def keys(doc_id: str, text: str, warnings: list[str] | None = None) -> list[str]:
    return [s.key for s in sections(REGISTRY[doc_id], [Page(text, 9)], warnings)]


# B2: the Declaration's appended Rules 2023 exhibits never carry governing authority.
def test_declaration_drops_appended_bylaws_and_rules_exhibits() -> None:
    excluded = set(REGISTRY["declaration"].exclude_pages)
    assert set(range(68, 84)) <= excluded
    assert not excluded & {59, *range(60, 66), 67}


# B3a: Rules use SECTION N with lettered subsections, not Articles.
@pytest.mark.parametrize("doc_id", ["rules-2023", "rules-2016"])
def test_rules_lettered_subsections(doc_id: str) -> None:
    text = (
        "SECTION 3. ENFORCEMENT PROVISIONS FOR VIOLATIONS:\n"
        "A. Intent\nThe purpose of the enforcement section.\n"
        "B. Fines\nThe fining structure shall be as follows:\n"
        "2nd offense - $50.00 fine\n3rd offense - $100.00 fine\n"
        "C. Notice and Hearing Procedure\n1. Any complaint.\n"
        "B. Not a new subsection out of order.\n"
    )
    result = sections(REGISTRY[doc_id], [Page(text, 9)])
    assert [s.key for s in result] == ["3.A", "3.B", "3.C"]
    fines = result[1]
    assert "$50.00" in fines.text and "$100.00" in fines.text
    title = REGISTRY[doc_id].title
    assert (
        citation(REGISTRY[doc_id], fines) == f"{title}, Section 3.B (Fines), PDF p. 9"
    )
    assert "Not a new subsection" in result[2].text


def test_rules_section_ignores_wrapped_cross_reference() -> None:
    text = "SECTION 2- RULES\n2.1 Use.\nSection 3.4 of the Declaration applies.\n"
    assert keys("rules-2023", text) == ["2.1"]


# B3b: one modal offset; never a label mixing printed and PDF numbers.
def test_printed_offset_is_modal_and_needs_agreement() -> None:
    pages = [Page("x", n, str(n - 5)) for n in range(16, 20)]
    pages.insert(2, Page("x", 18, "99"))  # One disagreeing OCR footer.
    fixed = printed_pages([Page("x", 6), Page("x", 7), *pages], "declaration")
    assert [(p.number, p.printed) for p in fixed][:3] == [
        (6, "1"),
        (7, "2"),
        (16, "11"),
    ]
    assert all(p.printed == str(p.number - 5) for p in fixed if p.number)
    two = [Page("x", 1, "1"), Page("x", 2, "2"), Page("x", 3)]
    assert all(p.printed is None for p in printed_pages(two, "declaration"))


# B3c/d: the next-expected reading of a dotless or mis-dotted number wins.
@pytest.mark.parametrize(
    ("doc_id", "text", "expected"),
    [
        ("declaration", "Article 8\n8.7 Quorum.\n838 Date of Commencement.", "8.8"),
        (
            "declaration",
            "Article 7\n7.2 Directors.\n7.2.1 Terms.\n7.22 Terms.",
            "7.2.2",
        ),
        (
            "declaration",
            "Article 7\n7.7 Powers.\n7.7.1 A.\n7.7.2 B.\n7.7.3 Taxes.\n7.74 Grounds.",
            "7.7.4",
        ),
        (
            "declaration",
            "Article 12\n12.3 Turnover.\n12.3.1 A.\n12.3.2 B.\n12.3.3 C.\n12.34 D.",
            "12.3.4",
        ),
        ("declaration", "Article 1\n1.8 Owner.\n1.10 Unit.", "1.10"),
        ("rules-2016", "SECTION 2- RULES\n2.2 Pets.\n23 Trash.", "2.3"),
        (
            "rules-2016",
            "SECTION 2- RULES\n2.9 Lights.\n2.10 Parking.\n211 Signs.",
            "2.11",
        ),
    ],
)
def test_dotless_headings_take_next_expected_key(
    doc_id: str, text: str, expected: str
) -> None:
    assert keys(doc_id, text)[-1] == expected


def test_non_monotonic_heading_logged_not_emitted() -> None:
    warnings: list[str] = []
    text = "Article 8\n8.2 Deposit.\n8.9 Duties.\n117.27 FEET\n8.3 Purpose."
    assert keys("declaration", text, warnings) == ["8.2", "8.3"]
    assert warnings == [
        "declaration: non-monotonic heading '8.9' after 8.2; kept as text",
        "declaration: non-monotonic heading '117.27' after 8.2; kept as text",
    ]


# B3e: an in-prose exhibit reference is not a heading.
def test_exhibit_reference_in_prose_is_not_heading() -> None:
    text = (
        "Article 2\n2.1 Property. The land described on\n"
        "Exhibit B (hereafter the Plat) is subject to this Declaration.\n"
        "EXHItsIT B\nLegal description."
    )
    result = sections(REGISTRY["declaration"], [Page(text, 9)])
    assert [s.key for s in result] == ["2.1", "exhibit-b"]
    assert "hereafter" in result[0].text
    assert (
        citation(REGISTRY["declaration"], result[1])
        == "Declaration, Exhibit B, PDF p. 9"
    )


# B3f: an Article heading OCR merged into the end of the previous paragraph.
def test_midline_article_heading() -> None:
    text = (
        "Article 12\n12.2 Indemnity. Officers shall be indemnified. "
        "Axticle 13 Aysessments As more fully provided in the Declaration, each "
        "Owner is obligated to pay assessments to the Association."
    )
    result = sections(REGISTRY["bylaws"], [Page(text, 10)])
    assert [s.key for s in result] == ["12.2", "article-13"]
    assert "Aysessments" not in result[0].text
    assert result[1].heading[0].startswith("Axticle 13 Aysessments")


# N1: heading-only sections fold into the next chunk and its heading_path.
def test_heading_only_sections_fold_forward() -> None:
    text = (
        "Article 5\nBuilding and Use Restrictions\n"
        "5.8 Antennas. No antennas.\n59 Fences.\n5.9.1 Height. Four feet."
    )
    result = sections(REGISTRY["declaration"], [Page(text, 17)])
    assert [s.key for s in result] == ["5.8", "5.9.1"]
    assert result[0].heading[0] == "Article 5 Building and Use Restrictions"
    assert result[1].text.startswith("59 Fences.")
    assert result[1].heading[-2:] == ("59 Fences.", "§5.9.1")


# N2: ALL-CAPS lines split only after a finished sentence; form blanks stay whole.
def test_clubhouse_caps_and_form_blanks() -> None:
    text = (
        "FUNCTION TIMING: The function must end and the renter must\n"
        "VACATE THE PREMISES BY 11 PM.\n"
        "POOL ACCESS: The pool is not available to guests.\n"
        "RENTER NAME: ______________________\n"
        "ADDRESS: ______________________\n"
        "DATE: ___________\n"
    )
    result = sections(REGISTRY["clubhouse"], [Page(text, 2)])
    assert [s.key for s in result] == ["1", "2", "3"]
    assert "VACATE" in result[0].text
    assert "ADDRESS" in result[2].text and "DATE" in result[2].text


# N3: clean display titles and labels.
def test_labels_and_titles() -> None:
    intro = sections(REGISTRY["declaration"], [Page("Recorded cover page.", 1)])
    assert citation(REGISTRY["declaration"], intro[0]) == "Declaration, PDF p. 1"
    for s in REGISTRY.values():
        assert s.title == s.title.strip() and "2O" not in s.title
    assert REGISTRY["minutes-2024-12-20"].title == "Minutes December 20, 2024"
    assert "december-2o-2024" in str(REGISTRY["minutes-2024-12-20"].url)


# N4: character repair only; meaning and numbering never change.
@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        (
            "Owners shall not park commercial vehicles on the street overnight.",
            "Owners shall park commercial vehicles on the street overnight.",
        ),
        (
            "Fines are imposed as provided in Section 8.2(d) of the Declaration.",
            "Fines are imposed as provided in Section 8.2(b) of the Declaration.",
        ),
        (
            "Notice is due thirty days before the annual meeting of Members.",
            "Notice is due thirteen days before the annual meeting of Members.",
        ),
        (
            "The Board may levy a special assessment after a vote of Owners.",
            "The Board may levy a special assessment only after a vote of Owners.",
        ),
        (
            "Tbe Board rnay levy a special assessrnent after a vote of Owners.",
            "The Board may levy a special assessment following an Owner vote.",
        ),
    ],
)
def test_cleanup_guard_falls_back(raw: str, repaired: str) -> None:
    fallbacks: list[str] = []
    assert clean(raw, "c", lambda _: repaired, fallbacks) == raw
    assert fallbacks == ["c"]


def test_cleanup_accepts_character_repair() -> None:
    fallbacks: list[str] = []
    raw = "Tbe Board may levy a special assessrnent after a vote of Owners."
    fixed = "The Board may levy a special assessment after a vote of Owners."
    assert clean(raw, "c", lambda _: fixed, fallbacks) == fixed
    assert not fallbacks


# Wave-3 review: the protected-word check is symmetric, with quantifiers and
# money words protected, and only pure OCR noise may be deleted.
BASE = "The Board may levy a special assessment after a vote of Owners."


@pytest.mark.parametrize(
    ("raw", "repaired"),
    [
        # A protected word appearing is as bad as one disappearing.
        (
            "Owners must now register every vehicle with the office.",
            "Owners must not register every vehicle with the office.",
        ),
        (
            "Pets are allowed, so leashes are required on common areas.",
            "Pets are allowed, no leashes are required on common areas.",
        ),
        # Quantifiers.
        (
            "All Owners may use the pool subject to these rules and fees.",
            "Any Owners may use the pool subject to these rules and fees.",
        ),
        (
            "Each Owner may bring guests subject to these rules and fees.",
            "Some Owner may bring guests subject to these rules and fees.",
        ),
        (
            "The fee applies to every rental of the clubhouse on weekends.",
            "The fee applies to any rental of the clubhouse on weekends.",
        ),
        # Money words.
        (
            "The deposit is three hundred dollars payable before the event.",
            "The deposit is three hundred cents payable before the event.",
        ),
        # A garbled protected word can no longer be "repaired" into one.
        (
            "Tbe Board rnay levy a special assessrnent after a vote of Owners.",
            "The Board may levy a special assessment after a vote of Owners.",
        ),
        # Deleting a real word, even a one-letter one, is not noise removal.
        (
            "The Board may levy a special assessment after a vote of all Owners.",
            "The Board may levy a special assessment after vote of all Owners.",
        ),
        (
            "The Board may levy a special assessment after a vote; I agree.",
            "The Board may levy a special assessment after a vote; agree.",
        ),
        (
            "The Board may levy a special assessment after a formal vote of Owners.",
            BASE,
        ),
        # Inserting even pure noise is still an insertion.
        (BASE, "The Board may levy a special ~ assessment after a vote of Owners."),
        # Digits are never noise (and the numeric guard agrees).
        (
            "The Board may levy a special assessment after 2 votes of Owners.",
            "The Board may levy a special assessment after votes of Owners.",
        ),
    ],
)
def test_cleanup_guard_is_symmetric_and_strict(raw: str, repaired: str) -> None:
    fallbacks: list[str] = []
    assert clean(raw, "c", lambda _: repaired, fallbacks) == raw
    assert fallbacks == ["c"]


@pytest.mark.parametrize(
    "raw",
    [
        "The Board may levy ~ a special assessment after a vote of Owners.",
        "The Board may levy a special | assessment after a vote of Owners.",
        "The Board may levy a special assessment -- after a vote of Owners.",
        "The Board may levy a special assessment after a vote of Owners. ‘",
        "The Board may levy a special assessment after a vote of j Owners.",
    ],
)
def test_cleanup_allows_deleting_pure_ocr_noise(raw: str) -> None:
    fallbacks: list[str] = []
    assert clean(raw, "c", lambda _: BASE, fallbacks) == BASE
    assert not fallbacks


def test_is_ocr_noise() -> None:
    for token in ("~", "|", "--", "‘", "•", "j", "Z"):
        assert is_ocr_noise(token), token
    for token in ("a", "A", "I", "i", "5", "no", "8.2", "(d)", "$", "%", "§", "&"):
        assert not is_ocr_noise(token), token


# N5: separator-free phone numbers and more street suffixes.
@pytest.mark.parametrize(
    "text",
    ["(312)555-0189", "3125550189", "312 5550189"]
    + [f"12 Maple {suffix}" for suffix in ("Cir", "Circle", "Trl", "Ter", "Way")]
    + [f"4 Ash {suffix}" for suffix in ("Pl", "Blvd", "Pkwy")],
)
def test_pii_formats_fail(text: str) -> None:
    with pytest.raises(ValueError, match="PII check failed"):
        check_pii(text, "home")


@pytest.mark.parametrize("text", ["(630)229-0092", "6302290092", "1-630-229-0092"])
def test_pii_org_phone_formats_pass(text: str) -> None:
    check_pii(text, "home")


# N6/N8: content vs displayed dates; website pages dated by the build's crawl.
def test_post_dates_and_crawl_date(tmp_path: Path) -> None:
    newsletter = REGISTRY["newsletter-2025-03"]
    assert newsletter.effective_date == date(2025, 3, 1)
    assert newsletter.published_date == date(2025, 5, 29)
    tips = REGISTRY["violation-tips"]
    assert tips.effective_date == tips.published_date == date(2022, 10, 20)
    assert all(
        s.effective_date == "crawl" for s in REGISTRY.values() if s.kind == "web_page"
    )
    page = Source.model_validate(
        {
            "doc_id": "home",
            "title": "Home",
            "url": "https://example.com/",
            "kind": "web_page",
            "authority": "website",
            "effective_date": "crawl",
        }
    )
    registry = tmp_path / "sources.yaml"
    registry.write_text(yaml.safe_dump([page.model_dump(mode="json")]))
    html = b'<div data-aid="BANNER_TEXT_RENDERED">2026 dues $452</div>'
    corpus = build(
        tmp_path / "out",
        registry,
        no_llm=True,
        fetcher=lambda _: html,
        crawl_date=date(2030, 1, 2),
    )
    assert {c.effective_date for c in corpus.chunks} == {date(2030, 1, 2)}
    with pytest.raises(ValidationError):
        Source.model_validate({**page.model_dump(mode="json"), "kind": "pdf"})


# N9: malformed Draft.js blocks fail loudly.
def test_blog_non_dict_block_fails() -> None:
    body = json.dumps({"blocks": ["text", {"text": "ok"}]})
    data = {"post": {"fullContent": body}}
    with pytest.raises(ValueError, match="Draft.js"):
        blog_text("window._BLOG_DATA = " + json.dumps(data) + ";")


# B1 end to end: the budget amount and its adoption vote share one dated chunk.
def test_minutes_build_keeps_budget_with_vote(tmp_path: Path) -> None:
    minutes = REGISTRY["minutes-2022-12-29"]
    registry = tmp_path / "sources.yaml"
    registry.write_text(yaml.safe_dump([minutes.model_dump(mode="json")]))
    blocks = [
        {"text": line} for line in (FIXTURES / "minutes.txt").read_text().split("\n")
    ]
    html = "window._BLOG_DATA = " + json.dumps(
        {"post": {"fullContent": json.dumps({"blocks": blocks})}}
    )
    corpus = build(
        tmp_path / "out", registry, no_llm=True, fetcher=lambda _: html.encode()
    )
    (chunk,) = corpus.chunks
    assert chunk.id == "minutes-2022-12-29-meeting"
    assert "$99" in chunk.text_clean and "Motion approved." in chunk.text_clean
    assert chunk.citation_label == "December 29, 2022 Minutes, Meeting 2022-12-29"
    assert chunk.heading_path == ("December 29, 2022 Minutes", "Meeting 2022-12-29")
    assert chunk.effective_date == date(2022, 12, 29)
    assert chunk.published_date == date(2023, 6, 14)


# Golden: ids and labels from committed real-text excerpts (headings plus one
# body line per section, with each page's resolved printed number).
@pytest.mark.parametrize("doc_id", ["declaration", "rules-2023"])
def test_golden_sections(doc_id: str) -> None:
    pages = json.loads((FIXTURES / "golden_pages.json").read_text())[doc_id]
    golden = json.loads((FIXTURES / "golden_sections.json").read_text())[doc_id]
    warnings: list[str] = []
    result = sections(
        REGISTRY[doc_id],
        [Page(p["text"], p["number"], p["printed"]) for p in pages],
        warnings,
    )
    actual = [[f"{doc_id}-{s.key}", citation(REGISTRY[doc_id], s)] for s in result]
    assert actual == golden
    assert not warnings

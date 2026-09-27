"""Offline tests against small public-source excerpts and synthetic edge cases."""

import hashlib
import json
from pathlib import Path

import httpx
import pymupdf
import pytest
import yaml
from pydantic import ValidationError

from hoa_qa.ingest import build
from hoa_qa.ingest.cleanup import clean, normalize, numeric_tokens
from hoa_qa.ingest.extract import Page, blog_text, extract, web_pages
from hoa_qa.ingest.fetch import Fetcher
from hoa_qa.ingest.pii import check_pii
from hoa_qa.ingest.section import citation, parts, sections
from hoa_qa.ingest.sources import Source, load_sources
from hoa_qa.models import load_corpus

FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parents[2]


def source(doc_id: str = "bylaws", **overrides: object) -> Source:
    return Source.model_validate(
        {
            "doc_id": doc_id,
            "title": doc_id.title(),
            "url": f"https://example.com/{doc_id}.pdf",
            "kind": "pdf",
            "authority": "governing",
            "effective_date": "2001-01-01",
            **overrides,
        }
    )


def test_blog_draftjs_and_html() -> None:
    text = blog_text((FIXTURES / "post.html").read_text())
    assert text == "V. New Business:\nA. Vote to adopt 2023 budget.\nMotion approved."
    assert "blocks" not in text and "Not the body" not in text
    data = {"post": {"fullContent": "<h2>Topic };</h2><p>$99 &amp; $5<br>Next</p>"}}
    assert blog_text("window._BLOG_DATA = " + json.dumps(data) + "; trailing;") == (
        "Topic };\n$99 & $5\nNext"
    )


@pytest.mark.parametrize(
    "html",
    [
        "<html>empty</html>",
        "window._BLOG_DATA = {};",
        "window._BLOG_DATA = {oops;",
        'window._BLOG_DATA = {"post":{"fullContent":""}};',
    ],
)
def test_blog_missing_body_fails(html: str) -> None:
    with pytest.raises(ValueError):
        blog_text(html)


@pytest.mark.parametrize(
    ("doc_id", "fixture", "expected"),
    [
        ("bylaws", "bylaws.txt", ["article-3", "3.3", "3.4", "article-4", "4.1"]),
        ("rules-2023", "rules.txt", ["article-2", "2.8", "2.9", "2.10"]),
    ],
)
def test_structured_sections(doc_id: str, fixture: str, expected: list[str]) -> None:
    pages = [Page((FIXTURES / fixture).read_text(), 2, "1")]
    result = sections(source(doc_id), pages)
    assert [s.key for s in result] == expected
    assert result == sections(source(doc_id), pages)
    assert all(s.heading for s in result)
    assert "Holiday" not in result[1].text


def test_minutes_motion_and_outcome_stay_together() -> None:
    s = source("minutes-2022-12-29", kind="blog_post", authority="board_decision")
    result = sections(s, [Page((FIXTURES / "minutes.txt").read_text())])
    motion = next(p for p in result if "Vote to adopt" in p.text)
    assert "Motion approved." in motion.text
    assert "Adjournment" not in motion.text
    assert len(result) == 4
    assert parts(motion, keep_together=True) == [motion]
    unresolved = sections(
        s, [Page("lV. Old Business:\nA. Bid received.\nVl l: Adjournment:")]
    )
    assert len(unresolved) == 2
    assert "approved" not in unresolved[0].text


@pytest.mark.parametrize("replacement", ["$100", "$99.00", "99", "$9.9"])
def test_numeric_mismatch_fallback(replacement: str) -> None:
    fallbacks: list[str] = []
    raw = "Dues are $99 under Section 8.4 on January 1, 2002."
    assert clean(raw, "dues", lambda t: t.replace("$99", replacement), fallbacks) == raw
    assert fallbacks == ["dues"]


def test_numeric_multisets_and_dates() -> None:
    assert numeric_tokens("1 1 2") != numeric_tokens("1 2 2")
    assert numeric_tokens("1/2/2023") != numeric_tokens("2/1/2023")
    assert numeric_tokens("January 1") != numeric_tokens("February 1")
    assert numeric_tokens("Section 2.9") != numeric_tokens("Section 9.2")
    fallbacks: list[str] = []
    assert clean("Ducs $99", "x", lambda _: "Dues $99", fallbacks) == "Dues $99"
    assert not fallbacks
    assert clean(" a\n b ", "x", None, fallbacks) == "a b"


@pytest.mark.parametrize(
    "text",
    [
        "2799 Oakmont Dr, (630) 229-0092, lakewoodcreek@comcast.net",
        "630-229-0092; 2799 Oakmont Drive",
    ],
)
def test_pii_organizational_pass(text: str) -> None:
    check_pii(text, "home")


@pytest.mark.parametrize(
    "text",
    [
        "Call 312-555-0189",
        "123 Private Street",
        "resident@example.com",
        "Visit 123 North Private Avenue",
        "(312) 555-0189",
    ],
)
def test_pii_unexpected_fails(text: str) -> None:
    with pytest.raises(ValueError, match="PII check failed"):
        check_pii(text, "home")


def test_pii_scoped_allowlist() -> None:
    check_pii("866-648-8482", "faq")
    with pytest.raises(ValueError):
        check_pii("866-648-8482", "bylaws")
    check_pii("30 RIGHT OF WAY", "declaration")


def test_pdf_page_indices_and_printed_labels() -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 80), "Article 8\n8.4 Assessments.\nAnnual cap $326.")
        page.insert_text((300, 800), "23")
        data = pdf.tobytes()
    s = source("declaration")
    pages = extract(s, data)
    assert pages[0].number == 1 and pages[0].printed == "23"
    result = sections(s, pages)
    assert citation(s, result[-1]).endswith("p. 23")
    assert "§8.4" in citation(s, result[-1])
    fallback = sections(s, [Page("8.4 Assessments.", 28)])
    assert citation(s, fallback[0]).endswith("p. 28")


def test_no_false_section_on_wrapped_reference_or_survey() -> None:
    s = source("declaration")
    text = (
        "Article 8\n84 Assessments.\nArticle 8. In the event of default.\n"
        "EXHIBIT B\nSECTION 1, TOWNSHIP 37\n117.27 FEET\n"
    )
    result = sections(s, [Page(text, 28, "23")])
    assert [x.key for x in result] == ["article-8", "8.4", "exhibit-b"]
    assert "In the event" in result[1].text


def test_long_sections_stable_and_bounded() -> None:
    s = source()
    text = "3.4 Suspension.\n" + ("A long provision stays in its own section. " * 200)
    text += "\n4.1 Meetings.\nA separate provision."
    result = sections(s, [Page(text, 2)])
    split = parts(result[0])
    assert [p.key for p in split][:2] == ["3.4-a", "3.4-b"]
    assert max(len(normalize(p.text)) for p in split) <= 3200
    assert normalize(" ".join(p.text for p in split)) == normalize(result[0].text)
    assert all("separate" not in p.text for p in split)


def test_registry() -> None:
    sources = load_sources(ROOT / "sources.yaml")
    assert len(sources) == 25
    assert all(s.authority for s in sources)
    excluded = [s for s in sources if s.exclude]
    assert [s.doc_id for s in excluded] == ["email-answers"]
    assert excluded[0].exclude_reason
    old = next(s for s in sources if s.doc_id == "rules-2016")
    assert old.superseded_by == "rules-2023"
    declaration = next(s for s in sources if s.doc_id == "declaration")
    assert set(range(68, 80)) <= set(declaration.exclude_pages)
    with pytest.raises(ValidationError):
        Source.model_validate({"doc_id": "missing"})
    with pytest.raises(ValidationError):
        source(exclude=True)


def test_web_main_content() -> None:
    html = (
        '<nav>Ignore</nav><div data-aid="BANNER_TEXT_RENDERED">'
        "2026 dues $452</div><footer>Cookies</footer>"
    )
    assert web_pages(html, "home")[0].text == "2026 dues $452"
    html += (
        '<h3 data-aid="FAQ_QUESTION_RENDERED_0">When?</h3>'
        '<div data-aid="FAQ_ANSWER_RENDERED_0">January 1.</div>'
    )
    assert web_pages(html, "faq")[0].text == "When?\nJanuary 1."
    with pytest.raises(ValueError):
        web_pages("<html>Site changed</html>", "home")


def test_end_to_end_no_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = source("minutes-test", kind="blog_post", authority="board_decision")
    excluded = source("excluded", exclude=True, exclude_reason="privacy")
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump([s.model_dump(mode="json"), excluded.model_dump(mode="json")])
    )
    data = (FIXTURES / "post.html").read_bytes()
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        return data

    corpus = build(tmp_path / "out", registry, no_llm=True, fetcher=fetch)
    assert calls == [s.url]
    assert corpus == load_corpus(tmp_path / "out/corpus.json")
    assert corpus.manifest.source_hashes == {s.doc_id: hashlib.sha256(data).hexdigest()}
    assert json.loads(
        (tmp_path / "out/corpus_manifest.json").read_text()
    ) == corpus.manifest.model_dump(mode="json")
    assert all(c.text_clean == normalize(c.text_raw) for c in corpus.chunks)
    again = build(tmp_path / "again", registry, no_llm=True, fetcher=fetch)
    assert corpus.chunks == again.chunks


def test_fetch_cache_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    count = 0

    def request(self: httpx.Client, url: str) -> httpx.Response:
        nonlocal count
        count += 1
        assert "LakewoodCreek" in self.headers["User-Agent"]
        return httpx.Response(
            503 if count == 1 else 200,
            content=b"source",
            request=httpx.Request("GET", url),
        )

    waits: list[int] = []
    monkeypatch.setattr(httpx.Client, "get", request)
    monkeypatch.setattr("hoa_qa.ingest.fetch.time.sleep", waits.append)
    fetch = Fetcher(tmp_path)
    assert fetch("https://example.com/doc") == b"source"
    assert fetch("https://example.com/doc") == b"source"
    assert count == 2 and waits == [1]


def test_build_fallback_logged_and_pii_stops_emission(tmp_path: Path) -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 80), "Article 8\n8.4 Dues are $99.")
        data = pdf.tobytes()
    registry = tmp_path / "sources.yaml"
    registry.write_text(yaml.safe_dump([source("declaration").model_dump(mode="json")]))
    corpus = build(
        tmp_path / "ok",
        registry,
        fetcher=lambda _: data,
        repair=lambda t: t.replace("$99", "$0"),
    )
    assert corpus.manifest.ocr_fallbacks == ("declaration-8.4",)
    assert "$99" in corpus.chunks[-1].text_clean
    with pytest.raises(ValueError, match="PII check failed"):
        build(
            tmp_path / "bad",
            registry,
            fetcher=lambda _: data,
            repair=lambda t: t + " resident@example.com",
        )
    assert not (tmp_path / "bad/corpus.json").exists()


def test_blank_pdf_requires_review_and_exclusion() -> None:
    with pymupdf.open() as pdf:
        pdf.new_page()
        data = pdf.tobytes()
    with pytest.raises(ValueError, match="no text layer"):
        extract(source(), data)
    assert extract(source(exclude_pages=[1]), data) == []


def test_born_digital_skips_repair(tmp_path: Path) -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 80), "SECTION 2 - Rules\n2.9 Holiday Decorations.")
        data = pdf.tobytes()
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            [source("rules-2023", authority="rules").model_dump(mode="json")]
        )
    )

    def forbidden(_: str) -> str:
        raise AssertionError("born-digital text must not call cleanup")

    corpus = build(tmp_path / "out", registry, fetcher=lambda _: data, repair=forbidden)
    assert not corpus.manifest.ocr_fallbacks


def test_cli_no_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from hoa_qa.ingest.__main__ import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        yaml.safe_dump(
            [
                source(
                    "minutes-test", kind="blog_post", authority="board_decision"
                ).model_dump(mode="json")
            ]
        )
    )
    monkeypatch.setattr(
        Fetcher, "__call__", lambda self, url: (FIXTURES / "post.html").read_bytes()
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "ingest",
            "build",
            "--out",
            str(tmp_path / "out"),
            "--sources",
            str(registry),
            "--no-llm",
        ],
    )
    main()
    assert load_corpus(tmp_path / "out/corpus.json").chunks

"""Static checks on the frontend and the Vercel config."""

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]
PUBLIC = ROOT / "public"
DISCLAIMER_WORDS = (
    "Unofficial tool, not legal advice; the governing documents and the Board are "
    "authoritative. Don't include personal information; questions are processed "
    "by third-party AI services."
)


def test_app_js_never_uses_innerhtml() -> None:
    source = (PUBLIC / "app.js").read_text()
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in source


def test_app_js_checks_https_before_href() -> None:
    source = (PUBLIC / "app.js").read_text()
    assert 'startsWith("https://")' in source
    assert 'rel = "noopener noreferrer"' in source


def test_page_has_no_inline_or_external_scripts_or_styles() -> None:
    html = (PUBLIC / "index.html").read_text()
    scripts = [
        '<script src="/app.js" defer>',
        '<script src="/analytics.js" defer>',
        '<script src="/_vercel/insights/script.js" defer>',
    ]
    assert re.findall(r"<script[^>]*>", html) == scripts
    assert re.findall(r"<script[^>]*>\s*</script>", html) == [
        script + "</script>" for script in scripts
    ]
    assert "<style" not in html
    assert not re.search(r"\sstyle\s*=", html)
    assert not re.search(r"\son[a-z]+\s*=", html)  # no inline event handlers
    assert not re.search(r"(https?:)?//", html)  # no external URLs at all


def test_stylesheet_loads_nothing_external() -> None:
    css = (PUBLIC / "styles.css").read_text()
    assert "@import" not in css
    assert not re.search(r"(https?:)?//", css)
    assert not re.search(r"url\(\s*['\"]?(?!data:)", css)


def test_page_accessibility_and_disclaimer() -> None:
    html = (PUBLIC / "index.html").read_text()
    assert '<html lang="en">' in html
    assert '<label for="question"' in html
    assert 'maxlength="500"' in html
    assert re.search(r'id="answer"[^>]*aria-live="polite"[^>]*aria-busy="false"', html)
    for landmark in ("<header", "<main>", "<footer"):
        assert landmark in html
    # Decorative beam layers are hidden from assistive technology.
    beams = re.findall(r'<span class="beam[^"]*"[^>]*>', html)
    assert beams and all('aria-hidden="true"' in beam for beam in beams)
    assert " ".join(DISCLAIMER_WORDS.split()) in " ".join(html.split())
    # The documents link works without JS via the same-origin redirect.
    assert re.search(r'<a id="documents-link" href="/documents"', html)


def test_vercel_json_bundles_corpus_into_entrypoint() -> None:
    config = json.loads((ROOT / "vercel.json").read_text())
    function = config["functions"]["api/index.py"]
    assert function["includeFiles"] == "corpus.json"
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["vercel"]["entrypoint"] == "api.index:app"


def test_vercel_entrypoint_exports_app() -> None:
    source = (ROOT / "api" / "index.py").read_text()
    assert "from hoa_qa.web.app import app" in source


def test_progress_status_is_a_separate_polite_region() -> None:
    # Loading status is announced from its own live region, outside the answer
    # region (which is aria-busy while a request is in flight).
    html = (PUBLIC / "index.html").read_text()
    status = re.search(r'<p id="progress-status"[^>]*>', html)
    assert status
    assert 'role="status"' in status.group(0)
    assert 'aria-live="polite"' in status.group(0)
    assert html.index('id="progress-status"') < html.index('id="answer"')


def test_progress_lines_are_visual_only_and_respect_reduced_motion() -> None:
    source = (PUBLIC / "app.js").read_text()
    assert 'loading.setAttribute("aria-hidden", "true")' in source
    assert "setInterval" not in source  # one timer per stage boundary, cleared on stop
    css = (PUBLIC / "styles.css").read_text()
    reduced = css[css.index("@media (prefers-reduced-motion: reduce)") :]
    assert re.search(r"\.progress-line\s*\{\s*animation:\s*none", reduced)


def test_questions_never_enter_page_urls() -> None:
    html = (PUBLIC / "index.html").read_text()
    # Native submission must not serialize question text into a query either.
    assert '<form method="post" action="/api/ask"' in html
    source = (PUBLIC / "app.js").read_text()
    assert 'fetch("/api/ask", {' in source
    assert 'method: "POST"' in source
    assert "body: JSON.stringify({ question: question })" in source
    assert re.search(
        r'form.addEventListener\("submit", function \(event\) \{'
        r"\s*event.preventDefault\(\);",
        source,
    )
    for api in ("location", "history", "URLSearchParams", "window.va", "sendBeacon"):
        assert not re.search(r"\b" + re.escape(api) + r"\b", source)


def test_local_analytics_script_missing_does_not_affect_assets(harness) -> None:
    assert harness.client.get("/_vercel/insights/script.js").status_code == 404
    for path in ("/", "/app.js", "/analytics.js"):
        assert harness.client.get(path).status_code == 200

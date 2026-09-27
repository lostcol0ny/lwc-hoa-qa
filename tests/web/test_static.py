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
    assert re.findall(r"<script[^>]*>", html) == ['<script src="/app.js" defer>']
    assert re.findall(r"<script[^>]*>\s*</script>", html) == [
        '<script src="/app.js" defer></script>'
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

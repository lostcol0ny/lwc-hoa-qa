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
    assert "<style" not in html
    assert " style=" not in html
    external = r"""(src|href)="(https?:)?//(?!lakewoodcreekhoa\.com/)"""
    assert not re.search(external, html)


def test_page_accessibility_and_disclaimer() -> None:
    html = (PUBLIC / "index.html").read_text()
    assert '<label for="question">' in html
    assert 'maxlength="500"' in html
    assert 'aria-live="polite"' in html
    assert " ".join(DISCLAIMER_WORDS.split()) in " ".join(html.split())


def test_vercel_json_bundles_corpus_into_entrypoint() -> None:
    config = json.loads((ROOT / "vercel.json").read_text())
    function = config["functions"]["api/index.py"]
    assert function["includeFiles"] == "corpus.json"
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["vercel"]["entrypoint"] == "api.index:app"


def test_vercel_entrypoint_exports_app() -> None:
    source = (ROOT / "api" / "index.py").read_text()
    assert "from hoa_qa.web.app import app" in source

import json
from pathlib import Path

import pytest
from web_fakes import QUESTION, Harness

from hoa_qa.web.security import CONTENT_SECURITY_POLICY, SECURITY_HEADERS

ROOT = Path(__file__).parents[2]


def assert_security_headers(headers) -> None:
    for name, value in SECURITY_HEADERS.items():
        assert headers[name] == value
    assert "unsafe-inline" not in headers["content-security-policy"]


@pytest.mark.parametrize("path", ["/", "/app.js", "/styles.css", "/api/health"])
def test_security_headers_on_page_and_api(harness: Harness, path: str) -> None:
    response = harness.client.get(path)
    assert response.status_code == 200
    assert_security_headers(response.headers)


def test_security_headers_on_ask_and_errors(harness: Harness) -> None:
    ok = harness.client.post("/api/ask", json={"question": QUESTION})
    assert_security_headers(ok.headers)
    assert ok.headers["cache-control"] == "no-store"
    invalid = harness.client.post("/api/ask", json={"question": "x" * 501})
    assert_security_headers(invalid.headers)


def test_csp_is_strict() -> None:
    assert CONTENT_SECURITY_POLICY == (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'none'; form-action 'self'"
    )


def test_vercel_json_headers_match_middleware() -> None:
    """public/ is CDN-served on Vercel, so vercel.json must carry the same headers."""
    config = json.loads((ROOT / "vercel.json").read_text())
    rules = [rule for rule in config["headers"] if rule["source"] == "/(.*)"]
    assert len(rules) == 1
    headers = {h["key"]: h["value"] for h in rules[0]["headers"]}
    assert headers == SECURITY_HEADERS


def test_no_cors_wildcard(harness: Harness) -> None:
    preflight = harness.client.options(
        "/api/ask",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "access-control-allow-origin" not in preflight.headers
    response = harness.client.post(
        "/api/ask",
        json={"question": QUESTION},
        headers={"Origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in response.headers
    assert response.status_code == 403
    assert harness.asker.questions == []


def test_same_origin_post_is_allowed(harness: Harness) -> None:
    response = harness.client.post(
        "/api/ask",
        json={"question": QUESTION},
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_null_origin_is_rejected(harness: Harness) -> None:
    response = harness.client.post(
        "/api/ask", json={"question": QUESTION}, headers={"Origin": "null"}
    )
    assert response.status_code == 403


def test_api_docs_are_disabled(harness: Harness) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert harness.client.get(path).status_code == 404

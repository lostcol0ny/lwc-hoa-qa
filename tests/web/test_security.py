import json
from pathlib import Path

import pytest
from web_fakes import QUESTION, Harness, build_harness

from hoa_qa.web.security import (
    CONTENT_SECURITY_POLICY,
    SECURITY_HEADERS,
    parse_origin,
)

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


def post_with_origin(h: Harness, origin: str | None, **headers: str):
    if origin is not None:
        headers["Origin"] = origin
    return h.client.post("/api/ask", json={"question": QUESTION}, headers=headers)


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        ("http://testserver", 200),
        ("http://testserver:80", 200),  # explicit default port
        ("HTTP://TestServer", 200),  # scheme and host are case-insensitive
        ("https://testserver", 403),  # scheme mismatch
        ("https://testserver:80", 403),
        ("http://testserver:8080", 403),  # port mismatch
        ("http://testserver.evil.example", 403),
    ],
)
def test_origin_compares_scheme_host_and_port(
    harness: Harness, origin: str, status: int
) -> None:
    assert post_with_origin(harness, origin).status_code == status


@pytest.mark.parametrize(
    "origin",
    [
        "",
        "null",
        "testserver",
        "http://",
        "http://testserver:abc",
        "http://testserver:99999",
        "http://[::1",
        "http://user@testserver",
        "http://testserver/",
        "http://testserver/path",
        "http://testserver?x=1",
        " http://testserver",
        "javascript:alert(1)",
        "ftp://testserver",
    ],
)
def test_malformed_origin_is_403_not_500(harness: Harness, origin: str) -> None:
    response = post_with_origin(harness, origin)
    assert response.status_code == 403
    assert response.json()["outcome"] == "error"
    assert harness.asker.questions == []


def test_missing_origin_is_allowed(harness: Harness) -> None:
    """Browsers always send Origin on POST; only non-browser clients omit it."""
    assert post_with_origin(harness, None).status_code == 200


@pytest.mark.parametrize(
    ("origin", "proto", "status"),
    [
        ("https://hoa.example", "https", 200),
        ("https://hoa.example:443", "https", 200),  # explicit default port
        ("http://hoa.example", "https", 403),  # downgraded scheme
        ("https://hoa.example", "", 200),  # Vercel serves https by default
        ("http://hoa.example", "http", 200),
        ("https://other.example", "https", 403),
    ],
)
def test_origin_on_vercel_uses_forwarded_proto(
    origin: str, proto: str, status: int
) -> None:
    h = build_harness(on_vercel=True)
    headers = {"host": "hoa.example", "x-forwarded-for": "203.0.113.5"}
    if proto:
        headers["x-forwarded-proto"] = proto
    assert post_with_origin(h, origin, **headers).status_code == status


def test_parse_origin_normalizes_default_ports() -> None:
    assert parse_origin("https://Hoa.Example") == ("https", "hoa.example", 443)
    assert parse_origin("http://hoa.example:8080") == ("http", "hoa.example", 8080)
    assert parse_origin("http://hoa.example:") == ("http", "hoa.example", 80)
    assert parse_origin("null") is None

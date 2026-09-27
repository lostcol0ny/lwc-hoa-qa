"""``GET /documents``: a same-origin, JS-free link to the HOA documents."""

import asyncio

import pytest
from web_fakes import QUESTION, build_harness

from hoa_qa.web.settings import DEFAULT_DOCUMENTS_URL


def test_redirects_to_configured_documents_url() -> None:
    url = "https://docs.example.org/hoa/?tab=governing#top"
    h = build_harness(documents_url=url)
    response = h.client.get("/documents", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == url
    # The app's security headers still apply to the redirect.
    assert "content-security-policy" in response.headers


@pytest.mark.parametrize(
    "bad",
    [
        "http://lakewoodcreekhoa.com/",
        "https://user:pw@evil.example/",
        "https://lakewoodcreekhoa.com@evil.example/",
        "javascript:alert(1)",
        " https://lakewoodcreekhoa.com/",
        "",
    ],
)
def test_invalid_documents_url_falls_back_to_default(bad: str) -> None:
    # WebSettings.from_env rejects these at startup; the route re-checks anyway.
    h = build_harness(documents_url=bad)
    response = h.client.get("/documents", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == DEFAULT_DOCUMENTS_URL


def test_unaffected_by_budget_and_rate_limit() -> None:
    h = build_harness(monthly_budget_usd=0.05, rate_limit_per_hour=1)
    asyncio.run(h.budget.add_spend(0.05))
    for _ in range(3):
        h.client.post("/api/ask", json={"question": QUESTION})
    assert h.client.post("/api/ask", json={"question": QUESTION}).status_code == 429
    for _ in range(5):
        response = h.client.get("/documents", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == DEFAULT_DOCUMENTS_URL
    assert h.asker.questions == []


def test_only_get_is_allowed() -> None:
    h = build_harness()
    assert h.client.post("/documents").status_code == 405


@pytest.mark.parametrize(
    "crlf",
    [
        "https://lakewoodcreekhoa.com/\r\nSet-Cookie: pwned=1",
        "https://lakewoodcreekhoa.com/\nX-Injected: 1",
        "https://lakewoodcreekhoa.com/%0d%0aX-Injected:%201",
    ],
)
def test_crlf_in_configured_url_never_splits_the_response(crlf: str) -> None:
    h = build_harness(documents_url=crlf)
    response = h.client.get("/documents", follow_redirects=False)
    assert response.status_code == 307
    location = response.headers["location"]
    assert "\r" not in location and "\n" not in location
    assert location.startswith("https://lakewoodcreekhoa.com/")
    assert "set-cookie" not in response.headers
    assert "x-injected" not in response.headers


def test_request_cannot_choose_the_destination() -> None:
    url = "https://docs.example.org/hoa/"
    h = build_harness(documents_url=url)
    attempts = [
        ("/documents?url=https://evil.example/", {}),
        ("/documents?next=https://evil.example/&redirect=//evil.example", {}),
        ("/documents//evil.example", {}),
        ("/documents", {"Host": "evil.example"}),
        ("/documents", {"X-Forwarded-Host": "evil.example"}),
        ("/documents", {"Referer": "https://evil.example/"}),
    ]
    for path, headers in attempts:
        response = h.client.get(path, headers=headers, follow_redirects=False)
        if response.status_code == 404:
            continue  # A different path is not the route at all.
        assert response.status_code == 307, path
        assert response.headers["location"] == url, (path, headers)


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_write_methods_are_not_allowed(method: str) -> None:
    h = build_harness()
    response = h.client.request(method, "/documents", follow_redirects=False)
    assert response.status_code == 405
    assert "location" not in response.headers

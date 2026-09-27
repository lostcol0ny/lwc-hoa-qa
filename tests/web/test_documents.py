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

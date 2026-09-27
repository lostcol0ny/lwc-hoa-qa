"""Security response headers and the same-origin guard.

On Vercel, files in ``public/`` are served from the CDN and never reach this
middleware, so ``vercel.json`` repeats these headers for every route. A test
checks that the two copies match.
"""

from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Citation links are plain navigations, which CSP doesn't govern, so the page
# can link to the HOA's PDFs without widening any directive.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
    "base-uri 'none'; form-action 'self'"
)

SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": (
        "camera=(), microphone=(), geolocation=(), payment=(), usb=(), "
        "interest-cohort=()"
    ),
    "X-Frame-Options": "DENY",
    "Cross-Origin-Opener-Policy": "same-origin",
}


class SecurityHeadersMiddleware:
    """Add ``SECURITY_HEADERS`` to every response; ``no-store`` for the API."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_api = scope["path"].startswith("/api/")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
                if is_api:
                    headers["Cache-Control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_with_headers)


def is_cross_origin(origin: str | None, host: str | None) -> bool:
    """True when a browser-sent ``Origin`` names a different host than ours.

    There's deliberately no CORS middleware: without ``Access-Control-Allow-*``
    headers, browsers block cross-origin reads and JSON preflights. This check
    also rejects "simple" cross-origin POSTs, which skip the preflight.
    """
    if not origin:
        return False
    if origin == "null" or not host:
        return True
    return urlsplit(origin).netloc.lower() != host.lower()

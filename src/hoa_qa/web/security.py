"""Security response headers and the same-origin guard.

On Vercel, files in ``public/`` are served from the CDN and never reach this
middleware, so ``vercel.json`` repeats these headers for every route. A test
checks that the two copies match.
"""

from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.requests import HTTPConnection
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


Origin = tuple[str, str, int]
"""A normalized (scheme, hostname, effective port) web origin."""

DEFAULT_PORTS = {"http": 80, "https": 443}


def parse_origin(value: str) -> Origin | None:
    """Parse ``scheme://host[:port]`` into a normalized origin; None if malformed.

    Anything a browser wouldn't send as an ``Origin`` (other schemes, a path,
    userinfo, a bad or explicit ``0`` port, surrounding whitespace, the opaque
    ``null``) is malformed. Control characters, ``?`` and ``#`` are rejected
    before parsing: ``urlsplit`` silently strips tabs and newlines, and an
    empty query or fragment (``https://host?``) would otherwise vanish.
    """
    if not value or value != value.strip():
        return None
    if any(ch in "?#" or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return None
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    if parts.username is not None or parts.password is not None:
        return None
    if parts.path or parts.query or parts.fragment:
        return None
    if port is None:
        port = DEFAULT_PORTS[scheme]
    elif port == 0:
        return None
    return (scheme, parts.hostname.lower(), port)


def request_origin(request: HTTPConnection, *, on_vercel: bool) -> Origin | None:
    """The origin the browser used to reach us: the trusted external origin.

    On Vercel the function sees an internal connection, so the scheme comes
    from ``x-forwarded-proto`` (set by Vercel's edge, https by default).
    Locally it comes from the request URL. The host and port come from
    ``Host``.
    """
    if on_vercel:
        proto = request.headers.get("x-forwarded-proto", "").split(",")[0]
        scheme = proto.strip().lower() or "https"
    else:
        scheme = request.url.scheme
    host = request.headers.get("host", "")
    return parse_origin(f"{scheme}://{host}") if host else None


def origin_allowed(request: HTTPConnection, *, on_vercel: bool) -> bool:
    """Allow a request with no ``Origin`` or a same-origin one; reject the rest.

    There's deliberately no CORS middleware: without ``Access-Control-Allow-*``
    headers, browsers block cross-origin reads and JSON preflights. This check
    also rejects "simple" cross-origin POSTs, which skip the preflight, and
    compares scheme and port as well as host, so ``http://`` doesn't pass for
    an ``https://`` site.

    A missing ``Origin`` is allowed: browsers send it on every POST, so only
    non-browser clients (curl, scripts) omit it, and those could forge any
    value anyway. The rate limits and budget bound them.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return True
    expected = request_origin(request, on_vercel=on_vercel)
    return expected is not None and parse_origin(origin) == expected

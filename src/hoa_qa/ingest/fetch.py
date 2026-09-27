"""Bounded HTTP requests with persistent byte-for-byte URL caching."""

import hashlib
import ssl
import time
from collections.abc import Callable, Mapping
from importlib.resources import files
from pathlib import Path

import certifi
import httpx

MAX_BYTES = 50 * 1024 * 1024  # The largest source (Declaration PDF) is ~8 MB.
USER_AGENT = "LakewoodCreekHOA-QA-Corpus/1.0 (public document index)"
# Seconds between requests to one host, from that host's robots.txt.
CRAWL_DELAYS: Mapping[str, float] = {"ftp.ilga.gov": 10.0}
# ILGA's servers send only their leaf certificate, without the intermediate,
# so no non-browser client can build the chain. The intermediate is added to
# the trust store below; it must still chain to a certifi root.
PINNED_INTERMEDIATES = ("sectigo-ov-r40.pem",)


class FetchError(ValueError):
    """A response the build must not ingest (scheme or size)."""


def _https_only(request: httpx.Request) -> None:
    # Runs for the first request and every redirect hop.
    if request.url.scheme != "https":
        raise FetchError(f"refusing non-HTTPS request {request.url}")


def tls_context() -> ssl.SSLContext:
    """certifi roots plus the pinned intermediates, with full-chain checks.

    Partial chains are refused, so a pinned intermediate is never a trust
    anchor on its own: a certificate it signs is accepted only because the
    intermediate itself verifies against a certifi root.
    """
    context = ssl.create_default_context(cafile=certifi.where())
    for name in PINNED_INTERMEDIATES:
        pem = files("hoa_qa.ingest").joinpath("certs", name).read_text("ascii")
        context.load_verify_locations(cadata=pem[pem.index("-----BEGIN") :])
    context.verify_flags &= ~ssl.VERIFY_X509_PARTIAL_CHAIN
    return context


class Fetcher:
    def __init__(
        self,
        cache: Path,
        *,
        transport: httpx.BaseTransport | None = None,
        delays: Mapping[str, float] = CRAWL_DELAYS,
        clock: Callable[[], float] = time.monotonic,
        pause: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache = cache
        self.transport = transport
        self.delays = delays
        self._clock = clock
        self._pause = pause
        self._last: dict[str, float] = {}
        cache.mkdir(parents=True, exist_ok=True)

    def _throttle(self, url: str) -> None:
        """Wait out the host's crawl delay; every attempt counts as a request."""
        host = httpx.URL(url).host
        delay = self.delays.get(host)
        if delay is None:
            return
        last = self._last.get(host)
        if last is not None:
            wait = last + delay - self._clock()
            if wait > 0:
                self._pause(wait)
        self._last[host] = self._clock()

    def _download(self, client: httpx.Client, url: str) -> bytes:
        self._throttle(url)
        with client.stream("GET", url) as response:
            response.raise_for_status()
            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > MAX_BYTES:
                raise FetchError(f"{url}: {declared} bytes exceeds {MAX_BYTES}")
            data = bytearray()
            for chunk in response.iter_bytes():
                data += chunk
                if len(data) > MAX_BYTES:
                    raise FetchError(f"{url}: response exceeds {MAX_BYTES} bytes")
            return bytes(data)

    def __call__(self, url: str) -> bytes:
        path = self.cache / hashlib.sha256(url.encode()).hexdigest()
        if path.exists():
            return path.read_bytes()
        data = self.fresh(url)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return data

    def fresh(self, url: str) -> bytes:
        """Download ``url`` without reading or writing the cache."""
        with httpx.Client(
            timeout=45,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            event_hooks={"request": [_https_only]},
            transport=self.transport,
            verify=tls_context(),
        ) as client:
            for attempt in range(4):
                try:
                    return self._download(client, url)
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    if isinstance(exc, httpx.HTTPStatusError) and (
                        exc.response.status_code < 500
                        and exc.response.status_code not in (408, 429)
                    ):
                        raise
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
        raise RuntimeError("fetch attempts exhausted")

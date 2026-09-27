"""Bounded HTTP requests with persistent byte-for-byte URL caching."""

import hashlib
import time
from pathlib import Path

import httpx

MAX_BYTES = 50 * 1024 * 1024  # The largest source (Declaration PDF) is ~8 MB.
USER_AGENT = "LakewoodCreekHOA-QA-Corpus/1.0 (public document index)"


class FetchError(ValueError):
    """A response the build must not ingest (scheme or size)."""


def _https_only(request: httpx.Request) -> None:
    # Runs for the first request and every redirect hop.
    if request.url.scheme != "https":
        raise FetchError(f"refusing non-HTTPS request {request.url}")


class Fetcher:
    def __init__(
        self, cache: Path, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.cache = cache
        self.transport = transport
        cache.mkdir(parents=True, exist_ok=True)

    def _download(self, client: httpx.Client, url: str) -> bytes:
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
        with httpx.Client(
            timeout=45,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            event_hooks={"request": [_https_only]},
            transport=self.transport,
        ) as client:
            for attempt in range(4):
                try:
                    data = self._download(client, url)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_bytes(data)
                    temporary.replace(path)
                    return data
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

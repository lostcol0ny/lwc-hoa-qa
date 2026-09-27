"""Bounded HTTP requests with persistent byte-for-byte URL caching."""

import hashlib
import time
from pathlib import Path

import httpx


class Fetcher:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, url: str) -> bytes:
        path = self.cache / hashlib.sha256(url.encode()).hexdigest()
        if path.exists():
            return path.read_bytes()
        with httpx.Client(
            timeout=45,
            follow_redirects=True,
            headers={
                "User-Agent": "LakewoodCreekHOA-QA-Corpus/1.0 (public document index)"
            },
        ) as client:
            for attempt in range(4):
                try:
                    response = client.get(url)
                    response.raise_for_status()
                    data = response.content
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

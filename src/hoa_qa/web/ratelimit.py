"""Per-IP fixed-window rate limiting over the shared ``CounterStore``.

This is the app-level layer; the Vercel Firewall rule in docs/web-ops.md is
the edge layer in front of it.
"""

import hashlib
import logging
from dataclasses import dataclass

from starlette.requests import Request

from hoa_qa.budget import Clock, CounterStore, utc_now

logger = logging.getLogger("hoa_qa.web")

HOUR_TTL_SECONDS = 2 * 60 * 60
DAY_TTL_SECONDS = 2 * 24 * 60 * 60


def client_ip(request: Request, *, on_vercel: bool) -> str:
    """Return the client IP used as the rate-limit key.

    ``x-forwarded-for``/``x-real-ip`` are trusted only on Vercel, whose edge
    overwrites them with the real client IP. Anywhere else a client could set
    them to any value and dodge the limit, so the socket peer address is used.
    """
    if on_vercel:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip
    return request.client.host if request.client else "unknown"


def _ip_digest(ip: str) -> str:
    # Keys hold a digest, not the raw IP, so Redis never stores addresses.
    return hashlib.sha256(ip.encode()).hexdigest()[:32]


@dataclass
class RateLimiter:
    counters: CounterStore
    per_hour: int
    per_day: int
    clock: Clock = utc_now

    async def hit(self, ip: str) -> bool:
        """Count one request for ``ip``; return True if it is within limits.

        Store failures fail open: the monthly budget (which fails closed) is
        the hard cap, and a Redis blip shouldn't lock every neighbor out.
        """
        now = self.clock()
        digest = _ip_digest(ip)
        try:
            hourly = await self.counters.incr(
                f"ratelimit:{digest}:h:{now:%Y-%m-%dT%H}", 1, HOUR_TTL_SECONDS
            )
            daily = await self.counters.incr(
                f"ratelimit:{digest}:d:{now:%Y-%m-%d}", 1, DAY_TTL_SECONDS
            )
        except Exception as exc:
            logger.warning("rate limiter unavailable error_type=%s", type(exc).__name__)
            return True
        return hourly <= self.per_hour and daily <= self.per_day

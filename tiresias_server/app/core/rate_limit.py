from __future__ import annotations

import collections
import threading
import time
from typing import Dict, Deque
from fastapi import HTTPException, Request, status


class InMemoryRateLimiter:
    """
    Lightweight, thread-safe, sliding-window rate limiter without external dependencies.
    Tracks client hit timestamps and enforces requests-per-window caps.
    """

    def __init__(self, requests_limit: int = 15, window_seconds: float = 60.0) -> None:
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def is_allowed(self, key: str) -> bool:
        """
        Determines whether a request with identifying key (e.g. client IP) is permitted.
        Returns True if allowed, False if limit exceeded.
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # Periodic cleanup of idle keys every 5 minutes or when table grows
            if now - self._last_cleanup > 300.0 or len(self._hits) > 5000:
                self._purge_idle(cutoff)
                self._last_cleanup = now

            timestamps = self._hits[key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) < self.requests_limit:
                timestamps.append(now)
                return True
            return False

    def _purge_idle(self, cutoff: float) -> None:
        idle_keys = []
        for k, timestamps in self._hits.items():
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if not timestamps:
                idle_keys.append(k)
        for k in idle_keys:
            self._hits.pop(k, None)

    def reset(self) -> None:
        """Clears all stored counters (useful for unit tests)."""
        with self._lock:
            self._hits.clear()


def get_client_ip(request: Request) -> str:
    """Extracts client IP address respecting X-Forwarded-For and X-Real-IP headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",")]
        if parts and parts[0]:
            return parts[0]
    real_ip = request.headers.get("x-real-ip")
    if real_ip and real_ip.strip():
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


# Default auth rate limiter: max 15 requests per 60 seconds per IP
auth_rate_limiter = InMemoryRateLimiter(requests_limit=15, window_seconds=60.0)


async def rate_limit_auth(request: Request) -> None:
    """FastAPI dependency enforcing rate limits on authentication endpoints."""
    client_ip = get_client_ip(request)
    if not auth_rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again in a minute.",
            headers={"Retry-After": "60"},
        )

"""Cross-process rate limiting on top of Redis.

Every process that talks to arXiv or Semantic Scholar (API, Celery workers, the CLI)
goes through the same Redis keys, so the limit holds for the whole machine:

    limiter = RedisRateLimiter(get_redis(), "arxiv", min_interval_s=3.0)
    with limiter.slot():
        ...one HTTP request...

Two mechanisms work together:
1. a *lock*: only one request is in flight at a time ("one connection at a time");
2. a *gate*: a key that lives for `min_interval_s`. A request may only start when it
   can create the key (`SET key NX PX ...`), so request starts are spaced apart.

The lock has a TTL, so a worker that crashes while holding it cannot block everyone
forever: Redis deletes the lock by itself after `lock_ttl_s`.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

import redis
from redis.exceptions import LockError

from app.config import get_settings

logger = logging.getLogger(__name__)


class RateLimitTimeout(Exception):
    """Could not get a request slot within `wait_timeout_s`."""


def get_redis() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


class RedisRateLimiter:
    def __init__(
        self,
        client: redis.Redis,
        name: str,
        min_interval_s: float,
        lock_ttl_s: float = 30.0,
        wait_timeout_s: float = 120.0,
    ) -> None:
        self._redis = client
        self.min_interval_s = min_interval_s
        self.lock_ttl_s = lock_ttl_s  # must exceed the longest single request
        self.wait_timeout_s = wait_timeout_s
        self.lock_key = f"ratelimit:{name}:lock"
        self.gate_key = f"ratelimit:{name}:gate"

    @contextmanager
    def slot(self) -> Iterator[None]:
        """Hold the lock and respect the gate while the caller makes ONE request."""
        deadline = time.monotonic() + self.wait_timeout_s
        lock = self._redis.lock(
            self.lock_key,
            timeout=self.lock_ttl_s,
            blocking_timeout=self.wait_timeout_s,
            thread_local=True,
        )
        if not lock.acquire():
            raise RateLimitTimeout(f"could not get lock {self.lock_key}")
        try:
            self._wait_for_gate(deadline)
            yield
        finally:
            try:
                lock.release()
            except LockError:
                # The TTL expired while we held it (request slower than lock_ttl_s);
                # someone else may own the lock now, so there is nothing to release.
                logger.warning("lock %s expired before release", self.lock_key)

    def _wait_for_gate(self, deadline: float) -> None:
        interval_ms = max(1, int(self.min_interval_s * 1000))
        while True:
            # Atomic "create if absent with expiry": exactly one caller wins per interval.
            if self._redis.set(self.gate_key, "1", nx=True, px=interval_ms):
                return
            remaining_ms = self._redis.pttl(self.gate_key)
            if time.monotonic() > deadline:
                raise RateLimitTimeout(f"gate {self.gate_key} did not open in time")
            # -2 = key vanished (retry now); otherwise sleep until it expires.
            time.sleep(max(remaining_ms, 0) / 1000 + 0.005)

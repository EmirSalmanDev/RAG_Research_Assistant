"""Limiter behaviour against the real local Redis (separate connections = separate processes)."""

import threading
import time
import uuid
from collections.abc import Iterator

import pytest

from app.sources.ratelimit import RateLimitTimeout, RedisRateLimiter, get_redis

pytestmark = pytest.mark.integration


@pytest.fixture
def name() -> Iterator[str]:
    n = f"test-{uuid.uuid4().hex[:8]}"
    yield n
    client = get_redis()
    client.delete(f"ratelimit:{n}:lock", f"ratelimit:{n}:gate")
    client.close()


def _limiter(name: str, **kw: float) -> RedisRateLimiter:
    # Each limiter gets its own connection, like separate worker processes would.
    return RedisRateLimiter(get_redis(), name, **kw)


def test_request_starts_are_spaced_by_the_interval(name: str) -> None:
    starts: list[float] = []

    def worker() -> None:
        limiter = _limiter(name, min_interval_s=0.2)
        for _ in range(3):
            with limiter.slot():
                starts.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    starts.sort()
    assert len(starts) == 6
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    assert min(gaps) >= 0.18  # 0.2s minus timer jitter


def test_only_one_request_in_flight(name: str) -> None:
    in_flight = 0
    peak = 0
    guard = threading.Lock()

    def worker() -> None:
        nonlocal in_flight, peak
        with _limiter(name, min_interval_s=0.01).slot():
            with guard:
                in_flight += 1
                peak = max(peak, in_flight)
            time.sleep(0.05)
            with guard:
                in_flight -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_crashed_holder_cannot_block_forever(name: str) -> None:
    """A worker that took the lock and died must not block others: the lock has a TTL."""
    limiter = _limiter(name, min_interval_s=0.01, lock_ttl_s=0.3)
    # Simulate the crash: another connection takes the same lock and never releases it.
    assert get_redis().lock(limiter.lock_key, timeout=0.3).acquire()

    started = time.monotonic()
    with limiter.slot():
        waited = time.monotonic() - started
    assert 0.1 < waited < 2.0  # got in once the dead holder's lock expired


def test_waiting_too_long_raises_instead_of_hanging(name: str) -> None:
    limiter = _limiter(name, min_interval_s=0.01, wait_timeout_s=0.3)
    assert get_redis().lock(limiter.lock_key, timeout=30).acquire()  # a live, slow holder
    with pytest.raises(RateLimitTimeout), limiter.slot():
        pytest.fail("should not get a slot")


def test_lock_expiring_during_a_slow_request_is_not_an_error(name: str) -> None:
    limiter = _limiter(name, min_interval_s=0.01, lock_ttl_s=0.1)
    with limiter.slot():
        time.sleep(0.3)  # request slower than the lock TTL; release must not raise

"""Shared HTTP plumbing for the source clients: timeouts, retries, backoff.

`call_with_retry` retries *transient* failures only (timeouts, connection errors,
429 and 5xx) with exponential backoff plus random jitter, and honours a
`Retry-After` header. Other errors (404, 400, ...) are returned/raised at once.
"""

import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

USER_AGENT = "research-copilot/0.1 (personal research tool)"
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
DEFAULT_TIMEOUT_S = 20.0
MAX_RETRY_AFTER_S = 60.0  # never sleep longer than this because a server asked


class Limiter(Protocol):
    def slot(self) -> AbstractContextManager[None]: ...


class RetryableError(Exception):
    """A transient failure (HTTP 429/5xx or a network problem) worth retrying."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


_backoff = wait_exponential_jitter(initial=1.0, max=30.0, jitter=1.0)


def _wait(state: RetryCallState) -> float:
    """Exponential backoff + jitter, but at least as long as the server's Retry-After."""
    delay = _backoff(state)
    exc = state.outcome.exception() if state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        delay = max(delay, min(float(retry_after), MAX_RETRY_AFTER_S))
    return delay


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`; retry on RetryableError / httpx transport errors; re-raise the last error."""
    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=_wait,
        retry=retry_if_exception_type((RetryableError, httpx.TransportError)),
        sleep=sleep,
        reraise=True,
    )
    return retrying(fn)


def make_client(
    base_url: str = "",
    headers: dict[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """httpx client with our User-Agent. `transport` lets tests inject a fake server."""
    return httpx.Client(
        base_url=base_url,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        timeout=DEFAULT_TIMEOUT_S,
        follow_redirects=True,
        transport=transport,
    )


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:  # an HTTP-date form; we just fall back to our own backoff
        return None


def send(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    limiter: Limiter | None = None,
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: object,
) -> httpx.Response:
    """One logical request with retries. Each attempt takes its own limiter slot.

    Retries happen *outside* the slot, so a backoff sleep never holds the lock that
    other processes are waiting for.
    """

    def once() -> httpx.Response:
        with limiter.slot() if limiter else nullcontext():
            resp = client.request(method, url, **kwargs)  # type: ignore[arg-type]
        if resp.status_code in RETRYABLE_STATUS:
            raise RetryableError(
                f"{method} {url} -> HTTP {resp.status_code}",
                status=resp.status_code,
                retry_after=_retry_after_seconds(resp),
            )
        return resp

    return call_with_retry(once, max_attempts=max_attempts, sleep=sleep)

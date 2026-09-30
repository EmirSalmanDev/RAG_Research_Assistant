"""Retry/backoff behaviour with a fake server (httpx.MockTransport): no network, no sleeping."""

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest

from app.sources.http import RetryableError, call_with_retry, make_client, send


class FakeLimiter:
    def __init__(self) -> None:
        self.slots = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.slots += 1
        yield


def _client(responses: list[httpx.Response | Exception]) -> tuple[httpx.Client, list[int]]:
    calls: list[int] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return make_client("https://example.test", transport=httpx.MockTransport(handler)), calls


def test_retries_5xx_then_succeeds_and_each_attempt_takes_a_slot() -> None:
    client, calls = _client(
        [httpx.Response(503), httpx.Response(502), httpx.Response(200, json={})]
    )
    limiter, sleeps = FakeLimiter(), list[float]()
    resp = send(client, "GET", "/x", limiter=limiter, sleep=sleeps.append)
    assert resp.status_code == 200
    assert len(calls) == 3 and limiter.slots == 3 and len(sleeps) == 2
    assert sleeps[1] >= sleeps[0] - 1.0  # backoff grows (jitter is at most 1s)


def test_retry_after_header_is_honoured() -> None:
    client, _ = _client([httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200)])
    sleeps: list[float] = []
    send(client, "GET", "/x", sleep=sleeps.append)
    assert sleeps[0] >= 7


def test_gives_up_after_max_attempts_with_the_last_error() -> None:
    client, calls = _client([httpx.Response(500)] * 3)
    with pytest.raises(RetryableError) as err:
        send(client, "GET", "/x", max_attempts=3, sleep=lambda s: None)
    assert err.value.status == 500 and len(calls) == 3


def test_non_transient_errors_are_not_retried() -> None:
    client, calls = _client([httpx.Response(404)])
    assert send(client, "GET", "/x", sleep=lambda s: None).status_code == 404
    assert len(calls) == 1


def test_network_errors_are_retried() -> None:
    client, calls = _client([httpx.ConnectError("boom"), httpx.Response(200)])
    assert send(client, "GET", "/x", sleep=lambda s: None).status_code == 200
    assert len(calls) == 2


def test_call_with_retry_does_not_retry_other_exceptions() -> None:
    def boom() -> None:
        raise ValueError("bug")

    with pytest.raises(ValueError):
        call_with_retry(boom, sleep=lambda s: None)

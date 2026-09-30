import json
from collections.abc import Callable

import httpx

from app.sources.s2 import S2Client, S2Paper, S2Unavailable
from tests.helpers import CountingLimiter, mock_client

PAPER = {
    "citationCount": 123,
    "references": [
        {"paperId": "p1", "externalIds": {"ArXiv": "2001.00001", "DOI": "10.1/x"}},
        {"paperId": "p2", "externalIds": {"DOI": "10.1/y"}},  # no arXiv id: skipped
        {"paperId": None, "externalIds": None},  # S2 sometimes has empty reference stubs
    ],
}


def _s2(
    handler: Callable[[httpx.Request], httpx.Response],
    api_key: str | None = None,
    max_attempts: int = 5,
) -> tuple[S2Client, list[httpx.Request], CountingLimiter]:
    seen: list[httpx.Request] = []

    def wrapped(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return handler(r)

    limiter = CountingLimiter()
    client = S2Client(client=mock_client(wrapped), api_key=api_key, limiter=limiter,
                      sleep=lambda s: None, max_attempts=max_attempts)  # fmt: skip
    return client, seen, limiter


def test_paper_parses_counts_and_arxiv_references() -> None:
    client, seen, limiter = _s2(lambda r: httpx.Response(200, json=PAPER))
    result = client.paper("2601.00001")
    assert isinstance(result, S2Paper)
    assert result.citation_count == 123 and result.references == ["2001.00001"]
    assert result.references_total == 3  # all listed references, even without an arXiv id
    assert seen[0].url.path == "/graph/v1/paper/arXiv:2601.00001"
    assert seen[0].url.params["fields"] == "citationCount,references.externalIds"
    assert limiter.slots == 1


def test_api_key_header_only_when_key_is_set() -> None:
    client, seen, _ = _s2(lambda r: httpx.Response(200, json=PAPER), api_key=None)
    client.paper("2601.00001")
    assert "x-api-key" not in seen[0].headers
    client, seen, _ = _s2(lambda r: httpx.Response(200, json=PAPER), api_key="k")
    client.paper("2601.00001")
    assert seen[0].headers["x-api-key"] == "k"


def test_batch_is_one_post_aligned_with_request_order_and_null_means_missing() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        assert json.loads(r.content) == {"ids": ["ARXIV:2601.00001", "ARXIV:9999.99999"]}
        return httpx.Response(200, json=[PAPER, None])

    client, seen, limiter = _s2(handler)
    results = client.batch(["2601.00001", "9999.99999"])
    assert isinstance(results["2601.00001"], S2Paper)
    missing = results["9999.99999"]
    assert isinstance(missing, S2Unavailable) and "not found" in missing.reason
    assert len(seen) == 1 and seen[0].method == "POST" and limiter.slots == 1


def test_batch_splits_into_sequential_chunks_of_500() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[PAPER] * len(json.loads(r.content)["ids"]))

    client, seen, limiter = _s2(handler)
    ids = [f"2601.{n:05d}" for n in range(501)]
    assert len(client.batch(ids)) == 501
    assert len(seen) == 2 and limiter.slots == 2


def test_429_after_retries_returns_unavailable_instead_of_raising() -> None:
    client, seen, limiter = _s2(
        lambda r: httpx.Response(429, headers={"Retry-After": "2"}), max_attempts=3
    )
    result = client.paper("2601.00001")
    assert isinstance(result, S2Unavailable) and "unavailable after retries" in result.reason
    assert len(seen) == 3 and limiter.slots == 3  # every retry took its own slot (1 req/s holds)


def test_batch_5xx_marks_every_id_unavailable() -> None:
    client, _, _ = _s2(lambda r: httpx.Response(503), max_attempts=2)
    results = client.batch(["2601.00001", "2601.00002"])
    assert all(isinstance(v, S2Unavailable) for v in results.values()) and len(results) == 2


def test_network_error_is_unavailable_too() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client, _, _ = _s2(handler, max_attempts=2)
    assert isinstance(client.paper("2601.00001"), S2Unavailable)


def test_404_and_unexpected_status_are_unavailable_without_retry() -> None:
    client, seen, _ = _s2(lambda r: httpx.Response(404))
    result = client.paper("2601.00001")
    assert isinstance(result, S2Unavailable) and "not found" in result.reason and len(seen) == 1
    client, _, _ = _s2(lambda r: httpx.Response(400))
    result = client.paper("2601.00001")
    assert isinstance(result, S2Unavailable) and result.reason == "HTTP 400"


def test_include_citations_requests_the_extra_field() -> None:
    body = {**PAPER, "citations": [{"externalIds": {"ArXiv": "2602.00002"}}]}
    client, seen, _ = _s2(lambda r: httpx.Response(200, json=body))
    result = client.paper("2601.00001", include_citations=True)
    assert isinstance(result, S2Paper) and result.citations == ["2602.00002"]
    assert "citations.externalIds" in seen[0].url.params["fields"]

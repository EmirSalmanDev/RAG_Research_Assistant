from collections.abc import Callable

import httpx
import pytest

from app.sources.hf import HFClient
from tests.helpers import mock_client


def _hf(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[HFClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return HFClient(client=mock_client(wrapped), sleep=lambda s: None), seen


def test_daily_papers_parses_entries_and_sends_trending_sort() -> None:
    payload = [
        {"paper": {"id": "2601.00001", "title": "T", "upvotes": 12, "githubRepo": "https://github.com/a/b",
                   "githubStars": 5, "publishedAt": "2026-01-01T00:00:00.000Z", "extra": "ignored"},
         "numComments": 3},
        {"paper": {"id": "2601.00002"}},
    ]  # fmt: skip
    client, seen = _hf(lambda r: httpx.Response(200, json=payload))
    papers = client.daily_papers(__import__("datetime").date(2026, 1, 1))
    assert [p.id for p in papers] == ["2601.00001", "2601.00002"]
    assert papers[0].upvotes == 12 and papers[0].github_repo == "https://github.com/a/b"
    assert papers[1].upvotes == 0 and papers[1].github_repo is None
    assert dict(seen[0].url.params) == {"date": "2026-01-01", "sort": "trending"}


def test_paper_found_and_missing() -> None:
    client, _ = _hf(
        lambda r: (
            httpx.Response(200, json={"id": "2601.00001", "upvotes": 7})
            if r.url.path.endswith("2601.00001")
            else httpx.Response(404)
        )
    )
    paper = client.paper("2601.00001")
    assert paper is not None and paper.upvotes == 7
    assert client.paper("9999.99999") is None


def test_paper_markdown_is_returned_as_text() -> None:
    client, seen = _hf(lambda r: httpx.Response(200, text="# Invented\n\nBody."))
    assert client.paper_markdown("2601.00001") == "# Invented\n\nBody."
    assert seen[0].url.path == "/papers/2601.00001.md"


@pytest.mark.parametrize(
    ("kind", "path"), [("model", "models"), ("dataset", "datasets"), ("space", "spaces")]
)
def test_artifacts_use_the_arxiv_filter(kind: str, path: str) -> None:
    client, seen = _hf(
        lambda r: httpx.Response(
            200, json=[{"id": "org/thing", "downloads": 10, "likes": 2, "pipeline_tag": "x"}]
        )
    )
    (item,) = client.artifacts("2601.00001", kind)  # type: ignore[arg-type]
    assert (item.kind, item.repo_id, item.downloads, item.likes) == (kind, "org/thing", 10, 2)
    assert seen[0].url.path == f"/api/{path}"
    assert seen[0].url.params["filter"] == "arxiv:2601.00001"


def test_server_errors_are_retried_then_raised() -> None:
    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    client, _ = _hf(handler)
    with pytest.raises(Exception, match="HTTP 500"):
        client.paper("2601.00001")
    assert len(calls) == 5

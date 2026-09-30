"""ArxivClient with a fake `requests` session serving invented Atom feeds (no network)."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from app.sources.arxiv import ArxivClient, split_version
from app.sources.http import RetryableError


def _entry(n: int) -> str:
    return f"""<entry>
  <id>http://arxiv.org/abs/2601.{n:05d}v2</id>
  <updated>2026-01-05T10:00:00Z</updated>
  <published>2026-01-02T09:30:00Z</published>
  <title>Invented Title
    Number {n}</title>
  <summary>  Invented abstract {n}.
  Second line. </summary>
  <author><name>Ada Example</name></author>
  <author><name>Bo Sample</name></author>
  <arxiv:primary_category term="cs.CL"/>
  <category term="cs.CL"/><category term="cs.LG"/>
</entry>"""


def _feed(first: int, count: int, total: int) -> bytes:
    entries = "".join(_entry(first + i) for i in range(count))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{total}</opensearch:totalResults>
  <opensearch:startIndex>{first}</opensearch:startIndex>
  <opensearch:itemsPerPage>{count}</opensearch:itemsPerPage>
  {entries}
</feed>""".encode()


class FakeResponse:
    def __init__(self, status: int, content: bytes = b"") -> None:
        self.status_code = status
        self.content = content


class FakeSession(requests.Session):
    """Serves a scripted list of responses and records every requested URL."""

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        super().__init__()
        self.queue = list(responses)
        self.urls: list[str] = []

    def get(self, url: str | bytes, **kwargs: object) -> FakeResponse:  # type: ignore[override]
        self.urls.append(str(url))
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class CountingLimiter:
    def __init__(self) -> None:
        self.slots = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.slots += 1
        yield


def _client(
    responses: list[FakeResponse | Exception], page_size: int = 2
) -> tuple[ArxivClient, FakeSession, CountingLimiter]:
    session, limiter = FakeSession(responses), CountingLimiter()
    client = ArxivClient(
        limiter=limiter, session=session, page_size=page_size, sleep=lambda s: None
    )
    return client, session, limiter


def test_get_paper_parses_fields_and_normalises_whitespace() -> None:
    client, session, limiter = _client([FakeResponse(200, _feed(1, 1, 1))])
    paper = client.get_paper("2601.00001")
    assert paper is not None
    assert (paper.arxiv_id, paper.version) == ("2601.00001", 2)
    assert paper.title == "Invented Title Number 1"
    assert paper.abstract == "Invented abstract 1. Second line."
    assert paper.authors == ["Ada Example", "Bo Sample"]
    assert paper.categories == ["cs.CL", "cs.LG"] and paper.primary_category == "cs.CL"
    assert paper.published_at == datetime(2026, 1, 2, 9, 30, tzinfo=UTC)
    assert len(session.urls) == 1 and limiter.slots == 1
    assert parse_qs(urlparse(session.urls[0]).query)["id_list"] == ["2601.00001"]


def test_get_paper_unknown_id_returns_none() -> None:
    client, *_ = _client([FakeResponse(200, _feed(1, 0, 0))])
    assert client.get_paper("9999.99999") is None


def test_new_papers_pages_one_request_per_slot() -> None:
    """5 results, page size 2 -> 3 separate requests, each inside its own limiter slot."""
    client, session, limiter = _client(
        [FakeResponse(200, _feed(1, 2, 5)), FakeResponse(200, _feed(3, 2, 5)),
         FakeResponse(200, _feed(5, 1, 5))]
    )  # fmt: skip
    papers = client.new_papers("cs.CL", since=datetime(2026, 1, 1, tzinfo=UTC), max_results=10)
    assert [p.arxiv_id for p in papers] == [f"2601.{n:05d}" for n in range(1, 6)]
    assert len(session.urls) == 3 and limiter.slots == 3
    starts = [parse_qs(urlparse(u).query)["start"] for u in session.urls]
    assert starts == [["0"], ["2"], ["4"]]
    query = parse_qs(urlparse(session.urls[0]).query)["search_query"][0]
    assert query.startswith("cat:cs.CL AND submittedDate:[202601010000 TO ")


def test_new_papers_stops_at_max_results() -> None:
    client, session, _ = _client(
        [FakeResponse(200, _feed(1, 2, 50)), FakeResponse(200, _feed(3, 1, 50))]
    )
    papers = client.new_papers("cs.CL", since=datetime(2026, 1, 1, tzinfo=UTC), max_results=3)
    assert len(papers) == 3 and len(session.urls) == 2  # never asks for a 3rd page


def test_retries_go_through_our_layer_and_each_retry_takes_a_slot() -> None:
    client, session, limiter = _client(
        [FakeResponse(503), requests.ConnectionError("reset"), FakeResponse(200, _feed(1, 1, 1))]
    )
    assert client.get_paper("2601.00001") is not None
    # The package itself must not have retried: one request per attempt, one slot per attempt.
    assert len(session.urls) == 3 and limiter.slots == 3


def test_gives_up_with_retryable_error_after_max_attempts() -> None:
    client, session, _ = _client([FakeResponse(503)] * 5)
    with pytest.raises(RetryableError):
        client.get_paper("2601.00001")
    assert len(session.urls) == 5


def test_client_is_configured_without_package_retries_or_delay() -> None:
    client, *_ = _client([])
    assert client._client.num_retries == 0 and client._client.delay_seconds == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1706.03762", ("1706.03762", None)), ("1706.03762v7", ("1706.03762", 7)),
     ("cs/0112017v1", ("cs/0112017", 1))],
)  # fmt: skip
def test_split_version(raw: str, expected: tuple[str, int | None]) -> None:
    assert split_version(raw) == expected

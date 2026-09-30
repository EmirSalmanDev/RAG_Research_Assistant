"""ArxivClient replayed from recorded responses (tests/cassettes/test_arxiv_cassettes/)."""

from datetime import UTC, datetime
from typing import Any

import pytest

from app.sources.arxiv import ArxivClient

PAPER_ID = "1706.03762"


@pytest.mark.vcr
def test_get_paper_from_cassette(arxiv_limiter: Any) -> None:
    paper = ArxivClient(limiter=arxiv_limiter).get_paper(PAPER_ID)
    assert paper is not None
    assert paper.arxiv_id == PAPER_ID and paper.version >= 1
    assert paper.title and paper.abstract and paper.authors
    assert "cs.CL" in paper.categories
    assert paper.published_at.tzinfo is not None


@pytest.mark.vcr
def test_new_papers_from_cassette(arxiv_limiter: Any) -> None:
    client = ArxivClient(limiter=arxiv_limiter, page_size=5)
    papers = client.new_papers(
        "cs.CL",
        since=datetime(2026, 9, 28, tzinfo=UTC),
        until=datetime(2026, 9, 29, tzinfo=UTC),
        max_results=5,
    )
    assert 0 < len(papers) <= 5
    assert all("cs.CL" in p.categories for p in papers)
    assert all(datetime(2026, 9, 28, tzinfo=UTC) <= p.published_at for p in papers)
    if hasattr(arxiv_limiter, "slots"):  # the counting stand-in (replay), not the real limiter
        assert arxiv_limiter.slots == 1  # one page = one request = one slot

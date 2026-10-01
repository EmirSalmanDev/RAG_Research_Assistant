"""ArxivClient.get_papers and .html replayed from recorded responses
(tests/cassettes/test_arxiv_fulltext_cassettes/), and a real round trip of the recorded
arXiv HTML: LaTeXML -> markdown -> clean -> chunk.

The HTML cassette holds only a trimmed excerpt (see tests/cassette_trim.py): real
LaTeXML structure, with real text only if the page's licence allows it.
"""

from typing import Any

import pytest

from app.ingest.chunk import chunk_id, chunk_paper
from app.ingest.clean import clean_markdown
from app.ingest.fulltext import html_to_markdown
from app.ingest.markdown import parse_heading
from app.sources.arxiv import ArxivClient

IDS = ["1706.03762", "2609.24972", "1301.3781"]
HTML_ID, HTML_VERSION = "2609.24972", 2


@pytest.mark.vcr
def test_get_papers_from_cassette(arxiv_limiter: Any) -> None:
    papers = ArxivClient(limiter=arxiv_limiter).get_papers(IDS)
    assert sorted(p.arxiv_id for p in papers) == sorted(IDS)
    assert all(p.title and p.abstract and p.categories and p.version >= 1 for p in papers)
    assert all(p.primary_category == p.categories[0] for p in papers)  # feed order check
    if hasattr(arxiv_limiter, "slots"):  # the counting stand-in (replay)
        assert arxiv_limiter.slots == 1  # 3 ids = one request


@pytest.mark.vcr
def test_html_round_trip_from_cassette(arxiv_limiter: Any) -> None:
    html = ArxivClient(limiter=arxiv_limiter).html(HTML_ID, HTML_VERSION)
    assert html is not None and "ltx_document" in html

    markdown = html_to_markdown(html)
    headings = [h for line in markdown.split("\n") if (h := parse_heading(line))]
    assert headings[0][0] == 1  # the paper title
    assert "References" in (text for _, text in headings), headings
    assert any(text.startswith("1") for _, text in headings)  # numbered sections

    cleaned = clean_markdown(markdown)
    assert cleaned.references  # the bibliography was cut out
    assert parse_heading(cleaned.markdown.split("\n")[0]) is not None  # starts at a section

    chunks = chunk_paper(HTML_ID, HTML_VERSION, "Recorded paper", cleaned.markdown)
    assert chunks and all(c.section_path for c in chunks)
    assert [c.id for c in chunks] == [chunk_id(HTML_ID, HTML_VERSION, c.ord) for c in chunks]

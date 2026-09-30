"""HFClient replayed from recorded responses (tests/cassettes/test_hf_cassettes/)."""

from datetime import date

import pytest

from app.sources.hf import HFClient

PAPER_ID = "1706.03762"


@pytest.mark.vcr
def test_daily_papers_from_cassette() -> None:
    papers = HFClient().daily_papers(date(2026, 9, 29))
    assert papers and all(p.id for p in papers)


@pytest.mark.vcr
def test_paper_from_cassette() -> None:
    paper = HFClient().paper(PAPER_ID)
    assert paper is not None and paper.id == PAPER_ID and paper.upvotes >= 0


@pytest.mark.vcr
def test_paper_markdown_from_cassette_is_short() -> None:
    text = HFClient().paper_markdown(PAPER_ID)
    assert text and len(text.encode()) <= 1000  # truncated when recorded (rule 14)


@pytest.mark.vcr
@pytest.mark.parametrize("kind", ["model", "dataset", "space"])
def test_artifacts_from_cassette(kind: str) -> None:
    items = HFClient().artifacts(PAPER_ID, kind)  # type: ignore[arg-type]
    assert all(a.kind == kind and a.repo_id for a in items)

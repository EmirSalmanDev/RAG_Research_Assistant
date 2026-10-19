"""SqlPaperStore on a scratch database: the real queries behind /papers and /chunks."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.api.papers import SqlPaperStore
from app.db.models import Chunk, Paper

pytestmark = pytest.mark.integration

PUBLISHED = datetime(2026, 9, 1, tzinfo=UTC)


def _seed(session: Session) -> uuid.UUID:
    session.add(
        Paper(
            arxiv_id="2609.00001",
            title="An Invented Paper",
            abstract="An invented abstract.",
            authors=[{"name": "Ada Example"}],
            categories=["cs.CL"],
            primary_category="cs.CL",
            published_at=PUBLISHED,
        )
    )
    session.flush()
    ids = [uuid.uuid4(), uuid.uuid4()]
    # Inserted out of order on purpose: the store must sort by ord.
    for ord_, cid, kind in ((1, ids[1], "section"), (0, ids[0], "abstract")):
        session.add(
            Chunk(
                id=cid,
                arxiv_id="2609.00001",
                chunk_type=kind,
                section_path=None if kind == "abstract" else "Method",
                ord=ord_,
                text=f"Invented text {ord_}.",
                token_count=4,
            )
        )
    session.commit()
    return ids[1]


def test_store_reads_paper_chunks_and_one_chunk(migrated_session: Session) -> None:
    section_id = _seed(migrated_session)
    # session_scope() uses the settings' DATABASE_URL, which the scratch fixture points
    # at the scratch database.
    store = SqlPaperStore()

    paper = store.get_paper("2609.00001")
    assert paper is not None and paper.authors == ["Ada Example"]
    assert store.get_paper("0000.00000") is None

    listing = store.list_chunks("2609.00001", 10)
    assert listing is not None
    assert [c.ord for c in listing.chunks] == [0, 1]
    limited = store.list_chunks("2609.00001", 1)
    assert limited is not None and len(limited.chunks) == 1
    assert store.list_chunks("0000.00000", 10) is None

    chunk = store.get_chunk(section_id)
    assert chunk is not None
    assert (chunk.title, chunk.section_path) == ("An Invented Paper", "Method")
    assert store.get_chunk(uuid.uuid4()) is None


def _paper(arxiv_id: str, title: str, day: int) -> Paper:
    return Paper(
        arxiv_id=arxiv_id, title=title, abstract="Invented.", authors=[], categories=["cs.CL"],
        primary_category="cs.CL", published_at=datetime(2026, 9, day, tzinfo=UTC),
    )  # fmt: skip


def test_store_lists_papers_by_title_or_exact_id_newest_first(migrated_session: Session) -> None:
    migrated_session.add_all(
        [
            _paper("2609.00001", "Attention Is All You Need (invented)", 1),
            _paper("2609.00002", "Attention to 100% of Detail_x", 3),
            _paper("2609.00003", "A Paper About Adapters", 2),
        ]
    )
    migrated_session.flush()
    migrated_session.add(
        Chunk(
            id=uuid.uuid4(), arxiv_id="2609.00001", chunk_type="section", section_path="1 Intro",
            ord=1, text="Invented body.", token_count=3,
        )
    )  # fmt: skip
    migrated_session.commit()
    store = SqlPaperStore()

    def ids(q: str = "", *, limit: int = 10, full: bool | None = None) -> list[str]:
        return [p.arxiv_id for p in store.list_papers(q, limit, full)]

    assert ids() == ["2609.00002", "2609.00003", "2609.00001"]  # newest first
    assert ids("attention") == ["2609.00002", "2609.00001"]  # case-insensitive substring
    assert ids("100%") == ["2609.00002"]  # % is literal, not a wildcard
    assert ids("%") == ["2609.00002"]
    assert ids("detail_x") == ["2609.00002"] and ids("detail-x") == []  # _ is literal
    assert ids("2609.00003") == ["2609.00003"]  # exact arXiv id
    assert ids("2609.00003v2") == ["2609.00003"]  # a version suffix is ignored
    assert ids("2609.0000") == []  # a partial id is not an id (and not in any title)
    assert ids(limit=2) == ["2609.00002", "2609.00003"]
    assert ids(full=True) == ["2609.00001"]
    assert ids(full=False) == ["2609.00002", "2609.00003"]
    only = store.list_papers("", 1, True)[0]
    assert only.has_fulltext and only.abs_url == "https://arxiv.org/abs/2609.00001"

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

"""Related papers (app/features/related.py and SqlPaperStore.related) on a throwaway
database and a `chunks_test_<random>` collection with fake vectors. No OpenAI; the real
collection is checked to be untouched."""

from datetime import UTC, datetime
from typing import Any

import pytest
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.api import papers as papers_api
from app.config import get_settings
from app.db.models import Citation, Paper
from app.features import related
from app.ingest import index as index_module
from app.ingest.chunk import FIRST_ANALYSIS_ORD, Chunk, abstract_chunk, chunk_id, embedding_text
from app.ingest.index import save_chunks, upsert_analysis_chunks, upsert_paper_chunks
from tests.integration.test_indexing import DIM, FakeDense, FakeSparse, collection  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

A, B, C, D = "2601.00001", "2601.00002", "2601.00003", "2601.00004"


def _paper(s: Session, arxiv_id: str, day: int, abstract: str) -> None:
    s.add(Paper(arxiv_id=arxiv_id, version=1, title=f"Invented {arxiv_id}", abstract=abstract,
                authors=[], categories=["cs.CL"],
                published_at=datetime(2026, 1, day, tzinfo=UTC)))  # fmt: skip
    s.flush()


@pytest.fixture
def corpus(migrated_session: Session, qdrant: QdrantClient, collection: str) -> dict[str, Any]:  # noqa: F811
    """A, B, C indexed (abstracts; A also a section and an analysis chunk); D not indexed.
    A cites B and a paper we do not hold; C cites A."""
    s = migrated_session
    for n, arxiv_id in enumerate((A, B, C, D), start=1):
        _paper(s, arxiv_id, n, f"Invented abstract number {n} about retrieval.")
    kw: dict[str, Any] = {"dense": FakeDense(), "sparse": FakeSparse(), "client": qdrant,
                          "collection": collection}  # fmt: skip
    for arxiv_id in (A, B, C):
        paper = s.get(Paper, arxiv_id)
        assert paper is not None
        chunks = [abstract_chunk(arxiv_id, 1, paper.title, paper.abstract)]
        if arxiv_id == A:
            text = "Invented section text."
            chunks.append(Chunk(chunk_id(A, 1, 1), 1, "section", "Method", text, 4,
                                embedding_text(paper.title, "Method", text)))  # fmt: skip
        save_chunks(s, arxiv_id, chunks)
        upsert_paper_chunks(s, arxiv_id, **kw)
    text = "Invented analysis text."
    path = "Analysis > Method"
    a_title = f"Invented {A}"
    save_chunks(s, A, [Chunk(chunk_id(A, 1, FIRST_ANALYSIS_ORD), FIRST_ANALYSIS_ORD, "analysis",
                             path, text, 4, embedding_text(a_title, path, text))])  # fmt: skip
    upsert_analysis_chunks(s, A, **kw)
    s.add_all([Citation(citing_id=A, cited_id=B), Citation(citing_id=A, cited_id="1706.03762"),
               Citation(citing_id=C, cited_id=A)])  # fmt: skip
    s.commit()
    return {"client": qdrant, "collection": collection}


def test_similar_papers_are_other_papers_abstracts(
    migrated_session: Session, corpus: dict[str, Any]
) -> None:
    status, similar = related.similar_papers(corpus["client"], corpus["collection"], A, 1, 10)
    assert status == "ok"
    assert sorted(p.arxiv_id for p in similar) == [B, C]  # not A itself, each paper once
    assert all(p.held and p.title and p.score is not None for p in similar)
    assert similar[0].published_at is not None
    assert related.similar_papers(corpus["client"], corpus["collection"], A, 1, 1)[1][0] in similar
    assert related.similar_papers(corpus["client"], corpus["collection"], D, 1, 10) == (
        "not_indexed", [],
    )  # fmt: skip


def test_cites_and_cited_by(migrated_session: Session, corpus: dict[str, Any]) -> None:
    s = migrated_session
    cites = related.cites(s, A, 10)
    assert [(p.arxiv_id, p.held) for p in cites] == [(B, True), ("1706.03762", False)]
    assert cites[0].title == f"Invented {B}" and cites[1].title is None
    assert [p.arxiv_id for p in related.cited_by(s, A, 10)] == [C]
    assert related.cited_by(s, B, 10)[0].arxiv_id == A
    assert related.cites(s, A, 1) == cites[:1]


def test_store_related_and_qdrant_failure(
    migrated_session: Session, corpus: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", corpus["collection"])
    get_settings.cache_clear()
    index_module.get_qdrant.cache_clear()
    store = papers_api.SqlPaperStore()
    out = store.related(A, 5)
    assert out is not None and out.similar_status == "ok"
    assert {p.arxiv_id for p in out.similar} == {B, C}
    assert [p.abs_url for p in out.cites] == [
        f"https://arxiv.org/abs/{B}", "https://arxiv.org/abs/1706.03762",
    ]  # fmt: skip
    assert [p.arxiv_id for p in out.cited_by] == [C]
    assert store.related("9999.99999", 5) is None

    def down(*a: object, **k: object) -> None:
        raise ConnectionError("Qdrant is down")

    monkeypatch.setattr(related, "similar_papers", down)
    out = store.related(A, 5)
    assert out is not None and out.similar_status == "unavailable" and out.similar == []
    assert [p.arxiv_id for p in out.cited_by] == [C]  # the citation lists still work
    get_settings.cache_clear()
    index_module.get_qdrant.cache_clear()

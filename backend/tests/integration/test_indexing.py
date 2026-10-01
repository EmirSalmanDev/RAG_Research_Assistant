"""Indexing against a throwaway Postgres database and a `chunks_test_<random>` Qdrant
collection, with fake embedders (no OpenAI calls). The real collection is checked to be
untouched after every test."""

import hashlib
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk as ChunkRow
from app.db.models import LLMUsage, Paper
from app.ingest.chunk import (
    FIRST_ANALYSIS_ORD,
    Chunk,
    abstract_chunk,
    chunk_id,
    chunk_paper,
    embedding_text,
)
from app.ingest.clean import clean_markdown
from app.ingest.embed import DenseEmbedder
from app.ingest.index import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    ensure_collection,
    index_abstracts,
    save_chunks,
    upsert_paper_chunks,
)
from app.llm.usage import embedding_cost, session_recorder

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

DIM = 1536
FIXTURES = Path(__file__).parents[1] / "fixtures" / "markdown"
ID = "2601.00001"


class FakeDense:
    """Deterministic, non-zero vectors; records every call (texts per call)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.calls.append((feature, list(texts)))
        out = []
        for text in texts:
            seed = hashlib.sha256(text.encode()).digest()
            out.append([1.0 + seed[i % 32] / 255 for i in range(DIM)])
        return out


class FakeSparse:
    def embed(self, texts: Sequence[str]) -> list[m.SparseVector]:
        return [m.SparseVector(indices=[len(t) % 1000, 1001], values=[1.0, 0.5]) for t in texts]


@pytest.fixture
def collection(qdrant: QdrantClient, scratch_collection: str) -> str:
    ensure_collection(qdrant, scratch_collection, DIM)
    return scratch_collection


def _paper(session: Session, arxiv_id: str = ID, version: int = 1, **kw: object) -> Paper:
    paper = Paper(
        arxiv_id=arxiv_id,
        version=version,
        title=f"Invented Title {arxiv_id}",
        abstract=f"An invented abstract for {arxiv_id}.",
        authors=[{"name": "Ada Example"}],
        categories=["cs.CL", "cs.LG"],
        primary_category="cs.CL",
        published_at=datetime(2026, 1, 2, 9, 30, tzinfo=UTC),
        **kw,
    )
    session.add(paper)
    session.commit()
    return paper


def _chunks(paper: Paper, fixture: str = "hf_style.md") -> list[Chunk]:
    markdown = clean_markdown((FIXTURES / fixture).read_text()).markdown
    return [
        abstract_chunk(paper.arxiv_id, paper.version, paper.title, paper.abstract),
        *chunk_paper(paper.arxiv_id, paper.version, paper.title, markdown),
    ]


def _rows(
    session: Session, arxiv_id: str = ID, types: tuple[str, ...] = ("abstract", "section")
) -> list[ChunkRow]:
    return list(
        session.scalars(
            select(ChunkRow)
            .where(ChunkRow.arxiv_id == arxiv_id, ChunkRow.chunk_type.in_(types))
            .order_by(ChunkRow.ord)
        )
    )


def _points(qdrant: QdrantClient, collection: str, arxiv_id: str = ID) -> list[m.Record]:
    records, _ = qdrant.scroll(
        collection,
        scroll_filter=m.Filter(
            must=[m.FieldCondition(key="arxiv_id", match=m.MatchValue(value=arxiv_id))]
        ),
        limit=1000,
        with_payload=True,
    )
    return records


def _of_types(points: list[m.Record], *types: str) -> list[m.Record]:
    return [p for p in points if (p.payload or {}).get("chunk_type") in types]


def _seed_analysis(session: Session, qdrant: QdrantClient, collection: str) -> uuid.UUID:
    """An 'analysis' chunk (M6) row and point that re-indexing must never delete."""
    analysis_id = chunk_id(ID, 1, FIRST_ANALYSIS_ORD)
    session.add(
        ChunkRow(
            id=analysis_id,
            arxiv_id=ID,
            chunk_type="analysis",
            section_path="Analysis",
            ord=FIRST_ANALYSIS_ORD,
            text="An invented analysis.",
            token_count=5,
        )
    )
    session.commit()
    qdrant.upsert(
        collection,
        points=[
            m.PointStruct(
                id=str(analysis_id),
                vector={
                    DENSE_VECTOR: [1.0] * DIM,
                    SPARSE_VECTOR: m.SparseVector(indices=[1], values=[1.0]),
                },
                payload={"arxiv_id": ID, "chunk_type": "analysis", "text": "An invented analysis."},
            )
        ],
        wait=True,
    )
    return analysis_id


# --- upsert_paper_chunks ---------------------------------------------------------------------


def test_upsert_writes_one_point_per_row_with_the_data_model_payload(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    paper = _paper(migrated_session, github_url="https://github.com/a/b")
    save_chunks(migrated_session, ID, _chunks(paper))
    migrated_session.commit()
    dense = FakeDense()
    n = upsert_paper_chunks(
        migrated_session, ID, dense=dense, sparse=FakeSparse(), client=qdrant, collection=collection
    )
    rows = _rows(migrated_session)
    points = _points(qdrant, collection)
    assert n == len(rows) == len(points) == 6  # abstract + 5 section chunks
    assert {str(r.id) for r in rows} == {str(p.id) for p in points}
    assert [f for f, _ in dense.calls] == ["embed-abstract", "embed-fulltext"]

    abstract_row = rows[0]
    [point] = qdrant.retrieve(
        collection, ids=[str(abstract_row.id)], with_payload=True, with_vectors=True
    )
    assert point.payload == {
        "arxiv_id": ID,
        "chunk_type": "abstract",
        "section_path": "Abstract",
        "title": paper.title,
        "published_at": "2026-01-02T09:30:00+00:00",
        "categories": ["cs.CL", "cs.LG"],
        "primary_category": "cs.CL",
        "has_code": True,
        "text": paper.abstract,
    }
    assert isinstance(point.vector, dict) and set(point.vector) == {DENSE_VECTOR, SPARSE_VECTOR}
    # what was embedded is the 'Title > Section' text built from the row
    assert dense.calls[0][1][0] == embedding_text(paper.title, "Abstract", paper.abstract)


def test_indexing_twice_gives_the_same_counts(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    paper = _paper(migrated_session)
    save_chunks(migrated_session, ID, _chunks(paper))
    migrated_session.commit()
    embedded = []
    for _ in range(2):
        dense = FakeDense()
        embedded.append(
            upsert_paper_chunks(
                migrated_session,
                ID,
                dense=dense,
                sparse=FakeSparse(),
                client=qdrant,
                collection=collection,
            )
        )
        migrated_session.commit()
        assert len(_rows(migrated_session)) == len(_points(qdrant, collection)) == 6
    assert embedded == [6, 0] and dense.calls == []  # the second run embeds nothing


def test_changed_metadata_updates_the_payload_without_embedding(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    paper = _paper(migrated_session)
    save_chunks(migrated_session, ID, _chunks(paper))
    migrated_session.commit()
    kw = {"dense": FakeDense(), "sparse": FakeSparse(), "client": qdrant, "collection": collection}
    upsert_paper_chunks(migrated_session, ID, **kw)  # type: ignore[arg-type]
    paper.github_url = "https://github.com/invented/repo"
    migrated_session.commit()
    dense = FakeDense()
    kw["dense"] = dense
    assert upsert_paper_chunks(migrated_session, ID, **kw) == 0  # type: ignore[arg-type]
    assert dense.calls == []
    assert all((p.payload or {})["has_code"] is True for p in _points(qdrant, collection))


def test_changed_chunk_text_is_embedded_again(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    paper = _paper(migrated_session)
    chunks = _chunks(paper)
    save_chunks(migrated_session, ID, chunks)
    migrated_session.commit()
    kw = {"dense": FakeDense(), "sparse": FakeSparse(), "client": qdrant, "collection": collection}
    upsert_paper_chunks(migrated_session, ID, **kw)  # type: ignore[arg-type]
    row = _rows(migrated_session)[1]
    row.text = row.text + " An invented extra sentence."
    migrated_session.commit()
    dense = FakeDense()
    kw["dense"] = dense
    assert upsert_paper_chunks(migrated_session, ID, **kw) == 1  # type: ignore[arg-type]
    [point] = qdrant.retrieve(collection, ids=[str(row.id)], with_payload=True)
    assert (point.payload or {})["text"].endswith("An invented extra sentence.")


def test_version_bump_removes_stale_points_and_rows_but_keeps_analysis(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    paper = _paper(migrated_session)
    v1 = _chunks(paper, "hf_style.md")  # abstract + 5 sections
    save_chunks(migrated_session, ID, v1)
    migrated_session.commit()
    analysis_id = _seed_analysis(migrated_session, qdrant, collection)
    upsert_paper_chunks(
        migrated_session,
        ID,
        dense=FakeDense(),
        sparse=FakeSparse(),
        client=qdrant,
        collection=collection,
    )
    migrated_session.commit()

    # v2 arrives with a shorter text: abstract + 3 sections.
    paper.version = 2
    migrated_session.commit()
    v2 = _chunks(paper, "empty_sections.md")
    assert len(v2) == 4
    save_chunks(migrated_session, ID, v2)
    migrated_session.commit()
    upsert_paper_chunks(
        migrated_session,
        ID,
        dense=FakeDense(),
        sparse=FakeSparse(),
        client=qdrant,
        collection=collection,
    )
    migrated_session.commit()

    rows = _rows(migrated_session)
    points = _points(qdrant, collection)
    assert [r.id for r in rows] == [c.id for c in v2]  # stale v1 rows (ords 4, 5) deleted
    assert {str(p.id) for p in _of_types(points, "abstract", "section")} == {str(c.id) for c in v2}
    assert qdrant.retrieve(collection, ids=[str(c.id) for c in v1]) == []  # no v1 point left
    # analysis survives in both stores
    assert [r.id for r in _rows(migrated_session, types=("analysis",))] == [analysis_id]
    assert [str(p.id) for p in _of_types(points, "analysis")] == [str(analysis_id)]
    assert len(rows) == len(_of_types(points, "abstract", "section")) == 4


def test_unknown_paper_is_an_error(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    with pytest.raises(ValueError, match="no paper"):
        upsert_paper_chunks(
            migrated_session,
            "2601.99999",
            dense=FakeDense(),
            sparse=FakeSparse(),
            client=qdrant,
            collection=collection,
        )


# --- index_abstracts ---------------------------------------------------------------------------


def test_index_abstracts_skips_already_indexed_papers(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    for i in range(3):
        _paper(migrated_session, f"2601.0000{i}")
    dense = FakeDense()
    kw = {"dense": dense, "sparse": FakeSparse(), "client": qdrant, "collection": collection}
    assert index_abstracts(migrated_session, **kw) == 3  # type: ignore[arg-type]
    assert len(dense.calls) == 1 and len(dense.calls[0][1]) == 3
    assert index_abstracts(migrated_session, **kw) == 0  # type: ignore[arg-type]
    assert len(dense.calls) == 1  # second run: no embedding call at all
    rows = migrated_session.scalar(select(func.count()).select_from(ChunkRow))
    assert rows == qdrant.count(collection, exact=True).count == 3

    # Only the given ids are considered.
    _paper(migrated_session, "2601.00009")
    assert index_abstracts(migrated_session, ["2601.00009"], **kw) == 1  # type: ignore[arg-type]


def test_index_abstracts_replaces_the_abstract_of_an_older_version(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    paper = _paper(migrated_session)
    kw = {"dense": FakeDense(), "sparse": FakeSparse(), "client": qdrant, "collection": collection}
    index_abstracts(migrated_session, **kw)  # type: ignore[arg-type]
    paper.version, paper.abstract = 2, "A revised invented abstract."
    migrated_session.commit()
    assert index_abstracts(migrated_session, **kw) == 1  # type: ignore[arg-type]
    [row] = _rows(migrated_session)
    [point] = _points(qdrant, collection)
    assert row.id == chunk_id(ID, 2, 0) and str(point.id) == str(row.id)
    assert (point.payload or {})["text"] == "A revised invented abstract."


# --- llm_usage rows from the real DenseEmbedder (fake OpenAI client) ------------------------------


def test_each_embedding_request_writes_one_llm_usage_row(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    from tests.unit.test_embed import FakeClient, FakeEmbeddings

    for i in range(150):
        _paper(migrated_session, f"2601.{i:05d}")
    dense = DenseEmbedder(FakeClient(FakeEmbeddings()), session_recorder(migrated_session))
    index_abstracts(
        migrated_session, dense=dense, sparse=FakeSparse(), client=qdrant, collection=collection
    )
    migrated_session.commit()
    usage = migrated_session.scalars(select(LLMUsage).order_by(LLMUsage.id)).all()
    assert [(u.feature, u.model, u.tokens_in, u.tokens_out) for u in usage] == [
        ("embed-abstract", "text-embedding-3-small", 1000, 0),  # 100 inputs x 10 fake tokens
        ("embed-abstract", "text-embedding-3-small", 500, 0),
    ]
    assert usage[0].cost_usd == embedding_cost(1000) == Decimal("0.000020")

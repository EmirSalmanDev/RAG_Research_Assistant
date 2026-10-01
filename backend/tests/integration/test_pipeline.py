"""run_paper end to end on a throwaway database and a `chunks_test_<random>` collection,
with fake source clients, fake full-text sources and fake embedders (no network, no
OpenAI). The real collection is checked to be untouched after every test."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk as ChunkRow
from app.db.models import Paper, PaperText
from app.ingest.chunk import FIRST_ANALYSIS_ORD, chunk_id
from app.ingest.fulltext import FulltextError, FulltextSource
from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection
from app.ingest.pipeline import PaperRunResult, run_paper
from tests.integration.test_fulltext import HF_MD, LONG_HTML, FakeSources
from tests.integration.test_indexing import DIM, FakeDense, FakeSparse
from tests.integration.test_ingest_one import ID, FakeArxiv, _arxiv_paper, _clients

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]


@pytest.fixture
def collection(qdrant: QdrantClient, scratch_collection: str) -> str:
    ensure_collection(qdrant, scratch_collection, DIM)
    return scratch_collection


def _run(
    session: Session,
    qdrant: QdrantClient,
    collection: str,
    sources: FakeSources,
    dense: FakeDense | None = None,
    force_source: FulltextSource | None = None,
) -> PaperRunResult:
    return run_paper(
        session,
        ID,
        clients=_clients(),
        sources=sources,
        dense=dense or FakeDense(),
        sparse=FakeSparse(),
        force_source=force_source,
        qdrant=qdrant,
        collection=collection,
    )


def _row_count(session: Session, *types: str) -> int:
    query = select(func.count()).select_from(ChunkRow).where(ChunkRow.arxiv_id == ID)
    if types:
        query = query.where(ChunkRow.chunk_type.in_(types))
    return session.scalar(query) or 0


def _point_count(qdrant: QdrantClient, collection: str, *types: str) -> int:
    must = [m.FieldCondition(key="arxiv_id", match=m.MatchValue(value=ID))]
    if types:
        must.append(m.FieldCondition(key="chunk_type", match=m.MatchAny(any=list(types))))
    return qdrant.count(collection, count_filter=m.Filter(must=must), exact=True).count


def test_run_twice_gives_the_same_counts_and_no_second_download_or_embedding(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    sources = FakeSources(hf=HF_MD, html=LONG_HTML)
    first = _run(migrated_session, qdrant, collection, sources)
    assert (first.fulltext_status, first.fulltext_source) == ("indexed", "hf")
    assert first.chunks["abstract"] == 1 and first.chunks["section"] >= 2
    assert first.embedded == sum(first.chunks.values())
    assert _row_count(migrated_session) == _point_count(qdrant, collection)

    sources.calls.clear()
    dense = FakeDense()
    second = _run(migrated_session, qdrant, collection, sources, dense=dense)
    assert second.fulltext_reused and sources.calls == []  # stored text reused
    assert second.embedded == 0 and dense.calls == []  # nothing embedded again
    assert second.chunks == first.chunks
    assert _row_count(migrated_session) == _point_count(qdrant, collection)

    text = migrated_session.get(PaperText, ID)
    assert text is not None and text.markdown.startswith("## 1 Introduction")
    assert text.references is not None and text.references.startswith("[1] A. Example")
    assert text.token_count > 0


def test_an_analysis_chunk_survives_re_runs(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    _run(migrated_session, qdrant, collection, FakeSources(hf=HF_MD))
    analysis_id = chunk_id(ID, 2, FIRST_ANALYSIS_ORD)
    migrated_session.add(
        ChunkRow(id=analysis_id, arxiv_id=ID, chunk_type="analysis", section_path="Analysis",
                 ord=FIRST_ANALYSIS_ORD, text="An invented analysis.", token_count=5)
    )  # fmt: skip
    migrated_session.commit()
    vector = {DENSE_VECTOR: [1.0] * DIM, SPARSE_VECTOR: m.SparseVector(indices=[1], values=[1.0])}
    qdrant.upsert(
        collection,
        points=[m.PointStruct(id=str(analysis_id), vector=vector,
                              payload={"arxiv_id": ID, "chunk_type": "analysis"})],
        wait=True,
    )  # fmt: skip
    # A forced different source re-fetches and re-chunks the paper.
    result = _run(
        migrated_session, qdrant, collection, FakeSources(html=LONG_HTML), force_source="html"
    )
    assert result.fulltext_source == "html" and not result.fulltext_reused
    assert _row_count(migrated_session, "analysis") == 1
    assert _point_count(qdrant, collection, "analysis") == 1
    assert _row_count(migrated_session, "abstract", "section") == _point_count(
        qdrant, collection, "abstract", "section"
    )


def test_no_full_text_leaves_the_abstract_searchable(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    result = _run(migrated_session, qdrant, collection, FakeSources())
    assert result.fulltext_status == "unavailable" and result.chunks == {"abstract": 1}
    paper = migrated_session.get(Paper, ID)
    assert paper is not None and paper.fulltext_status == "unavailable"
    assert _point_count(qdrant, collection) == 1


def test_a_failure_marks_the_paper_failed_and_a_new_run_resets_it(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    broken = FakeSources(hf=RuntimeError("HF is down"))
    with pytest.raises(FulltextError):
        _run(migrated_session, qdrant, collection, broken)
    paper = migrated_session.get(Paper, ID)
    assert paper is not None and paper.fulltext_status == "failed"
    result = _run(migrated_session, qdrant, collection, FakeSources(hf=HF_MD))  # manual re-run
    assert result.fulltext_status == "indexed"


def test_a_new_arxiv_version_replaces_the_old_chunks(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    _run(migrated_session, qdrant, collection, FakeSources(hf=HF_MD))
    old_ids = [str(i) for i in migrated_session.scalars(select(ChunkRow.id))]
    v3 = _arxiv_paper()
    v3.version = 3
    clients = _clients()
    clients.arxiv = FakeArxiv(v3)
    sources = FakeSources(hf=HF_MD)
    result = run_paper(
        migrated_session, ID, clients=clients, sources=sources, dense=FakeDense(),
        sparse=FakeSparse(), qdrant=qdrant, collection=collection,
    )  # fmt: skip
    assert not result.fulltext_reused and sources.calls == ["hf"]  # v3 text fetched
    assert qdrant.retrieve(collection, ids=old_ids) == []  # no v2 point left
    rows = migrated_session.scalars(select(ChunkRow)).all()
    assert all(r.id == chunk_id(ID, 3, r.ord) for r in rows)
    assert _row_count(migrated_session) == _point_count(qdrant, collection)

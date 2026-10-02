"""The search stage's Qdrant request, checked with a fake client (no Qdrant, no OpenAI)."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m
from qdrant_client.http import models as rest

from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR
from app.rag.search import QUERY_FEATURE, SearchFilters, SearchMode, build_filter, search

CHUNK = uuid.UUID("0214d8ce-b010-59e2-b4d6-06b4e659c8d5")
PAYLOAD = {
    "arxiv_id": "2601.00001",
    "chunk_type": "section",
    "section_path": "3 Method",
    "title": "An Invented Paper",
    "published_at": "2026-01-02T09:30:00+00:00",
    "categories": ["cs.CL"],
    "text": "Invented chunk text.",
}


class FakeClient:
    """Records each query_points call and answers with one fixed point."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query_points(self, collection_name: str, **kwargs: Any) -> rest.QueryResponse:
        self.calls.append((collection_name, kwargs))
        point = m.ScoredPoint(id=str(CHUNK), version=0, score=0.75, payload=PAYLOAD)
        return rest.QueryResponse(points=[point])  # what query_points returns


class FakeDense:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.calls.append((list(texts), feature))
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeSparse:
    """Stands in for BM25: a question of only stopwords gets an empty vector."""

    def embed(self, texts: Sequence[str]) -> list[m.SparseVector]:
        return [
            m.SparseVector(indices=[], values=[])
            if t == "the of and"
            else m.SparseVector(indices=[7, 42], values=[0.5, 1.0])
            for t in texts
        ]


FILTERS = SearchFilters(categories=("cs.CL",))


def _search(
    mode: SearchMode, query: str = "what is low-rank adaptation", **kw: Any
) -> tuple[FakeClient, FakeDense, list[Any]]:
    client, dense = FakeClient(), FakeDense()
    hits = search(
        query,
        kw.pop("filters", FILTERS),
        mode=mode,
        limit=10,
        prefetch_limit=50,
        rrf_k=60,
        dense=dense,
        sparse=FakeSparse(),
        client=cast(QdrantClient, client),
        collection="chunks_unit",
        **kw,
    )
    return client, dense, hits


def test_hybrid_is_one_call_with_two_filtered_prefetches_fused_by_rrf() -> None:
    client, dense, _ = _search("hybrid")
    assert len(client.calls) == 1
    collection, kw = client.calls[0]
    assert collection == "chunks_unit"
    dense_pf, sparse_pf = kw["prefetch"]
    assert (dense_pf.using, dense_pf.limit, dense_pf.query) == (DENSE_VECTOR, 50, [0.1, 0.2, 0.3])
    assert (sparse_pf.using, sparse_pf.limit) == (SPARSE_VECTOR, 50)
    assert sparse_pf.query == m.SparseVector(indices=[7, 42], values=[0.5, 1.0])
    # The same filter on BOTH prefetches: BM25 cannot bring back a filtered-out chunk.
    assert dense_pf.filter == sparse_pf.filter == build_filter(FILTERS)
    assert kw["query"] == m.RrfQuery(rrf=m.Rrf(k=60))
    assert kw["limit"] == 10
    assert dense.calls == [(["what is low-rank adaptation"], QUERY_FEATURE)]


def test_dense_mode_queries_the_dense_vector_only() -> None:
    client, dense, _ = _search("dense")
    _, kw = client.calls[0]
    assert (kw["using"], kw["query"], kw["limit"]) == (DENSE_VECTOR, [0.1, 0.2, 0.3], 10)
    assert kw["query_filter"] == build_filter(FILTERS)
    assert "prefetch" not in kw
    assert len(dense.calls) == 1


def test_bm25_mode_makes_no_embedding_request() -> None:
    client, dense, _ = _search("bm25")
    _, kw = client.calls[0]
    assert kw["using"] == SPARSE_VECTOR
    assert kw["query"] == m.SparseVector(indices=[7, 42], values=[0.5, 1.0])
    assert dense.calls == []  # free: no OpenAI request


def test_points_become_hits() -> None:
    _, _, hits = _search("bm25")
    (hit,) = hits
    assert hit.chunk_id == CHUNK
    assert (hit.arxiv_id, hit.title, hit.section_path) == (
        "2601.00001",
        "An Invented Paper",
        "3 Method",
    )
    assert (hit.chunk_type, hit.text, hit.score) == ("section", "Invented chunk text.", 0.75)


def test_stopword_only_question_in_bm25_mode_returns_nothing_without_a_query() -> None:
    client, _, hits = _search("bm25", query="the of and")
    assert hits == [] and client.calls == []


def test_stopword_only_question_in_hybrid_mode_falls_back_to_dense() -> None:
    client, _, _ = _search("hybrid", query="the of and")
    _, kw = client.calls[0]
    assert kw["using"] == DENSE_VECTOR and "prefetch" not in kw


@pytest.mark.parametrize(
    ("mode", "query", "message"),
    [("fuzzy", "a question", "unknown search mode"), ("dense", "   ", "empty question")],
)
def test_bad_input_fails_before_any_call(mode: Any, query: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _search(mode, query=query)


def test_no_filters_means_no_filter() -> None:
    assert build_filter(SearchFilters()) is None
    client, _, _ = _search("dense", filters=None)
    assert client.calls[0][1]["query_filter"] is None


def test_every_filter_field_becomes_a_must_condition() -> None:
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 31, tzinfo=UTC)
    flt = build_filter(
        SearchFilters(
            arxiv_ids=("2601.00001", "2601.00002"),
            categories=("cs.CL", "cs.IR"),
            published_from=start,
            published_to=end,
        )
    )
    assert flt == m.Filter(
        must=[
            m.FieldCondition(key="arxiv_id", match=m.MatchAny(any=["2601.00001", "2601.00002"])),
            m.FieldCondition(key="categories", match=m.MatchAny(any=["cs.CL", "cs.IR"])),
            m.FieldCondition(key="published_at", range=m.DatetimeRange(gte=start, lte=end)),
        ]
    )


def test_an_open_ended_date_range_sets_one_bound() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    flt = build_filter(SearchFilters(published_from=start))
    assert flt is not None and flt.must == [
        m.FieldCondition(key="published_at", range=m.DatetimeRange(gte=start, lte=None))
    ]

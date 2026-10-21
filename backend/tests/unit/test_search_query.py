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
from app.rag.search import (
    QUERY_FEATURE,
    SearchFilters,
    SearchMode,
    build_filter,
    complete_ties,
    search,
)

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
    assert kw["limit"] == 11  # one more than needed, to see a tie at the cut
    assert dense.calls == [(["what is low-rank adaptation"], QUERY_FEATURE)]


def test_dense_mode_queries_the_dense_vector_only() -> None:
    client, dense, _ = _search("dense")
    _, kw = client.calls[0]
    assert (kw["using"], kw["query"], kw["limit"]) == (DENSE_VECTOR, [0.1, 0.2, 0.3], 11)
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
    [
        ("fuzzy", "a question", "unknown search mode"),
        ("dense", "   ", "empty question"),
        ("dense", "a question", "limit must be at least 1"),
    ],
)
def test_bad_input_fails_before_any_call(mode: Any, query: str, message: str) -> None:
    limit = 0 if "limit" in message else 10
    client, dense = FakeClient(), FakeDense()
    with pytest.raises(ValueError, match=message):
        search(
            query, None, mode=mode, limit=limit, prefetch_limit=50, rrf_k=60, dense=dense,
            sparse=FakeSparse(), client=cast(QdrantClient, client), collection="chunks_unit",
        )  # fmt: skip
    assert client.calls == [] and dense.calls == []


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


# --- ties: a fixed order, also at the cut ------------------------------------------------------


def _uuid(n: int) -> uuid.UUID:
    return uuid.UUID(int=n)


class PoolClient:
    """Answers query_points from a fixed pool: sorted by score, but equal scores in the
    order given (Qdrant promises no order for them). Records each requested limit."""

    def __init__(self, pool: list[tuple[int, float]]) -> None:
        self.pool = pool
        self.limits: list[int] = []

    def query_points(self, collection_name: str, **kwargs: Any) -> rest.QueryResponse:
        self.limits.append(kwargs["limit"])
        ranked = sorted(self.pool, key=lambda p: -p[1])  # stable: ties keep the pool's order
        points = [
            m.ScoredPoint(id=str(_uuid(n)), version=0, score=score, payload=PAYLOAD)
            for n, score in ranked[: kwargs["limit"]]
        ]
        return rest.QueryResponse(points=points)


def _pool_search(pool: list[tuple[int, float]], limit: int, mode: SearchMode = "hybrid") -> Any:
    client, dense = PoolClient(pool), FakeDense()
    hits = search(
        "a question", None, mode=mode, limit=limit, prefetch_limit=50, rrf_k=60, dense=dense,
        sparse=FakeSparse(), client=cast(QdrantClient, client), collection="chunks_unit",
    )  # fmt: skip
    return [h.chunk_id for h in hits], client.limits, dense.calls


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_equal_scores_come_out_in_chunk_id_order_whatever_qdrant_returns(mode: Any) -> None:
    """An exact RRF tie (positions 1 and 2 in one list, 2 and 1 in the other: both
    1/61 + 1/62 with k = 60), returned by Qdrant in either order."""
    tie = 1 / 61 + 1 / 62
    forward = [(1, 0.5), (5, tie), (3, tie), (9, 0.01)]
    backward = [(1, 0.5), (3, tie), (5, tie), (9, 0.01)]
    out_forward, _, _ = _pool_search(forward, 4, mode)
    out_backward, _, _ = _pool_search(backward, 4, mode)
    assert out_forward == out_backward == [_uuid(1), _uuid(3), _uuid(5), _uuid(9)]


def test_a_tie_at_the_cut_is_settled_by_the_same_rule() -> None:
    """Points 7 and 2 tie for the last place; Qdrant's order alone would keep 7. The
    extra point shows the tie, a second fetch finds the end of the tied group, and the
    smaller id wins."""
    pool = [(1, 0.9), (7, 0.5), (2, 0.5), (4, 0.1)]
    out, limits, dense_calls = _pool_search(pool, 2)
    assert out == [_uuid(1), _uuid(2)]
    assert limits == [3, 6]  # 3: the 3rd point ties the 2nd; 6: only 4 points exist
    assert len(dense_calls) == 1  # the question is embedded once for both fetches


def test_a_long_tied_group_is_fetched_until_it_ends() -> None:
    pool = [(1, 0.9), *[(n, 0.5) for n in (80, 60, 50, 30, 20, 70, 40)], (99, 0.1)]
    out, limits, _ = _pool_search(pool, 2)
    assert out == [_uuid(1), _uuid(20)]  # the smallest id of the seven tied points
    assert limits == [3, 6, 12]  # 6 points all tie at the end; 12 > the 9 in the pool


def test_without_a_tie_at_the_cut_one_request_is_enough() -> None:
    pool = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6)]
    out, limits, _ = _pool_search(pool, 2)
    assert (out, limits) == ([_uuid(1), _uuid(2)], [3])


def test_complete_ties_stops_when_the_pool_runs_out() -> None:
    points = [m.ScoredPoint(id=str(_uuid(n)), version=0, score=1.0) for n in (3, 1, 2)]
    sizes: list[int] = []

    def fetch(size: int) -> list[m.ScoredPoint]:
        sizes.append(size)
        return points[:size]

    assert [str(p.id) for p in complete_ties(fetch, 5)] == [str(_uuid(n)) for n in (1, 2, 3)]
    assert sizes == [6]

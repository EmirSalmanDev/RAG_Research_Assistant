"""Search stage: find candidate chunks for a question in Qdrant (docs/adr/0005).

One `query_points` call per question, in one of three modes:
- dense:  the chunks whose OpenAI embedding is closest to the question's (`text-dense`).
          Finds paraphrases ("model compression" ~ "distillation"), but can miss exact
          names and numbers. Costs one embedding request per question.
- bm25:   the chunks that share the question's rarer words (`text-sparse`). Finds exact
          terms ("LoRA", "Llama-3"), misses synonyms. Local and free.
- hybrid: both lists (`prefetch_limit` candidates each), merged by Reciprocal Rank
          Fusion (RRF) inside Qdrant, so the strengths of both add up.

RRF in short: every chunk gets 1 / (k + position) from each list it appears in
(position 0 = the top), and the sums are sorted. Only positions count, not raw scores,
which is what lets it merge cosine similarities and BM25 scores (different scales). A
large k (60) flattens the curve, so a chunk found by both lists beats one that is first
in only one list; Qdrant's default k = 2 rewards the very top of each list much more.

Filters run inside Qdrant on indexed payload fields, before candidates are counted. In
hybrid mode the same filter is put on each prefetch: the BM25 list can never bring back
a chunk the filter removed from the dense list.

Ties (docs/adr/0012): Qdrant returns equal scores in no fixed order. In hybrid mode exact
ties are common (dense #1 + BM25 #2 scores exactly what dense #2 + BM25 #1 scores), and
identical runs swapped such chunks. So every result list is put in a fixed order here:
best score first and, among equal scores, the smaller chunk id first. A tie at the cut
is settled by the same rule (see `complete_ties`).
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from qdrant_client import QdrantClient
from qdrant_client import models as m

from app.config import get_settings
from app.ingest.embed import DenseEncoder, SparseEncoder
from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant

SearchMode = Literal["dense", "bm25", "hybrid"]
SEARCH_MODES: tuple[SearchMode, ...] = ("dense", "bm25", "hybrid")
QUERY_FEATURE = "embed-query"  # llm_usage.feature of a search question's embedding


@dataclass(frozen=True)
class SearchFilters:
    """Metadata filters. Empty fields filter nothing; set fields must all hold (AND)."""

    arxiv_ids: tuple[str, ...] = ()  # only these papers
    categories: tuple[str, ...] = ()  # papers listed in ANY of these categories
    published_from: datetime | None = None  # inclusive; timezone-aware
    published_to: datetime | None = None  # inclusive; timezone-aware


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk, with the payload fields a caller needs to show and cite it."""

    chunk_id: uuid.UUID
    arxiv_id: str
    title: str
    section_path: str | None
    chunk_type: str
    published_at: str  # RFC 3339, as stored in the payload
    text: str
    score: float  # cosine, BM25 or RRF score here; the reranker replaces it with its own


def build_filter(filters: SearchFilters) -> m.Filter | None:
    """Turn SearchFilters into a Qdrant filter (None when nothing is filtered).

    Every condition goes in `must` (all must hold). `MatchAny` on the list field
    `categories` matches a point if any of its categories is in the given list.
    """
    must: list[m.Condition] = []
    if filters.arxiv_ids:
        must.append(m.FieldCondition(key="arxiv_id", match=m.MatchAny(any=list(filters.arxiv_ids))))
    if filters.categories:
        must.append(
            m.FieldCondition(key="categories", match=m.MatchAny(any=list(filters.categories)))
        )
    if filters.published_from is not None or filters.published_to is not None:
        dates = m.DatetimeRange(gte=filters.published_from, lte=filters.published_to)
        must.append(m.FieldCondition(key="published_at", range=dates))
    return m.Filter(must=must) if must else None


def search(
    query: str,
    filters: SearchFilters | None = None,
    *,
    mode: SearchMode,
    limit: int,
    prefetch_limit: int,
    rrf_k: int,
    dense: DenseEncoder | None = None,
    sparse: SparseEncoder | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> list[SearchHit]:
    """The best `limit` chunks for `query`, best first (equal scores: smaller chunk id
    first).

    `dense` is needed in the dense and hybrid modes (it embeds the question, a paid
    request logged as 'embed-query'); `sparse` in the bm25 and hybrid modes. The question
    is embedded once, even when a tie at the cut makes Qdrant be asked a second time.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
    if not query.strip():
        raise ValueError("cannot search for an empty question")
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    client = client or get_qdrant()
    collection = collection or get_settings().qdrant_collection
    flt = build_filter(filters or SearchFilters())

    dense_vector = _embed_dense(query, dense) if mode in ("dense", "hybrid") else None
    sparse_vector = _embed_sparse(query, sparse) if mode in ("bm25", "hybrid") else None
    # A question made only of stopwords has no BM25 terms: there is nothing to match.
    if sparse_vector is not None and not sparse_vector.indices:
        if mode == "bm25":
            return []
        sparse_vector = None  # hybrid falls back to the dense list alone

    fetch: Callable[[int], list[m.ScoredPoint]]
    if dense_vector is not None and sparse_vector is not None:
        # Hybrid: Qdrant runs both prefetches, then fuses the two ranked lists with RRF.
        n = prefetch_limit
        prefetch = [
            m.Prefetch(query=dense_vector, using=DENSE_VECTOR, filter=flt, limit=n),
            m.Prefetch(query=sparse_vector, using=SPARSE_VECTOR, filter=flt, limit=n),
        ]

        def fetch(size: int) -> list[m.ScoredPoint]:
            return client.query_points(
                collection,
                prefetch=prefetch,
                query=m.RrfQuery(rrf=m.Rrf(k=rrf_k)),
                limit=size,
                with_payload=True,
            ).points

    else:
        # One list only: plain nearest-neighbour search on one named vector.
        vector, using = (
            (dense_vector, DENSE_VECTOR)
            if dense_vector is not None
            else (sparse_vector, SPARSE_VECTOR)
        )

        def fetch(size: int) -> list[m.ScoredPoint]:
            return client.query_points(
                collection, query=vector, using=using, query_filter=flt, limit=size,
                with_payload=True,
            ).points  # fmt: skip

    return [_hit(point) for point in complete_ties(fetch, limit)]


def complete_ties(fetch: Callable[[int], list[m.ScoredPoint]], limit: int) -> list[m.ScoredPoint]:
    """The best `limit` points in a fixed order: score (best first), then point id.

    `fetch(n)` returns Qdrant's best n points, sorted by score but with ties in any
    order. Sorting them by (score, id) fixes the order inside the list; the cut needs
    more care: if the point just past the cut has the same score as the last one kept,
    Qdrant alone chose which of them made the list. So one point more than needed is
    fetched, and while the last point fetched still has the score of the last one kept
    (the tied group may go on beyond it) the fetch is repeated with twice the size.
    Then every point with that score is in hand, and the rule picks among them.

    Exact ties are rare outside hybrid mode, so this almost always costs one request
    for `limit + 1` points.
    """
    size = limit + 1
    while True:
        points = list(fetch(size))
        if len(points) < size or points[-1].score != points[limit - 1].score:
            break  # the result set is exhausted, or the tied group ends inside it
        size *= 2
    ordered = sorted(points, key=lambda p: (-p.score, uuid.UUID(str(p.id))))
    return ordered[:limit]


def _embed_dense(query: str, dense: DenseEncoder | None) -> list[float]:
    if dense is None:
        raise ValueError("this search mode needs a dense encoder")
    return dense.embed([query], QUERY_FEATURE)[0]


def _embed_sparse(query: str, sparse: SparseEncoder | None) -> m.SparseVector:
    if sparse is None:
        raise ValueError("this search mode needs a sparse (BM25) encoder")
    return sparse.embed([query])[0]


def _hit(point: m.ScoredPoint) -> SearchHit:
    p: dict[str, Any] = point.payload or {}
    return SearchHit(
        chunk_id=uuid.UUID(str(point.id)),
        arxiv_id=p["arxiv_id"],
        title=p["title"],
        section_path=p.get("section_path"),
        chunk_type=p["chunk_type"],
        published_at=p["published_at"],
        text=p["text"],
        score=point.score,
    )

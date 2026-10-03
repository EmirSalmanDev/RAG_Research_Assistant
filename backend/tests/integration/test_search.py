"""The search stage against a real Qdrant, in a `chunks_test_<random>` collection.

Dense vectors are 4-dimensional and hand-made (no OpenAI); BM25 vectors come from the
real local BM25 encoder. All chunk texts are invented. The real collection is checked to
be untouched after every test.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m

from app.ingest.embed import get_sparse_embedder
from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection
from app.rag.search import SearchFilters, SearchHit, SearchMode, search

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

DIM = 4
# (id, arxiv_id, categories, published, dense vector, text)
POINTS = [
    ("a", "2601.00001", ["cs.CL"], datetime(2026, 1, 10, tzinfo=UTC), [1.0, 0.0, 0.0, 0.0],
     "Low-rank adaptation (LoRA) freezes the weights and trains small update matrices."),
    ("b", "2601.00002", ["cs.LG"], datetime(2026, 2, 10, tzinfo=UTC), [0.0, 1.0, 0.0, 0.0],
     "A mixture of experts routes each token to two experts."),
    ("c", "2601.00003", ["cs.CL", "cs.IR"], datetime(2026, 3, 10, tzinfo=UTC), [0.0, 0.0, 1.0, 0.0],
     "Retrieval augmented generation adds retrieved passages to the prompt."),
    ("d", "2601.00003", ["cs.CL", "cs.IR"], datetime(2026, 3, 10, tzinfo=UTC), [0.0, 0.0, 0.9, 0.1],
     "Dense retrieval embeds questions and passages in one vector space."),
]  # fmt: skip
IDS = {name: uuid.uuid5(uuid.NAMESPACE_URL, name) for name, *_ in POINTS}


class FakeDense:
    """Returns the hand-made vector registered for each question."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        return [self.vectors[t] for t in texts]


@pytest.fixture
def collection(qdrant: QdrantClient, scratch_collection: str) -> str:
    ensure_collection(qdrant, scratch_collection, DIM)
    sparse = get_sparse_embedder().embed([text for *_, text in POINTS])
    qdrant.upsert(
        scratch_collection,
        points=[
            m.PointStruct(
                id=str(IDS[name]),
                vector={DENSE_VECTOR: vector, SPARSE_VECTOR: sv},
                payload={
                    "arxiv_id": arxiv_id,
                    "chunk_type": "section",
                    "section_path": "1 Introduction",
                    "title": f"Invented Paper {arxiv_id}",
                    "published_at": published.isoformat(),
                    "categories": categories,
                    "text": text,
                },
            )
            for (name, arxiv_id, categories, published, vector, text), sv in zip(
                POINTS, sparse, strict=True
            )
        ],
        wait=True,
    )
    return scratch_collection


def _search(
    qdrant: QdrantClient,
    collection: str,
    mode: SearchMode,
    query: str,
    vector: list[float] | None = None,
    filters: SearchFilters | None = None,
) -> list[SearchHit]:
    return search(
        query,
        filters,
        mode=mode,
        limit=10,
        prefetch_limit=50,
        rrf_k=60,
        dense=FakeDense({query: vector or [0.0, 0.0, 0.0, 1.0]}),
        sparse=get_sparse_embedder(),
        client=qdrant,
        collection=collection,
    )


def _names(hits: list[SearchHit]) -> list[str]:
    by_id = {v: k for k, v in IDS.items()}
    return [by_id[h.chunk_id] for h in hits]


def test_dense_mode_ranks_by_vector_similarity(qdrant: QdrantClient, collection: str) -> None:
    hits = _search(qdrant, collection, "dense", "q", vector=[0.9, 0.1, 0.0, 0.0])
    assert _names(hits)[:2] == ["a", "b"]
    assert hits[0].arxiv_id == "2601.00001" and hits[0].text.startswith("Low-rank")


def test_bm25_mode_only_returns_chunks_sharing_words(qdrant: QdrantClient, collection: str) -> None:
    assert _names(_search(qdrant, collection, "bm25", "LoRA low-rank adaptation")) == ["a"]


def test_bm25_stopword_question_returns_nothing(qdrant: QdrantClient, collection: str) -> None:
    assert _search(qdrant, collection, "bm25", "the of and") == []


def test_hybrid_puts_the_chunk_first_in_both_lists_on_top(
    qdrant: QdrantClient, collection: str
) -> None:
    hits = _search(
        qdrant, collection, "hybrid", "retrieval augmented generation", [0.0, 0.0, 1.0, 0.0]
    )
    assert _names(hits)[:2] == ["c", "d"]  # c: first in both lists; d: second in both
    # Qdrant's RRF: 1 / (k + position) per list, position 0 at the top. With k=60, a
    # chunk first in both lists scores 2/60; Qdrant's default k=2 would give 2/2.
    assert hits[0].score == pytest.approx(2 / 60)


def test_hybrid_filter_applies_to_both_lists(qdrant: QdrantClient, collection: str) -> None:
    # c would be first in BOTH lists; the filter must remove it from both.
    only_a_and_b = SearchFilters(arxiv_ids=("2601.00001", "2601.00002"))
    hits = _search(
        qdrant, collection, "hybrid", "retrieval augmented generation", [0.0, 0.0, 1.0, 0.0],
        filters=only_a_and_b,
    )  # fmt: skip
    assert set(_names(hits)) == {"a", "b"}


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        (SearchFilters(categories=("cs.IR",)), {"c", "d"}),
        (SearchFilters(categories=("cs.LG", "cs.IR")), {"b", "c", "d"}),
        (SearchFilters(published_from=datetime(2026, 2, 1, tzinfo=UTC)), {"b", "c", "d"}),
        (
            SearchFilters(
                published_from=datetime(2026, 2, 1, tzinfo=UTC),
                published_to=datetime(2026, 2, 28, tzinfo=UTC),
            ),
            {"b"},
        ),
        (SearchFilters(arxiv_ids=("2601.00003",), categories=("cs.CL",)), {"c", "d"}),
    ],
)
def test_filters_run_inside_qdrant(
    qdrant: QdrantClient, collection: str, filters: SearchFilters, expected: set[str]
) -> None:
    hits = _search(qdrant, collection, "dense", "q", filters=filters)
    assert set(_names(hits)) == expected

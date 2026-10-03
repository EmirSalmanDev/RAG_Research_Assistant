"""The retrieval pipeline: search, then rerank and cap, each switched by config.

`GET /search` and the retrieval evaluation both run this, so what is measured is what
the API serves. The evaluation compares configurations by changing one field at a time
(`dataclasses.replace(config, mode="bm25", rerank=False)`).
"""

from dataclasses import dataclass

from qdrant_client import QdrantClient

from app.config import get_settings
from app.ingest.embed import DenseEncoder, SparseEncoder, get_dense_embedder, get_sparse_embedder
from app.rag.rerank import CrossEncoder, cap_per_paper, get_cross_encoder, rerank
from app.rag.search import SearchFilters, SearchHit, SearchMode, search


@dataclass(frozen=True)
class RetrievalConfig:
    mode: SearchMode  # dense | bm25 | hybrid
    prefetch_limit: int  # candidates from each list in hybrid mode
    rrf_k: int  # RRF constant
    candidates: int  # results from search, i.e. what the reranker rescores
    rerank: bool  # rerank stage on/off
    rerank_batch_size: int
    max_chunks_per_paper: int  # diversity cap; 0 = off

    @classmethod
    def from_settings(cls) -> "RetrievalConfig":
        s = get_settings()
        return cls(
            mode=s.search_mode,
            prefetch_limit=s.search_prefetch_limit,
            rrf_k=s.search_rrf_k,
            candidates=s.search_candidates,
            rerank=s.rerank_enabled,
            rerank_batch_size=s.rerank_batch_size,
            max_chunks_per_paper=s.max_chunks_per_paper,
        )


def retrieve(
    query: str,
    filters: SearchFilters | None = None,
    *,
    limit: int,
    config: RetrievalConfig,
    dense: DenseEncoder | None = None,
    sparse: SparseEncoder | None = None,
    cross_encoder: CrossEncoder | None = None,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> list[SearchHit]:
    """The best `limit` chunks for `query`.

    1. search: `candidates` chunks (at least `limit`) in the configured mode;
    2. rerank (if on): the cross-encoder reorders them;
    3. cap (if on): at most `max_chunks_per_paper` chunks per paper;
    4. the first `limit` are returned.

    Encoders default to the process-wide ones, and only for the stages that run: the
    bm25 mode never builds the OpenAI client, rerank off never loads the model.
    """
    if config.mode != "bm25":
        dense = dense or get_dense_embedder()
    if config.mode != "dense":
        sparse = sparse or get_sparse_embedder()
    hits = search(
        query,
        filters,
        mode=config.mode,
        limit=max(config.candidates, limit),
        prefetch_limit=config.prefetch_limit,
        rrf_k=config.rrf_k,
        dense=dense,
        sparse=sparse,
        client=client,
        collection=collection,
    )
    if config.rerank:
        encoder = cross_encoder or get_cross_encoder()
        hits = rerank(query, hits, encoder, batch_size=config.rerank_batch_size)
    hits = cap_per_paper(hits, config.max_chunks_per_paper)
    return hits[:limit]

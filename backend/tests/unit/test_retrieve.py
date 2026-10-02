"""The retrieval pipeline's stage switches, with search replaced by a fake (no stores)."""

import uuid
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import pytest

from app.config import get_settings
from app.rag import retrieve as retrieve_module
from app.rag.retrieve import RetrievalConfig, retrieve
from app.rag.search import SearchHit

CONFIG = RetrievalConfig(
    mode="hybrid", prefetch_limit=50, rrf_k=60, candidates=6, rerank=True,
    rerank_batch_size=16, max_chunks_per_paper=2,
)  # fmt: skip


def _hit(name: str, arxiv_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, name), arxiv_id=arxiv_id, title="T",
        section_path=None, chunk_type="section", published_at="2026-01-02T00:00:00+00:00",
        text=name, score=0.0,
    )  # fmt: skip


# Search order: three chunks of paper A first, then B and C.
SEARCH_HITS = [_hit("a1", "A"), _hit("a2", "A"), _hit("a3", "A"), _hit("b1", "B"), _hit("c1", "C")]


class ReverseCrossEncoder:
    """Scores the search order backwards: the last candidate becomes the best."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
        self.calls += 1
        return [float(i) for i, _ in enumerate(documents)]


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_search(query: str, filters: Any, **kw: Any) -> list[SearchHit]:
        calls.append({"query": query, "filters": filters, **kw})
        return list(SEARCH_HITS)

    monkeypatch.setattr(retrieve_module, "search", fake_search)
    return calls


@pytest.fixture(autouse=True)
def no_default_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that forgets to pass a fake fails loudly instead of loading a real model."""

    def forbidden() -> Any:
        raise AssertionError("a real model or client was requested")

    for name in ("get_dense_embedder", "get_sparse_embedder", "get_cross_encoder"):
        monkeypatch.setattr(retrieve_module, name, forbidden)


def _names(hits: list[SearchHit]) -> list[str]:
    return [h.text for h in hits]


STANDIN: Any = object()  # an encoder the fake search never calls


def _retrieve(config: RetrievalConfig, limit: int = 10, **kw: Any) -> list[SearchHit]:
    kw.setdefault("dense", STANDIN)
    kw.setdefault("sparse", STANDIN)
    return retrieve("q", None, limit=limit, config=config, **kw)


def test_search_gets_the_configured_mode_and_candidate_counts(
    searches: list[dict[str, Any]],
) -> None:
    _retrieve(replace(CONFIG, rerank=False), limit=3)
    (call,) = searches
    assert (call["mode"], call["limit"], call["prefetch_limit"], call["rrf_k"]) == (
        "hybrid", 6, 50, 60,
    )  # fmt: skip


def test_candidates_never_fall_below_the_requested_limit(searches: list[dict[str, Any]]) -> None:
    _retrieve(replace(CONFIG, rerank=False), limit=20)
    assert searches[0]["limit"] == 20


def test_rerank_off_keeps_search_order_and_never_touches_the_model(
    searches: list[dict[str, Any]],
) -> None:
    out = _retrieve(replace(CONFIG, rerank=False))
    assert _names(out) == ["a1", "a2", "b1", "c1"]  # a3 dropped by the cap of 2


def test_rerank_on_reorders_before_the_cap(searches: list[dict[str, Any]]) -> None:
    encoder = ReverseCrossEncoder()
    out = _retrieve(CONFIG, cross_encoder=encoder)
    assert encoder.calls == 1
    # Reversed: c1 b1 a3 a2 a1; the cap of 2 per paper then drops a1.
    assert _names(out) == ["c1", "b1", "a3", "a2"]


def test_cap_zero_keeps_every_chunk(searches: list[dict[str, Any]]) -> None:
    out = _retrieve(replace(CONFIG, rerank=False, max_chunks_per_paper=0))
    assert _names(out) == ["a1", "a2", "a3", "b1", "c1"]


def test_limit_cuts_after_rerank_and_cap(searches: list[dict[str, Any]]) -> None:
    out = _retrieve(CONFIG, limit=2, cross_encoder=ReverseCrossEncoder())
    assert _names(out) == ["c1", "b1"]


def test_bm25_mode_needs_no_dense_encoder(searches: list[dict[str, Any]]) -> None:
    config = replace(CONFIG, mode="bm25", rerank=False)
    out = retrieve("q", None, limit=3, config=config, sparse=STANDIN)
    assert searches[0]["dense"] is None and len(out) == 3


def test_dense_mode_needs_no_bm25_encoder(searches: list[dict[str, Any]]) -> None:
    config = replace(CONFIG, mode="dense", rerank=False)
    retrieve("q", None, limit=3, config=config, dense=STANDIN)
    assert searches[0]["sparse"] is None


def test_config_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARCH_MODE", "bm25")
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("MAX_CHUNKS_PER_PAPER", "0")
    get_settings.cache_clear()
    config = RetrievalConfig.from_settings()
    assert (config.mode, config.rerank, config.max_chunks_per_paper) == ("bm25", False, 0)
    assert (config.prefetch_limit, config.rrf_k, config.candidates) == (50, 60, 50)
    get_settings.cache_clear()

"""The rerank stage with a fake cross-encoder, the per-paper cap, and offline-only loading.

No real model is ever loaded or downloaded here: the cache tests use invented files, and
the session-wide network guard (tests/conftest.py) fails any test that tries the network.
"""

import threading
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from app.ingest.chunk import embedding_text
from app.rag import rerank as rerank_module
from app.rag.rerank import (
    RerankerUnavailable,
    cap_per_paper,
    fetch,
    get_cross_encoder,
    load_cross_encoder,
    rerank,
)
from app.rag.search import SearchHit

MODEL = "BAAI/bge-reranker-base"


def _hit(
    name: str, arxiv_id: str = "2601.00001", score: float = 0.0, chunk_type: str = "section"
) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, name),
        arxiv_id=arxiv_id,
        title=f"Invented {arxiv_id}",
        section_path="2 Method",
        chunk_type=chunk_type,
        published_at="2026-01-02T00:00:00+00:00",
        text=f"text of {name}",
        score=score,
    )


class FakeCrossEncoder:
    """Scores each document by a fixed table; records what it was asked."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> Iterable[float]:
        docs = list(documents)
        self.calls.append((query, docs, batch_size))
        return [next(v for k, v in self.scores.items() if doc.endswith(k)) for doc in docs]


# --- rerank -----------------------------------------------------------------------------------


def test_rerank_orders_by_cross_encoder_score_and_replaces_scores() -> None:
    hits = [_hit("x", score=0.9), _hit("y", score=0.8), _hit("z", score=0.7)]
    encoder = FakeCrossEncoder({"text of x": -2.0, "text of y": 5.0, "text of z": 1.5})
    out = rerank("what is it?", hits, encoder, batch_size=16)
    assert [h.text for h in out] == ["text of y", "text of z", "text of x"]
    assert [h.score for h in out] == [5.0, 1.5, -2.0]


def test_rerank_reads_title_and_section_with_the_text() -> None:
    hit = _hit("x")
    encoder = FakeCrossEncoder({"text of x": 1.0})
    rerank("q", [hit], encoder, batch_size=8)
    assert encoder.calls == [("q", [embedding_text(hit.title, hit.section_path, hit.text)], 8)]


def test_equal_scores_keep_the_search_order() -> None:
    hits = [_hit("x"), _hit("y"), _hit("z")]
    out = rerank("q", hits, FakeCrossEncoder({"of x": 1.0, "of y": 1.0, "of z": 1.0}), batch_size=4)
    assert [h.text for h in out] == ["text of x", "text of y", "text of z"]


def test_no_hits_means_no_model_call() -> None:
    encoder = FakeCrossEncoder({})
    assert rerank("q", [], encoder, batch_size=4) == [] and encoder.calls == []


def test_a_wrong_number_of_scores_is_an_error() -> None:
    class Short:
        def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
            return [1.0]

    with pytest.raises(RuntimeError, match="1 scores for 2 chunks"):
        rerank("q", [_hit("x"), _hit("y")], Short(), batch_size=4)


# --- cap_per_paper ----------------------------------------------------------------------------


def test_cap_keeps_the_best_chunks_of_each_paper_in_order() -> None:
    a1, a2, b1, a3, a4, c1 = (
        _hit("a1", "A"), _hit("a2", "A"), _hit("b1", "B"), _hit("a3", "A"), _hit("a4", "A"),
        _hit("c1", "C"),
    )  # fmt: skip
    for analysis in (1, -1):  # no analysis chunk here: both rules give the same list
        assert cap_per_paper([a1, a2, b1, a3, a4, c1], 3, max_analysis_per_paper=analysis) == [
            a1, a2, b1, a3, c1,
        ]  # fmt: skip
        assert cap_per_paper([a1, a2, b1, a3], 1, max_analysis_per_paper=analysis) == [a1, b1]


def test_cap_zero_switches_it_off() -> None:
    hits = [_hit(str(i), "A") for i in range(5)]
    assert cap_per_paper(hits, 0, max_analysis_per_paper=1) == hits


def _analysis(name: str, arxiv_id: str = "A") -> SearchHit:
    return _hit(name, arxiv_id, chunk_type="analysis")


def test_analysis_chunks_never_take_a_slot_of_the_papers_own_text() -> None:
    x1, x2, s1, s2, s3, s4 = (
        _analysis("x1"), _analysis("x2"), _hit("s1", "A"), _hit("s2", "A"), _hit("s3", "A"),
        _hit("s4", "A"),
    )  # fmt: skip
    # Three sections of A survive, as without analysis chunks, plus A's best analysis chunk.
    assert cap_per_paper([x1, x2, s1, s2, s3, s4], 3, max_analysis_per_paper=1) == [
        x1, s1, s2, s3,
    ]  # fmt: skip


def test_each_paper_keeps_its_own_best_analysis_chunks_in_order() -> None:
    xa1, sb1, xa2, xb1, xb2, sa1 = (
        _analysis("xa1", "A"), _hit("sb1", "B"), _analysis("xa2", "A"), _analysis("xb1", "B"),
        _analysis("xb2", "B"), _hit("sa1", "A"),
    )  # fmt: skip
    hits = [xa1, sb1, xa2, xb1, xb2, sa1]
    assert cap_per_paper(hits, 3, max_analysis_per_paper=1) == [xa1, sb1, xb1, sa1]
    assert cap_per_paper(hits, 3, max_analysis_per_paper=2) == [xa1, sb1, xa2, xb1, xb2, sa1]


def test_an_analysis_limit_of_zero_leaves_them_out() -> None:
    x1, s1, x2 = _analysis("x1"), _hit("s1", "A"), _analysis("x2", "B")
    assert cap_per_paper([x1, s1, x2], 3, max_analysis_per_paper=0) == [s1]
    assert cap_per_paper([x1, s1, x2], 0, max_analysis_per_paper=0) == [s1]


def test_a_negative_analysis_limit_counts_them_like_any_chunk() -> None:
    """The behaviour before the separate count: analysis chunks fill the paper's cap."""
    x1, x2, s1, s2 = _analysis("x1"), _analysis("x2"), _hit("s1", "A"), _hit("s2", "A")
    assert cap_per_paper([x1, x2, s1, s2], 3, max_analysis_per_paper=-1) == [x1, x2, s1]
    assert cap_per_paper([x1, x2, s1, s2], 0, max_analysis_per_paper=-1) == [x1, x2, s1, s2]


def test_the_analysis_limit_holds_when_the_cap_is_off() -> None:
    x1, x2, s1, s2, s3, s4 = (
        _analysis("x1"), _analysis("x2"), _hit("s1", "A"), _hit("s2", "A"), _hit("s3", "A"),
        _hit("s4", "A"),
    )  # fmt: skip
    assert cap_per_paper([s1, x1, s2, x2, s3, s4], 0, max_analysis_per_paper=1) == [
        s1, x1, s2, s3, s4,
    ]  # fmt: skip


# --- loading: offline only --------------------------------------------------------------------

SNAPSHOT = "0" * 40
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _invented_cache(root: Path, *, with_onnx: bool) -> None:
    """The Hugging Face cache layout FastEmbed reads, with invented (unusable) files."""
    repo = root / "models--BAAI--bge-reranker-base"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(SNAPSHOT)
    snapshot = repo / "snapshots" / SNAPSHOT
    (snapshot / "onnx").mkdir(parents=True)
    for name in TOKENIZER_FILES:
        (snapshot / name).write_text("{}")
    if with_onnx:
        (snapshot / "onnx" / "model.onnx").write_bytes(b"invented, not a model")


def test_missing_model_is_a_clear_error_without_network(tmp_path: Path) -> None:
    with pytest.raises(RerankerUnavailable, match="make fetch-reranker"):
        load_cross_encoder(MODEL, tmp_path)


def test_incomplete_cache_is_a_clear_error_without_network(tmp_path: Path) -> None:
    _invented_cache(tmp_path, with_onnx=False)
    with pytest.raises(RerankerUnavailable):
        load_cross_encoder(MODEL, tmp_path)


def test_fastembed_finds_a_complete_cache_without_network(tmp_path: Path) -> None:
    """Contract test for a FastEmbed upgrade: its cache check for this model must not go
    online (BM25's does). lazy_load=True stops before the invented files are parsed, and
    local_files_only is left at its default to prove the cache alone is enough."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    _invented_cache(tmp_path, with_onnx=True)
    TextCrossEncoder(MODEL, cache_dir=str(tmp_path), lazy_load=True)  # no NetworkBlockedError


def test_model_is_loaded_once_even_under_concurrent_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str] = []

    def slow_load(model_name: str, cache_dir: Path, **runtime: Any) -> FakeCrossEncoder:
        loads.append(model_name)
        time.sleep(0.05)  # long enough for the other threads to arrive meanwhile
        return FakeCrossEncoder({})

    monkeypatch.setattr(rerank_module, "_loaded", None)
    monkeypatch.setattr(rerank_module, "load_cross_encoder", slow_load)
    results: list[object] = []
    threads = [
        threading.Thread(target=lambda: results.append(get_cross_encoder())) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loads == [MODEL]
    assert len({id(r) for r in results}) == 1


def _recording_text_cross_encoder(
    monkeypatch: pytest.MonkeyPatch, created: list[dict[str, Any]]
) -> None:
    class FakeTextCrossEncoder:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            created.append({"model": model_name, **kwargs})

    import fastembed.rerank.cross_encoder as fastembed_cross_encoder

    monkeypatch.setattr(fastembed_cross_encoder, "TextCrossEncoder", FakeTextCrossEncoder)


def test_the_default_load_passes_no_runtime_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exactly the call measured in docs/adr/0005: no threads, no arena option."""
    created: list[dict[str, Any]] = []
    _recording_text_cross_encoder(monkeypatch, created)
    load_cross_encoder(MODEL, tmp_path)
    assert created == [{"model": MODEL, "cache_dir": str(tmp_path), "local_files_only": True}]


def test_a_thread_limit_and_the_arena_switch_reach_onnx_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[dict[str, Any]] = []
    _recording_text_cross_encoder(monkeypatch, created)
    load_cross_encoder(MODEL, tmp_path, threads=2, cpu_arena=False)
    assert created == [
        {"model": MODEL, "cache_dir": str(tmp_path), "local_files_only": True, "threads": 2,
         "enable_cpu_mem_arena": False}
    ]  # fmt: skip
    with pytest.raises(ValueError, match="at least 1"):
        load_cross_encoder(MODEL, tmp_path, threads=0)


def test_other_reranker_settings_release_the_old_model_and_load_the_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An evaluation runs several configurations in one process: each must get the model
    its settings name, never the one loaded for the configuration before."""
    from app.config import get_settings

    loads: list[tuple[str, int | None, bool]] = []
    holding_at_load: list[object] = []

    def load(model_name: str, cache_dir: Path, *, threads: int | None, cpu_arena: bool) -> Any:
        holding_at_load.append(rerank_module._loaded)  # what was still held at that moment
        loads.append((model_name, threads, cpu_arena))
        return FakeCrossEncoder({})

    monkeypatch.setattr(rerank_module, "_loaded", None)
    monkeypatch.setattr(rerank_module, "load_cross_encoder", load)
    first = get_cross_encoder()
    assert get_cross_encoder() is first  # same settings: the same model
    monkeypatch.setenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")
    monkeypatch.setenv("RERANK_THREADS", "2")
    get_settings.cache_clear()
    second = get_cross_encoder()
    assert second is not first
    assert loads == [(MODEL, None, True), ("Xenova/ms-marco-MiniLM-L-12-v2", 2, True)]
    assert holding_at_load == [None, None]  # the old model was let go before the new load
    key = rerank_module.loaded_reranker()
    assert key is not None and (key.model, key.threads) == ("Xenova/ms-marco-MiniLM-L-12-v2", 2)


# --- fetch (the only download) ----------------------------------------------------------------


def test_fetch_downloads_without_loading_and_lists_the_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[dict[str, Any]] = []

    class FakeTextCrossEncoder:
        """Stands in for the real download: writes the invented cache instead."""

        def __init__(self, model_name: str, **kwargs: Any) -> None:
            created.append({"model": model_name, **kwargs})
            _invented_cache(Path(kwargs["cache_dir"]), with_onnx=True)

        @staticmethod
        def list_supported_models() -> list[dict[str, Any]]:
            return [{"model": MODEL}]

    import fastembed.rerank.cross_encoder as fastembed_cross_encoder

    monkeypatch.setattr(fastembed_cross_encoder, "TextCrossEncoder", FakeTextCrossEncoder)
    summary = fetch(MODEL, tmp_path)
    # Downloads (local_files_only not set) but does not load the model into RAM.
    assert created == [{"model": MODEL, "cache_dir": str(tmp_path), "lazy_load": True}]
    files = summary["files"]
    assert isinstance(files, dict)
    assert files[f"snapshots/{SNAPSHOT}/onnx/model.onnx"] == len(b"invented, not a model")
    assert summary["total_bytes"] == sum(files.values())


def test_fetch_rejects_an_unknown_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a FastEmbed cross-encoder"):
        fetch("BAAI/bge-reranker-v2-m3", tmp_path)

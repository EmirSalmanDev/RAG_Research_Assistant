"""Pieces of eval/run.py that need no stores: the vector cache, scoring, the table."""

import json
import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from app.config import get_settings
from app.ingest.tokens import count_tokens
from app.llm.usage import embedding_cost
from eval.golden import GoldenItem
from eval.run import (
    CONFIGURATIONS,
    CachedQueryEncoder,
    QueryVectorCache,
    build_configs,
    embedding_estimate,
    main,
    percentile,
    results_table,
    score,
)


@pytest.mark.parametrize("search_default", ["false", "true"])
def test_eval_configurations_ignore_the_search_rerank_default(
    monkeypatch: pytest.MonkeyPatch, search_default: str
) -> None:
    """GET /search reranks only on request (RERANK_ENABLED); the evaluation must not
    follow that default: hybrid-rerank always reranks, the others never do, and all four
    keep the configured 50 candidates."""
    monkeypatch.setenv("RERANK_ENABLED", search_default)
    get_settings.cache_clear()
    configs = build_configs(list(CONFIGURATIONS))
    get_settings.cache_clear()
    assert {name: c.rerank for name, c in configs.items()} == {
        "dense": False, "bm25": False, "hybrid": False, "hybrid-rerank": True,
    }  # fmt: skip
    assert {c.candidates for c in configs.values()} == {50}


def test_cache_round_trip_and_atomic_save(tmp_path: Path) -> None:
    path = tmp_path / ".cache" / "vectors.json"
    cache = QueryVectorCache(path, "embed-model", 3)
    assert cache.missing(["a?", "b?"]) == ["a?", "b?"]
    cache.put("a?", [0.1, 0.2, 0.3])
    cache.save()
    reloaded = QueryVectorCache(path, "embed-model", 3)
    assert reloaded.get("a?") == [0.1, 0.2, 0.3]
    assert reloaded.missing(["a?", "b?"]) == ["b?"]
    assert not path.with_suffix(".tmp").exists()
    assert "a?" not in path.read_text()  # keys are hashes, not the question text


def test_cache_key_includes_model_and_dimension(tmp_path: Path) -> None:
    path = tmp_path / "vectors.json"
    cache = QueryVectorCache(path, "embed-model", 3)
    cache.put("a?", [0.1, 0.2, 0.3])
    cache.save()
    assert QueryVectorCache(path, "other-model", 3).get("a?") is None
    assert QueryVectorCache(path, "embed-model", 1536).get("a?") is None
    assert len(json.loads(path.read_text())) == 1


def test_cached_encoder_never_embeds(tmp_path: Path) -> None:
    cache = QueryVectorCache(tmp_path / "v.json", "embed-model", 2)
    cache.put("known?", [1.0, 0.0])
    encoder = CachedQueryEncoder(cache)
    assert encoder.embed(["known?"], "any-feature") == [[1.0, 0.0]]
    with pytest.raises(RuntimeError, match="no cached embedding"):
        encoder.embed(["unknown?"], "any-feature")


def test_embedding_estimate_counts_tokens_locally() -> None:
    questions = ["What does LoRA freeze?", "Which experts does the router pick?"]
    tokens, requests, cost = embedding_estimate(questions)
    assert tokens == sum(count_tokens(q) for q in questions)
    assert requests == 1  # 2 questions fit one batch of 100
    assert Decimal(cost) == embedding_cost(tokens)


def test_score_uses_the_reference_chunks() -> None:
    a, b, c = (uuid.uuid5(uuid.NAMESPACE_URL, n) for n in "abc")
    item = GoldenItem(id="q01", question="q?", reference_chunk_ids=[b])
    assert score([a, b, c], item) == {
        "recall@8": 1.0,
        "mrr": 0.5,
        "ndcg@10": pytest.approx(1 / 1.584962500721156),  # 1 / log2(3)
    }


def test_percentile_is_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 50) == 20.0
    assert percentile(values, 95) == 40.0
    assert percentile([7.0], 95) == 7.0


def test_results_table_marks_missing_values() -> None:
    metrics = {"recall@8": 0.5, "mrr": None, "ndcg@10": 0.25, "n_scored": 2,
               "latency_ms_p50": 3.5, "latency_ms_p95": 9.0}  # fmt: skip
    table = results_table([(1, "bm25", metrics)])
    assert table.splitlines()[-1] == "| 1 | bm25 | 0.500 | n/a | 0.250 | 2 | 3.500 | 9.000 |"


def test_unknown_configuration_is_rejected_before_any_store_is_touched() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--configs", "dense,colbert"])
    assert exit_info.value.code == 2

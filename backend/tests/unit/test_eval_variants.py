"""eval/variants.py: the plan's 8 ablations and the reranker comparisons as configurations."""

import pytest

from app.config import get_settings
from eval.variants import (
    STEPS,
    IndexSpec,
    all_variants,
    find,
    index_overrides,
    resolve,
    resolved_from_record,
    variants_of,
)


def test_every_planned_ablation_is_a_variant_and_every_step_has_one_incumbent() -> None:
    s = get_settings()
    assert sorted({v.ablation for v in all_variants(s) if v.ablation}) == [1, 2, 3, 4, 5, 6, 7, 8]
    for step in STEPS:
        assert [v.id for v in variants_of(step, s) if v.incumbent] != []
        assert len([v for v in variants_of(step, s) if v.incumbent]) == 1
    ids = [v.id for v in all_variants(s)]
    assert len(ids) == len(set(ids))


def test_the_incumbents_are_what_the_app_runs_today() -> None:
    s = get_settings()
    base = resolve(find("a3-hybrid", s), s)
    assert base.overrides(s) == {"search_mode": "hybrid", "rerank_enabled": False}
    assert resolve(find("a7-c512", s), s, base).index == IndexSpec.configured(s)
    assert resolve(find("a8-small", s), s, base).overrides(s) == base.overrides(s)


def test_model_names_come_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_SECOND_RERANKER_MODEL", "jinaai/jina-reranker-v1-turbo-en")
    monkeypatch.setenv("EVAL_LARGE_EMBEDDING_MODEL", "invented-large")
    get_settings.cache_clear()
    s = get_settings()
    assert (
        find("a5-rerank-second", s).overrides["reranker_model"]
        == "jinaai/jina-reranker-v1-turbo-en"
    )
    assert find("a8-large", s).index["embedding_model"] == "invented-large"


def test_an_index_variant_has_its_own_collection_model_dimension_and_price() -> None:
    s = get_settings()
    large = resolve(find("a8-large-short", s), s)
    assert large.index.request_dimensions == s.embedding_dim
    over = large.overrides(s)
    assert over["qdrant_collection"] == (
        f"{s.qdrant_collection}__eval__c512o64__text-embedding-3-large-1536-short"
    )
    assert over["embedding_model"] == s.eval_large_embedding_model
    assert over["embedding_price_per_1m"] == s.eval_large_embedding_price_per_1m
    assert "embedding_dim" not in over  # shortened to the configured size
    full = resolve(find("a8-large", s), s).overrides(s)
    assert full["embedding_dim"] == s.eval_large_embedding_dim
    chunked = resolve(find("a7-c1024", s), s).overrides(s)
    assert chunked["chunk_tokens"] == 1024 and chunked["chunk_overlap"] == 128
    assert chunked["qdrant_collection"].endswith("__c1024o128__text-embedding-3-small-1536")
    assert index_overrides(IndexSpec.configured(s), s) == {}


def test_a_later_step_builds_on_the_winner_and_a_stored_record_round_trips() -> None:
    s = get_settings()
    winner = resolve(find("a4-rerank", s), s)
    chunked = resolve(find("a7-c256", s), s, resolve(find("a6-no-rewrite", s), s, winner))
    assert chunked.settings["rerank_enabled"] is True
    assert chunked.settings["plan_rephrasings_enabled"] is False
    assert chunked.index.chunk_tokens == 256
    embedded = resolve(find("a8-large", s), s, chunked)
    assert (embedded.index.chunk_tokens, embedded.index.embedding_dim) == (256, 3072)
    again = resolved_from_record(embedded.describe(), s)
    assert again.overrides(s) == embedded.overrides(s) and again.base == "a7-c256"


def test_runtime_only_reranker_variants_need_no_paid_answer_run() -> None:
    s = get_settings()
    parts = {v.id: v.parts for v in variants_of("reranker", s)}
    assert parts["r-threads-1"] == ("retrieval",) and parts["r-arena-off"] == ("retrieval",)
    assert parts["r-candidates-10"] == ("retrieval", "answers")
    assert parts["r-small-model"] == ("retrieval", "answers")
    with pytest.raises(KeyError):
        find("a9-unknown", s)

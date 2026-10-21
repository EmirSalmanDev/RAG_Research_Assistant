"""The ablation runner's core (eval/runner.py) on fakes: no network, no stores, no model.

The main risk it guards against: the pipeline reads the process-wide settings, so several
configurations run in one process could silently run the configuration before them. The
tests run different configurations one after another in one process and check, for each,
what the stages were actually given (collection, mode, embedding model, reranker), what
was stored as its configuration, and that the settings are back afterwards."""

import os
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import get_settings
from app.egress_guard import Guard
from app.rag.rerank import RerankerKey, model_cache_dir
from eval.budget import Budget, BudgetExhausted, policy
from eval.golden import SetQuestion
from eval.run import QueryVectorCache
from eval.runner import (
    ConfigMismatch,
    IndexMissing,
    MissingQueryVectors,
    Observed,
    RunnerDeps,
    Suite,
    run_part,
    verify_observed,
)
from eval.variants import Resolved, find, resolve
from tests.eval_fakes import (
    ABS,
    SEC,
    FakeJudge,
    FakeOpenAIEmbeddings,
    FakeRelevancyEmbed,
    FakeRetrieve,
    FakeSearch,
    KeyedCrossEncoders,
    MemoryStore,
    auto_chat,
    question,
)

QUESTIONS = [question("gen-001"), question("gen-002"), question("q01", "heldout")]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.01
        return self.now


@dataclass
class FakeQdrant:
    existing: set[str] | None = None  # None: every collection exists, with 5 points

    def collection_exists(self, name: str) -> bool:
        return self.existing is None or name in self.existing

    def count(self, name: str, exact: bool = True) -> Any:
        return SimpleNamespace(count=5)


@dataclass
class World:
    """Every fake one test uses, and the deps built on them."""

    tmp: Path
    store: MemoryStore = field(default_factory=MemoryStore)
    search: FakeSearch = field(default_factory=lambda: FakeSearch([SEC, ABS]))
    retrieve: FakeRetrieve = field(default_factory=lambda: FakeRetrieve([SEC, ABS]))
    encoders: KeyedCrossEncoders = field(default_factory=KeyedCrossEncoders)
    embeddings: FakeOpenAIEmbeddings = field(default_factory=FakeOpenAIEmbeddings)
    qdrant: FakeQdrant = field(default_factory=FakeQdrant)
    budget: Budget | None = None

    def deps(self) -> RunnerDeps:
        return RunnerDeps(
            store=self.store, chat_client=auto_chat, judge=FakeJudge,
            relevancy_embedder=FakeRelevancyEmbed, search_fn=self.search,
            retrieve_fn=self.retrieve, sparse=lambda: None,  # type: ignore[arg-type,return-value]
            cross_encoder=self.encoders.get, loaded_reranker=self.encoders.loaded,
            openai_client=lambda: self.embeddings,
            record_embedding=lambda f, m, t: None, qdrant=lambda: self.qdrant,  # type: ignore[arg-type,return-value]
            query_cache_path=self.tmp / "vectors.json", budget=self.budget,
            git_sha=lambda: "invented-sha", clock=Clock(), peak_rss=lambda: 123.0,
        )  # fmt: skip

    def cache_vectors(self, model: str, dim: int) -> None:
        cache = QueryVectorCache(self.tmp / "vectors.json", model, dim)
        for q in QUESTIONS:
            cache.put(q.item.question, [0.1] * dim)
        cache.save()


def _variant(variant_id: str, base: str | None = None) -> Resolved:
    s = get_settings()
    return resolve(find(variant_id, s), s, resolve(find(base, s), s) if base else None)


SUITE = Suite("ablation", "test")


def test_configurations_run_in_one_process_never_leak_into_each_other(tmp_path: Path) -> None:
    s = get_settings()
    world = World(tmp_path)
    large = s.eval_large_embedding_model
    world.cache_vectors(s.embedding_model, s.embedding_dim)
    world.cache_vectors(large, s.eval_large_embedding_dim)
    deps = world.deps()
    bm25 = _variant("a2-bm25")
    large_rerank = _variant("a8-large", base="a4-rerank")  # another index, another model
    small_reranker = _variant("a5-rerank-second")
    plan = [bm25, large_rerank, small_reranker, bm25]  # and back to the first

    runs: list[
        tuple[Resolved, int, int, int, int]
    ] = []  # variant, run id, search/retrieve/emb marks
    for resolved in plan:
        marks = (len(world.search.calls), len(world.retrieve.calls), len(world.embeddings.requests))
        retrieval = run_part("retrieval", resolved, QUESTIONS, deps, SUITE)
        answers = run_part("answers", resolved, QUESTIONS, deps, SUITE)
        assert (retrieval.status, answers.status) == ("complete", "complete")
        runs.append((resolved, answers.run_id, *marks))

    large_collection = f"{s.qdrant_collection}__eval__c512o64__text-embedding-3-large-3072"
    expect = {
        "a2-bm25": ("bm25", s.qdrant_collection, None, None),
        "a8-large": ("hybrid", large_collection, (large, s.eval_large_embedding_dim),
                     s.reranker_model),
        "a5-rerank-second": ("hybrid", s.qdrant_collection, (s.embedding_model, s.embedding_dim),
                             s.eval_second_reranker_model),
    }  # fmt: skip
    for k, (resolved, answers_id, search_mark, retrieve_mark, emb_mark) in enumerate(runs):
        mode, collection, embedding, reranker = expect[resolved.variant.id]
        end = runs[k + 1][2:] if k + 1 < len(runs) else (None, None, None)
        searches = world.search.calls[search_mark : end[0]]
        retrieves = world.retrieve.calls[retrieve_mark : end[1]]
        requests = world.embeddings.requests[emb_mark : end[2]]
        # What the stages were given while this configuration ran:
        assert {c["mode"] for c in searches} == {mode}
        assert {c["collection_in_settings"] for c in searches} == {collection}
        assert {c["config"].mode for c in retrieves} == {mode}
        assert {c["collection"] for c in retrieves} == {collection}
        assert {(m, d) for m, d, _ in requests} == ({embedding} if embedding else set())
        # Rephrasings stay on in every one (only the retrieval stages differ).
        assert len(searches) == 3 * 3  # 3 questions x (question + 2 rephrasings)
        # What was stored as its configuration (read back inside the scope):
        effective = world.store.runs[answers_id]["config"]["effective"]
        assert effective["retrieval"]["mode"] == mode
        assert effective["collection"] == collection
        assert (effective["reranker"] or {}).get("model") == reranker
        observed = world.store.runs[answers_id]["metrics"]["observed"]
        assert observed["collections"] == [collection]
    # Each reranker was loaded when its configuration asked for it, never reused for another.
    assert [key.model for key in world.encoders.loads] == [
        s.reranker_model,
        s.eval_second_reranker_model,
    ]
    # And afterwards the process runs on its own settings again.
    after = get_settings()
    assert (after.search_mode, after.qdrant_collection, after.embedding_model,
            after.reranker_model, after.rerank_enabled) == (
        s.search_mode, s.qdrant_collection, s.embedding_model, s.reranker_model, s.rerank_enabled
    )  # fmt: skip
    assert "SEARCH_MODE" not in os.environ and "QDRANT_COLLECTION" not in os.environ


def test_a_reranker_other_than_the_settings_ask_for_stops_the_run(tmp_path: Path) -> None:
    """If the stage got another model than the configuration's (the leak this guards
    against), the run is stopped instead of storing numbers for the wrong configuration."""
    world = World(tmp_path)
    world.cache_vectors(get_settings().embedding_model, get_settings().embedding_dim)
    deps = world.deps()
    stale = RerankerKey("an-older-model", str(model_cache_dir()), None, True)
    deps.loaded_reranker = lambda: stale
    with pytest.raises(ConfigMismatch, match="rerankers"):
        run_part("retrieval", _variant("a4-rerank"), QUESTIONS, deps, SUITE)
    (run,) = world.store.runs.values()
    assert run["metrics"]["status"].startswith("stopped: ConfigMismatch")


def test_verify_observed_allows_a_subset_and_names_every_difference() -> None:
    allowed = Observed({"chunks"}, {"hybrid"}, {("m", 4)}, set())
    verify_observed(Observed({"chunks"}, set(), set(), set()), allowed)  # did less: fine
    with pytest.raises(ConfigMismatch) as err:
        verify_observed(Observed({"chunks", "other"}, {"bm25"}, {("m", 4)}, set()), allowed)
    assert "collections" in str(err.value) and "modes" in str(err.value)


def test_the_retrieval_part_scores_ids_only_under_the_configured_chunking(tmp_path: Path) -> None:
    world = World(tmp_path)
    world.cache_vectors(get_settings().embedding_model, get_settings().embedding_dim)
    deps = world.deps()
    same = run_part("retrieval", _variant("a3-hybrid"), QUESTIONS, deps, SUITE)
    other = run_part("retrieval", _variant("a7-c256"), QUESTIONS, deps, SUITE)
    first = world.store.of(same.run_id)[0]["metrics"]
    assert first["recall@8"] == 1.0 and first["mrr"] == 1.0  # SEC is the reference, ranked first
    assert first["coverage@8"] == pytest.approx(1.0)
    assert first["hits"][0]["text"] == SEC.text  # the text is kept with the result
    # Chunks of 256 tokens: the same ids mean other chunks, so only text coverage counts.
    moved = world.store.of(other.run_id)[0]["metrics"]
    assert "recall@8" not in moved and moved["coverage@8"] == pytest.approx(1.0)
    assert world.store.runs[other.run_id]["config"]["same_chunking"] is False
    assert same.metrics["generated"]["n_questions"] == 2
    assert same.metrics["heldout"]["n_questions"] == 1


def test_missing_question_vectors_stop_before_anything_is_stored(tmp_path: Path) -> None:
    world = World(tmp_path)  # empty vector cache
    with pytest.raises(MissingQueryVectors, match="3 of 3 questions"):
        run_part("retrieval", _variant("a1-dense"), QUESTIONS, world.deps(), SUITE)
    assert world.store.runs == {} and world.embeddings.requests == []


def test_an_unbuilt_variant_index_stops_before_anything_is_stored(tmp_path: Path) -> None:
    world = World(tmp_path, qdrant=FakeQdrant(existing=set()))
    with pytest.raises(IndexMissing, match="make eval-index VARIANT=a7-c1024"):
        run_part("answers", _variant("a7-c1024"), QUESTIONS, world.deps(), SUITE)
    assert world.store.runs == {}


def test_the_answers_part_stores_sources_with_texts_and_source_metrics(tmp_path: Path) -> None:
    world = World(tmp_path)
    result = run_part("answers", _variant("a4-rerank"), QUESTIONS, world.deps(), SUITE)
    stored = world.store.of(result.run_id)
    assert len(stored) == 3
    m = stored[0]["metrics"]
    assert [s["text"] for s in m["sources"]] == [SEC.text, ABS.text]
    assert m["source_recall"] == 1.0 and m["source_coverage"] == pytest.approx(1.0)
    assert m["context_recall"] == 1.0 and m["rerank_ms"] is not None
    assert result.metrics["all"]["n_source_coverage"] == 3
    assert result.metrics["reranker_load_s"] is not None
    assert world.store.runs[result.run_id]["name"] == "ablation:a4-rerank:answers"


def _budget(cap: str) -> Budget:
    return Budget(Guard(policy(Decimal(cap), 1000, get_settings()), log=lambda line: None))


def test_a_tripped_guard_stops_the_run_and_drops_the_unfinished_question(tmp_path: Path) -> None:
    world = World(tmp_path, budget=_budget("1.00"))
    assert world.budget is not None
    guard = world.budget.guard
    inner = FakeSearch([SEC, ABS])

    def search(query: str, filters: Any, **kw: Any) -> Any:
        if query == QUESTIONS[1].item.question:
            guard.tripped = "spend bound would pass the cap"  # a refused request, mid-turn
        return inner(query, filters, **kw)

    world.search = search  # type: ignore[assignment]
    with pytest.raises(BudgetExhausted):
        run_part("answers", _variant("a3-hybrid"), QUESTIONS, world.deps(), SUITE)
    (run_id,) = world.store.runs
    assert [r["metrics"]["golden_id"] for r in world.store.of(run_id)] == ["gen-001"]
    assert world.store.runs[run_id]["metrics"]["status"].startswith("stopped: budget")


def test_the_run_stops_cleanly_before_a_question_the_cap_cannot_cover(tmp_path: Path) -> None:
    world = World(tmp_path, budget=_budget("1.00"))
    assert world.budget is not None
    guard = world.budget.guard
    inner = FakeSearch([SEC, ABS])

    def search(query: str, filters: Any, **kw: Any) -> Any:
        if "rephrasing" not in query:
            guard.spend_bound_usd += Decimal("0.4")  # each question's worst case: $0.40
        return inner(query, filters, **kw)

    world.search = search  # type: ignore[assignment]
    with pytest.raises(BudgetExhausted, match="left"):
        run_part("answers", _variant("a3-hybrid"), QUESTIONS, world.deps(), SUITE)
    (run_id,) = world.store.runs
    assert len(world.store.of(run_id)) == 2  # $0.80 spent bound; $0.20 left < $0.40
    assert guard.tripped is None  # stopped before, not by a refused request


def test_the_stored_configuration_names_the_variant_suite_and_questions(tmp_path: Path) -> None:
    world = World(tmp_path)
    result = run_part("answers", _variant("a6-no-rewrite", base="a3-hybrid"), QUESTIONS,
                      world.deps(), SUITE)  # fmt: skip
    config = world.store.runs[result.run_id]["config"]
    assert config["variant"]["variant"] == "a6-no-rewrite"
    assert config["variant"]["base"] == "a3-hybrid"
    assert config["variant"]["settings"]["plan_rephrasings_enabled"] is False
    assert config["effective"]["stages"]["plan_rephrasings"] is False
    assert config["suite"]["id"] == SUITE.id
    assert config["questions"]["ids"] == ["gen-001", "gen-002", "q01"]
    assert config["judge"]["model"] == "invented-judge"
    assert len(world.search.calls) == 3  # no rephrasings: one search per question
    assert uuid.UUID(world.store.of(result.run_id)[0]["metrics"]["sources"][0]["chunk_id"])
    assert isinstance(QUESTIONS[0], SetQuestion)

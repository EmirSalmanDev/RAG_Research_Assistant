"""The ablation runner against a throwaway Postgres database: runs and results stored by
SqlRunStore, the selection rule read from them (generated questions only) and stored, the
next step building on the stored winner, and the report CI posts.

OpenAI, the judge, Qdrant and the reranker are fakes; every row is invented and the
scratch database is dropped afterwards."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, EvalRun, Paper
from eval import ablations
from eval.ablations import base_for, select_step
from eval.runner import RunnerDeps, SqlRunStore, Suite, run_part
from eval.suite import export_runs, suite_run_ids, write_report
from eval.variants import find, resolve, variants_of
from tests.eval_fakes import (
    ABS,
    SEC,
    FakeJudge,
    FakeOpenAIEmbeddings,
    FakeRelevancyEmbed,
    FakeRetrieve,
    FakeSearch,
    KeyedCrossEncoders,
    auto_chat,
    question,
)

pytestmark = pytest.mark.integration

GENERATED = [question(f"gen-{i:03d}") for i in range(1, 31)]
HELD_OUT = question("q01", "heldout", text="Invented held-out question?")
QUESTIONS = [HELD_OUT, *GENERATED]


class RefsQdrant:
    """Answers the reference check (every point exists) and the index check."""

    def retrieve(self, collection: str, ids: list[str], **kw: Any) -> list[Any]:
        return [SimpleNamespace(id=i) for i in ids]

    def collection_exists(self, name: str) -> bool:
        return True

    def count(self, name: str, exact: bool = True) -> Any:
        return SimpleNamespace(count=5)


@pytest.fixture
def stored_chunks(migrated_session: Session) -> Session:
    for arxiv_id in (SEC.arxiv_id, ABS.arxiv_id):
        migrated_session.add(Paper(
            arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", abstract="Invented abstract.",
            authors=[], categories=["cs.CL"], primary_category="cs.CL",
            published_at=datetime(2026, 9, 1, tzinfo=UTC),
        ))  # fmt: skip
    migrated_session.flush()
    for h, ord_ in ((SEC, 1), (ABS, 0)):
        migrated_session.add(Chunk(id=h.chunk_id, arxiv_id=h.arxiv_id, chunk_type=h.chunk_type,
                                   section_path=h.section_path, ord=ord_, text=h.text,
                                   token_count=12))  # fmt: skip
    migrated_session.commit()
    return migrated_session


def _deps() -> RunnerDeps:
    encoders = KeyedCrossEncoders()
    embeddings = FakeOpenAIEmbeddings()
    return RunnerDeps(
        store=SqlRunStore(qdrant=lambda: RefsQdrant()),  # type: ignore[arg-type,return-value]
        chat_client=auto_chat,
        # Dense retrieval finds every reference sentence, the others half of them.
        judge=lambda: FakeJudge(recall=1.0 if get_settings().search_mode == "dense" else 0.5),
        relevancy_embedder=FakeRelevancyEmbed, search_fn=FakeSearch([SEC, ABS]),
        retrieve_fn=FakeRetrieve([SEC, ABS]), sparse=lambda: None,  # type: ignore[arg-type,return-value]
        cross_encoder=encoders.get, loaded_reranker=encoders.loaded,
        openai_client=lambda: embeddings,
        record_embedding=lambda f, m, t: None, qdrant=lambda: RefsQdrant(),  # type: ignore[arg-type,return-value]
        git_sha=lambda: "invented-sha", peak_rss=lambda: 100.0,
    )  # fmt: skip


def test_a_step_is_run_stored_selected_on_generated_questions_and_built_on(
    stored_chunks: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    s = get_settings()
    monkeypatch.setattr(ablations, "load_question_set", lambda: QUESTIONS)
    monkeypatch.setattr(ablations, "git_commit", lambda: "invented-sha")
    deps = _deps()
    suite = Suite("ablation", "retrieval")
    for variant in variants_of("retrieval", s):
        result = run_part("answers", resolve(variant, s), QUESTIONS, deps, suite)
        assert result.status == "complete"

    with Session(stored_chunks.get_bind()) as session:
        runs = session.scalars(select(EvalRun).order_by(EvalRun.id)).all()
        assert [r.name for r in runs] == [
            f"ablation:{v.id}:answers" for v in variants_of("retrieval", s)
        ]
        assert all(r.git_sha == "invented-sha" and r.metrics["status"] == "complete" for r in runs)
        first = runs[0]
        assert first.metrics["generated"]["n_questions"] == 30
        assert first.metrics["heldout"]["n_questions"] == 1
        assert first.config["questions"]["ids"][0] == "q01"

        record = select_step(session, "retrieval", s, suite.id)
        session.commit()
    assert record["winner"] == "a1-dense"
    # 30 pairs each: the 30 generated questions; the held-out one is never paired.
    assert {c["challenger"]: c["paired"] for c in record["comparisons"]} == {
        "a1-dense": 30, "a2-bm25": 30, "a4-rerank": 30, "a5-rerank-second": 30,
    }  # fmt: skip
    assert record["comparisons"][0]["outcome"] == "better"

    with Session(stored_chunks.get_bind()) as session:
        selection = session.scalars(
            select(EvalRun).where(EvalRun.name == "selection:retrieval")
        ).one()
        assert selection.config["questions"] == "generated only (for_tuning)"
        base = base_for("rewriting", s, session, None)
    assert base is not None and base.variant.id == "a1-dense"
    assert base.overrides(s)["search_mode"] == "dense"
    rewriting = resolve(find("a6-no-rewrite", s), s, base)
    assert rewriting.overrides(s)["search_mode"] == "dense"  # the step builds on the winner

    # The report CI posts and the artifact: every run of the suite, without chunk texts.
    with Session(stored_chunks.get_bind()) as session:
        ids = suite_run_ids(session, suite.id)
        exported = export_runs(session, ids[:1])
    assert len(ids) == 5
    sources = exported[0]["results"][1]["metrics"]["sources"]
    assert sources and all("text" not in s for s in sources)
    write_report(ids[:1], tmp_path / "report")
    assert "ablation:a1-dense:answers" in (tmp_path / "report" / "report.md").read_text()
    assert SEC.text not in (tmp_path / "report" / "runs.json").read_text()
    assert json.loads((tmp_path / "report" / "runs.json").read_text())[0]["id"] == ids[0]


def test_selection_refuses_runs_of_different_question_sets(
    stored_chunks: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = get_settings()
    monkeypatch.setattr(ablations, "load_question_set", lambda: QUESTIONS)
    deps = _deps()
    suite = Suite("ablation", "retrieval")
    for variant in variants_of("retrieval", s):
        run_part("answers", resolve(variant, s), QUESTIONS, deps, suite)
    with Session(stored_chunks.get_bind()) as session:
        run = session.scalars(select(EvalRun).order_by(EvalRun.id)).first()
        assert run is not None
        run.config = {**run.config, "generated_sha256": "another file"}
        session.commit()
        with pytest.raises(ValueError, match="differ in question set"):
            select_step(session, "retrieval", s, suite.id)


def test_without_a_stored_selection_a_later_step_needs_a_base(stored_chunks: Session) -> None:
    s = get_settings()
    with Session(stored_chunks.get_bind()) as session, pytest.raises(LookupError, match="--base"):
        base_for("chunking", s, session, None)
    with Session(stored_chunks.get_bind()) as session:
        named = base_for("chunking", s, session, "a3-hybrid")
    assert named is not None and named.variant.id == "a3-hybrid"


def test_the_reranker_comparison_is_read_from_stored_runs(
    stored_chunks: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runtime-only variants are judged by identical lists, the others by their answers;
    a variant that did not run alone in its process has no peak memory of its own."""
    s = get_settings()
    monkeypatch.setattr(ablations, "load_question_set", lambda: QUESTIONS)
    from eval.run import QueryVectorCache

    cache = QueryVectorCache(tmp_path / "vectors.json", s.embedding_model, s.embedding_dim)
    for q in QUESTIONS:
        cache.put(q.item.question, [0.1] * s.embedding_dim)
    cache.save()
    deps = _deps()
    deps.query_cache_path = tmp_path / "vectors.json"
    base = base_for("reranker", s, None, None)
    # With held-out questions in the runs (speed and memory are measured over every
    # question a run took), the reranker comparison refuses to decide.
    mixed = Suite("ablation", "reranker")
    for variant in variants_of("reranker", s):
        for part in variant.parts:
            run_part(part, resolve(variant, s, base), QUESTIONS, deps, mixed)
    with Session(stored_chunks.get_bind()) as session, pytest.raises(ValueError, match="held-out"):
        select_step(session, "reranker", s, mixed.id)
    suite = Suite("ablation", "reranker")
    for variant in variants_of("reranker", s):
        for part in variant.parts:
            run_part(part, resolve(variant, s, base), GENERATED, deps, suite)
    with Session(stored_chunks.get_bind()) as session:
        record = select_step(session, "reranker", s, suite.id)
        session.commit()
    rows = {r["variant"]: r for r in record["variants"]}
    assert set(rows) == {v.id for v in variants_of("reranker", s) if not v.incumbent}
    assert not any("lists differ" in reason for reason in rows["r-threads-1"]["reasons"])
    assert any("not run alone" in reason for reason in rows["r-threads-1"]["reasons"])
    assert any("10 candidates" in reason for reason in rows["r-candidates-10"]["reasons"])
    assert record["recommended"] == "r-reference"  # nothing qualifies: keep the reference


def test_isolated_variants_share_one_cap_and_stop_when_a_bound_is_unknown(
    stored_chunks: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reranker step runs each variant in its own process; each child gets what the
    ones before left, and a child whose spend bound is unknown stops the loop."""
    monkeypatch.setattr(ablations, "load_question_set", lambda: QUESTIONS)
    caps: list[str | None] = []
    used = iter(["0.40", "0.70", None])

    def child(args: Any, variant: str, suite: Any, max_usd: Any, summary: Path) -> int:
        caps.append(None if max_usd is None else str(max_usd))
        bound = next(used, "0")
        if bound is not None:
            summary.write_text(json.dumps({"bound_usd": bound}))
        return 0

    monkeypatch.setattr(ablations, "_child", child)
    code = ablations.main(["run", "--step", "reranker", "--max-usd", "2.00"])
    assert code == ablations.EXIT_BUDGET
    assert caps == ["2.00", "1.60", "0.90"]  # then the third child's bound was unknown
    caps.clear()
    used = iter(["1.50", "0.60"])
    assert (
        ablations.main(["run", "--step", "reranker", "--max-usd", "2.00"]) == ablations.EXIT_BUDGET
    )
    assert caps == ["2.00", "0.50"]  # the cap was used up before the third variant
    assert ablations.main(["run", "--step", "reranker"]) == ablations.EXIT_SETUP  # paid: no cap


def test_the_selection_never_mixes_suites_or_corpora(
    stored_chunks: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs of two suites can stand on different bases (another --base), another corpus or
    another commit: the rule takes one suite in which every variant ran."""
    s = get_settings()
    monkeypatch.setattr(ablations, "load_question_set", lambda: QUESTIONS)
    deps = _deps()
    variants = variants_of("retrieval", s)
    first, second = Suite("ablation", "retrieval"), Suite("ablation", "retrieval")
    for variant in variants[:3]:
        run_part("answers", resolve(variant, s), QUESTIONS, deps, first)
    for variant in variants[3:]:
        run_part("answers", resolve(variant, s), QUESTIONS, deps, second)
    with Session(stored_chunks.get_bind()) as session:
        with pytest.raises(LookupError, match="in any one suite"):
            select_step(session, "retrieval", s, None)
    whole = Suite("ablation", "retrieval")
    for variant in variants:  # the whole step again, as one suite
        run_part("answers", resolve(variant, s), QUESTIONS, deps, whole)
    with Session(stored_chunks.get_bind()) as session:
        record = select_step(session, "retrieval", s, None)
        runs = session.scalars(select(EvalRun).where(EvalRun.id.in_(record["runs"].values()))).all()
        assert len({r.config["suite"]["id"] for r in runs}) == 1
        # The same step on another corpus size cannot be compared.
        runs[0].config = {**runs[0].config, "corpus": {"configured_points": 999}}
        session.commit()
        with pytest.raises(ValueError, match="base or corpus"):
            select_step(session, "retrieval", s, runs[0].config["suite"]["id"])


def test_the_report_is_written_even_when_a_run_crashes(
    stored_chunks: Session, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An error the CLI does not know (a judge API error, say) still leaves report.md for
    the pull request comment and the artifact."""
    from eval import suite as suite_module

    s = get_settings()
    monkeypatch.setattr(suite_module, "load_question_set", lambda: QUESTIONS)
    monkeypatch.setattr(suite_module, "cli_deps", lambda max_usd, settings: _deps())

    def crash(resolved: Any, questions: Any, parts: Any, deps: Any, suite: Any, **kw: Any) -> Any:
        run_part("retrieval", resolve(find("a2-bm25", s), s), questions, deps, suite)
        raise RuntimeError("an error guarded() does not know")

    monkeypatch.setattr(suite_module, "run_variants", crash)
    with pytest.raises(RuntimeError, match="does not know"):
        suite_module.main(["full", "--max-usd", "1", "--report-dir", str(tmp_path / "r")])
    assert "full:a2-bm25:retrieval" in (tmp_path / "r" / "report.md").read_text()

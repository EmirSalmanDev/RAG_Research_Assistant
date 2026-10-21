"""Storing an answer run and reading measured chat turns, in a throwaway Postgres database.

All rows are invented; nothing touches the real database (the scratch DB is dropped)."""

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import EvalItem, EvalResult, EvalRun, LLMUsage
from app.rag.citations import Source
from app.rag.search import SearchHit
from eval.answers import AnswerSample, SampleScores, TokenStats, measured_stats, store_run
from eval.golden import GoldenItem, SetQuestion

pytestmark = pytest.mark.integration

HIT = SearchHit(
    chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, "store-sec"), arxiv_id="2601.00001",
    title="Invented Paper", section_path="3 Method", chunk_type="section",
    published_at="2026-01-02T00:00:00+00:00", text="Invented text.", score=1.0,
)  # fmt: skip


def _sample(question: SetQuestion, **kw: object) -> AnswerSample:
    base: dict[str, object] = {
        "question": question, "sources": [Source(1, HIT)], "answer": "Invented answer [1].",
        "refusal": None, "error": None, "intent": "qa", "cost_usd": Decimal("0.0012"),
        "latency_ms": 900.0, "first_token_ms": 300.0,
    }  # fmt: skip
    return AnswerSample(**(base | kw))  # type: ignore[arg-type]


def test_a_run_and_one_result_per_question_are_stored(migrated_session: Session) -> None:
    golden = SetQuestion(
        GoldenItem(
            id="q01", question="Invented golden question?", reference_chunk_ids=[HIT.chunk_id]
        ),
        "heldout",
    )
    unanswerable = SetQuestion(
        GoldenItem(id="gen-001", question="Invented unanswerable?", type="unanswerable"),
        "generated",
    )
    rows = [
        (_sample(golden), SampleScores({"faithfulness": 0.5}, {"sentences": 1}, {})),
        (_sample(unanswerable, answer=None, refusal="no_sources", sources=[]), SampleScores()),
    ]
    run = store_run(
        migrated_session, "unit", rows, config={"invented": True},
        judge_cost_usd=Decimal("0.000321"), git_sha="abc123",
    )  # fmt: skip
    migrated_session.commit()

    stored = migrated_session.get(EvalRun, run.id)
    assert stored is not None
    assert (stored.name, stored.git_sha, stored.config) == (
        "answers:unit",
        "abc123",
        {"invented": True},
    )
    assert stored.metrics["judge_cost_usd"] == "0.000321"
    assert stored.metrics["all"]["faithfulness"] == 0.5
    assert stored.metrics["generated"]["refusal_accuracy"] == 1.0
    results = migrated_session.scalars(
        select(EvalResult).where(EvalResult.run_id == run.id).order_by(EvalResult.id)
    ).all()
    assert [r.metrics["golden_id"] for r in results] == ["q01", "gen-001"]
    first, second = results
    assert first.retrieved_chunk_ids == [HIT.chunk_id] and first.answer == "Invented answer [1]."
    assert (first.metrics["split"], first.metrics["faithfulness"]) == ("heldout", 0.5)
    assert first.metrics["cost_usd"] == "0.0012" and first.metrics["details"] == {"sentences": 1}
    assert (second.answer, second.metrics["refusal"], second.retrieved_chunk_ids) == (
        None, "no_sources", [],
    )  # fmt: skip
    questions = migrated_session.scalars(select(EvalItem.question).order_by(EvalItem.id)).all()
    assert questions == ["Invented golden question?", "Invented unanswerable?"]


def test_measured_stats_read_the_real_chat_turn_rows(migrated_session: Session) -> None:
    for feature, tokens_in, tokens_out in (
        ("chat-generate", 2000, 300), ("chat-generate", 3000, 400), ("chat-plan", 600, 60),
        ("chat-verify", 2200, 80), ("eval-judge-statements", 99999, 9999),
    ):  # fmt: skip
        migrated_session.add(
            LLMUsage(feature=feature, model="gpt-4o-mini", tokens_in=tokens_in,
                     tokens_out=tokens_out, cost_usd=Decimal("0.001"))
        )  # fmt: skip
    migrated_session.commit()
    stats = measured_stats(migrated_session)
    assert set(stats) == {"chat-generate", "chat-plan", "chat-verify"}  # judge rows ignored
    assert stats["chat-generate"] == TokenStats(2, 2500.0, 3000, 350.0, 400)

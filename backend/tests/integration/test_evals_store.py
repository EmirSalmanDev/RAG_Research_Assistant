"""The /evals page's store (app/api/evals.SqlEvalStore) on a throwaway Postgres database.

Every row is invented; the scratch database is dropped afterwards."""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.api.evals import SqlEvalStore, detail, needs_lookup, summary
from app.db.models import Chunk, EvalItem, EvalResult, EvalRun, Paper

pytestmark = pytest.mark.integration


def test_runs_results_and_chunk_texts_are_read_back(migrated_session: Session) -> None:
    migrated_session.add(Paper(arxiv_id="2601.00001", title="Invented paper", abstract="A.",
                               authors=[], categories=["cs.CL"], primary_category="cs.CL",
                               published_at=datetime(2026, 9, 1, tzinfo=UTC)))  # fmt: skip
    migrated_session.flush()
    chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, "store-chunk")
    migrated_session.add(Chunk(id=chunk_id, arxiv_id="2601.00001", chunk_type="section",
                               section_path="3 Method", ord=1, text="Invented chunk text.",
                               token_count=4))  # fmt: skip
    item = EvalItem(question="Invented question?", reference_chunk_ids=[chunk_id])
    older = EvalRun(name="hybrid", config={"mode": "hybrid", "collection": "chunks"},
                    metrics={"recall@8": 1.0, "n_scored": 1, "n_unscored": 0})  # fmt: skip
    newer = EvalRun(name="smoke:configured:answers", config={"kind": "answers"},
                    metrics={"status": "complete", "all": {"n_questions": 1}})  # fmt: skip
    migrated_session.add_all([item, older, newer])
    migrated_session.flush()
    migrated_session.add(EvalResult(run_id=older.id, item_id=item.id,
                                    retrieved_chunk_ids=[chunk_id],
                                    metrics={"recall@8": 1.0, "golden_id": "q01"}))  # fmt: skip
    migrated_session.commit()

    store = SqlEvalStore()
    runs = store.list_runs(10)
    assert [r.id for r in runs] == [newer.id, older.id]  # newest first
    assert store.list_runs(1)[0].id == newer.id
    assert summary(runs[1]).kind == "retrieval" and summary(runs[0]).kind == "answers"
    run = store.get_run(older.id)
    assert run is not None and store.get_run(999_999) is None
    (record,) = store.results(older.id)
    assert record.reference_chunk_ids == [chunk_id] and record.question == "Invented question?"
    assert store.result(older.id, item.id) == record and store.result(newer.id, item.id) is None
    lookup = store.chunks(needs_lookup(run, record))
    out = detail(run, record, lookup)
    assert out.retrieved[0].text == "Invented chunk text."
    assert out.retrieved[0].title == "Invented paper" and out.retrieved[0].is_reference
    assert store.chunks([]) == {}

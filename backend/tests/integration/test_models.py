"""Constraint behaviour on a migrated scratch database."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Chunk, EvalItem, EvalResult, EvalRun, HFArtifact, Paper, Repo

pytestmark = pytest.mark.integration


def _paper(arxiv_id: str = "0000.00001") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title="Invented paper",
        abstract="Invented abstract.",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_paper_defaults_and_timezone_aware_timestamps(migrated_session: Session) -> None:
    migrated_session.add(_paper())
    migrated_session.commit()
    p = migrated_session.scalars(select(Paper)).one()
    assert p.fulltext_status == "none"
    assert p.published_at.tzinfo is not None and p.created_at.tzinfo is not None
    assert p.authors == [] and p.categories == []


def test_child_rows_cascade_when_paper_is_deleted(migrated_session: Session) -> None:
    s = migrated_session
    s.add_all([_paper(), Repo(arxiv_id="0000.00001", full_name="a/b")])
    s.commit()
    s.delete(s.get(Paper, "0000.00001"))
    s.commit()
    assert s.scalar(select(func.count()).select_from(Repo)) == 0


def test_foreign_key_and_check_constraints(migrated_session: Session) -> None:
    s = migrated_session
    s.add(Repo(arxiv_id="missing", full_name="a/b"))  # no such paper
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()

    s.add(_paper())
    s.add(HFArtifact(arxiv_id="0000.00001", kind="weights", repo_id="x/y"))  # bad kind
    with pytest.raises(IntegrityError):
        s.commit()
    s.rollback()

    s.add(Chunk(id=__import__("uuid").uuid4(), arxiv_id="nope", chunk_type="abstract",
                ord=0, text="t", token_count=1))  # fmt: skip
    with pytest.raises(IntegrityError):
        s.commit()


def test_one_eval_result_per_run_and_item(migrated_session: Session) -> None:
    s = migrated_session
    run, item = EvalRun(name="r"), EvalItem(question="q?")
    s.add_all([run, item])
    s.commit()
    s.add(EvalResult(run_id=run.id, item_id=item.id))
    s.commit()
    s.add(EvalResult(run_id=run.id, item_id=item.id))
    with pytest.raises(IntegrityError):
        s.commit()

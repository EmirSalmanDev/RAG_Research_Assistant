"""The analysis Celery tasks on a throwaway database (run with `.run()`, no broker).
No OpenAI client is ever created here: the paid paths are switched off or have no work."""

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import IngestRun
from app.llm import tasks

pytestmark = pytest.mark.integration


@pytest.fixture
def no_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> Any:
        raise AssertionError("no OpenAI client may be created in this test")

    monkeypatch.setattr(tasks, "_openai", refuse)


@pytest.mark.usefixtures("no_openai")
def test_scheduled_submit_is_off_by_default(migrated_session: Session) -> None:
    assert get_settings().analysis_auto_submit is False
    counts = tasks.submit_analyses.run()
    assert counts == {"skipped": "analysis_auto_submit is off"}
    run = migrated_session.scalars(select(IngestRun)).one()
    assert run.source == "analysis-submit" and run.counts == counts


@pytest.mark.usefixtures("no_openai")
def test_collect_without_open_batches_does_nothing(migrated_session: Session) -> None:
    assert tasks.collect_analyses.run() == {"open": 0}
    assert migrated_session.scalars(select(IngestRun)).all() == []  # nothing recorded

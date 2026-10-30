"""The monthly budget guard (app/features/budget.py) and the background tasks it pauses.

Every paid path is replaced by a function that fails the test when it is called, so "paused"
means "no embedder, no OpenAI client and no Batch call was even reached"."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import IngestRun, LLMUsage, Paper
from app.features import app_settings, budget
from app.features import tasks as digest_tasks
from app.ingest import tasks as ingest_tasks
from app.llm import tasks as llm_tasks
from tests.digest_fakes import FakeDense, FakeVectors, make_interest, make_paper

pytestmark = pytest.mark.integration


def _spend(s: Session, usd: str, when: datetime | None = None) -> None:
    s.add(
        LLMUsage(
            feature="t",
            model="m",
            tokens_in=1,
            tokens_out=0,
            cost_usd=Decimal(usd),
            created_at=when or datetime.now(UTC),
        )  # fmt: skip
    )
    s.commit()


def _cap(s: Session, usd: float | None) -> None:
    app_settings.save(s, {app_settings.KEY_BUDGET_CAP: usd})
    s.commit()


# --- the state ---------------------------------------------------------------------------


def test_no_cap_means_no_guard_however_much_was_spent(migrated_session: Session) -> None:
    _spend(migrated_session, "999")
    state = budget.budget_state(migrated_session)
    assert state.cap_usd is None and state.spent_usd == Decimal("999") and not state.paused


@pytest.mark.parametrize(
    ("cap", "spent", "paused"),
    [
        (1.0, "0.99", False),
        (1.0, "1.00", True),  # exactly at the cap is paused: the cap is a ceiling
        (1.0, "1.50", True),
        (0.0, "0", True),  # a cap of 0 pauses everything that spends
        (0.1, "0.1", True),  # 0.1 as a JSON float must not become 0.1000000000000000055
    ],
)
def test_paused_from_the_cap_and_the_month_to_date_spend(
    migrated_session: Session, cap: float, spent: str, paused: bool
) -> None:
    _cap(migrated_session, cap)
    _spend(migrated_session, spent)
    assert budget.is_paused(migrated_session) is paused


def test_a_new_month_starts_again_from_zero(migrated_session: Session) -> None:
    s = migrated_session
    _cap(s, 1.0)
    _spend(s, "5", datetime(2026, 9, 15, 12, 0, tzinfo=UTC))
    assert budget.is_paused(s, now=datetime(2026, 9, 20, tzinfo=UTC))
    assert not budget.is_paused(s, now=datetime(2026, 10, 2, tzinfo=UTC))


def test_raising_the_cap_resumes_without_a_restart(migrated_session: Session) -> None:
    s = migrated_session
    _cap(s, 1.0)
    _spend(s, "1.2")
    assert budget.is_paused(s)
    _cap(s, 2.0)
    assert not budget.is_paused(s)


# --- the tasks ---------------------------------------------------------------------------


def _forbidden(name: str) -> Any:
    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} must not be reached while the budget is paused")

    return refuse


@pytest.fixture
def paused(migrated_session: Session) -> Session:
    _cap(migrated_session, 0.5)
    _spend(migrated_session, "0.5")
    return migrated_session


def _last_run(s: Session, source: str) -> IngestRun:
    s.expire_all()
    return s.scalars(select(IngestRun).where(IngestRun.source == source)).one()


def test_embed_paper_is_skipped_and_the_paper_keeps_its_status(
    paused: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.ingest.index as index

    make_paper(paused, "2610.00001").fulltext_status = "fetched"
    paused.commit()
    monkeypatch.setattr(index, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(index, "upsert_paper_chunks", _forbidden("upsert_paper_chunks"))
    monkeypatch.setattr(ingest_tasks, "_embedders", _forbidden("_embedders"))

    counts = ingest_tasks.embed_paper.run("2610.00001")

    assert counts == {"arxiv_id": "2610.00001", "skipped": budget.PAUSED_REASON}
    assert _last_run(paused, "embed-paper").counts == counts
    paused.expire_all()
    assert paused.get(Paper, "2610.00001").fulltext_status == "fetched"  # type: ignore[union-attr]


def test_index_abstracts_is_skipped(paused: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    import app.ingest.index as index

    monkeypatch.setattr(index, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(index, "index_abstracts", _forbidden("index_abstracts"))
    monkeypatch.setattr(ingest_tasks, "_embedders", _forbidden("_embedders"))

    assert ingest_tasks.index_abstracts.run() == {"skipped": budget.PAUSED_REASON}


def test_select_fulltext_starts_no_chain_while_paused_and_resumes_stuck_papers_after(
    migrated_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = migrated_session
    started: list[tuple[str, str]] = []

    class FakeChain:
        def __init__(self, kind: str, arxiv_id: str) -> None:
            self.kind, self.arxiv_id = kind, arxiv_id

        def apply_async(self) -> None:
            started.append((self.kind, self.arxiv_id))

    monkeypatch.setattr(ingest_tasks, "paper_chain", lambda i: FakeChain("full", i))
    monkeypatch.setattr(ingest_tasks, "resume_chain", lambda i: FakeChain("resume", i))
    make_paper(s, "2610.00001").fulltext_status = "requested"  # needs a download
    make_paper(s, "2610.00002").fulltext_status = "fetched"  # text stored, never indexed
    make_paper(s, "2610.00003")  # nothing to do
    s.commit()
    _cap(s, 0.5)
    _spend(s, "0.5")

    counts = ingest_tasks.select_fulltext.run()
    assert counts["skipped"] == budget.PAUSED_REASON and started == []

    _cap(s, 5.0)  # the owner raises the cap: the next run does the work
    counts = ingest_tasks.select_fulltext.run()
    assert started == [("full", "2610.00001"), ("resume", "2610.00002")]
    assert counts["selected"] == 1 and counts["resumed"] == 1


def test_score_relevance_scores_with_stored_vectors_but_embeds_nothing(
    paused: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    dense, vectors = FakeDense(), FakeVectors()
    now = datetime.now(UTC)
    interest = make_interest(paused, "Alpha topic", embedding=[1.0, 0.0])
    make_paper(paused, "2610.00001", now=now)
    vectors.vectors["2610.00001"] = [1.0, 0.0]
    monkeypatch.setattr(digest_tasks, "_dense", lambda: dense)
    monkeypatch.setattr(digest_tasks, "_vectors", lambda: vectors)

    counts = digest_tasks.score_relevance.run()

    assert dense.calls == [] and counts["embed_skipped"] == budget.PAUSED_REASON
    assert "embed_error" not in counts and interest.id is not None
    paused.expire_all()
    assert paused.get(Paper, "2610.00001").relevance_score is not None  # type: ignore[union-attr]


def test_analysis_submit_is_skipped_even_when_auto_submit_is_on(
    paused: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.llm.batch as batch

    monkeypatch.setenv("ANALYSIS_AUTO_SUBMIT", "true")
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(llm_tasks, "_openai", _forbidden("_openai"))
    monkeypatch.setattr(batch, "submit", _forbidden("submit"))

    assert llm_tasks.submit_analyses.run() == {"skipped": budget.PAUSED_REASON}


def test_analysis_collect_leaves_open_batches_alone_and_writes_no_row(
    paused: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.llm.batch as batch

    monkeypatch.setattr(batch, "open_batch_count", lambda session: 2)
    monkeypatch.setattr(batch, "collect", _forbidden("collect"))
    monkeypatch.setattr(llm_tasks, "_openai", _forbidden("_openai"))

    assert llm_tasks.collect_analyses.run() == {"open": 2, "skipped": budget.PAUSED_REASON}
    assert paused.scalars(select(IngestRun)).all() == []  # every 30 minutes: no row spam

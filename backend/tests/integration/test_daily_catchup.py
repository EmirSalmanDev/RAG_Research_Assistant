"""The same-day catch-up (app/ingest/catchup.py) on a scratch Postgres: when it starts the
daily chain again, when it skips, the `daily-catchup` row it writes, and how /status shows it.

Faked: the daily chain itself (a recorder: nothing is fetched) and the arXiv client (it fails
the test if anything builds it). Real: the decision's queries, the task, `ingest_runs`, the
/status reading."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import status
from app.config import Settings, get_settings
from app.db.models import IngestRun
from app.ingest import catchup
from app.ingest import tasks as ingest_tasks

pytestmark = pytest.mark.integration

NOW = datetime(2026, 10, 29, 10, 0, tzinfo=UTC)  # 13:00 Istanbul, the first catch-up
SINCE = datetime(2026, 10, 29, 4, 0, tzinfo=UTC)  # today's 07:00 Istanbul
WILL_RETRY = "will retry in 16 min (attempt 3 of 4): RetryableError: HTTP 429"
LAST_ATTEMPT = "last attempt (4 of 4), no retry: RetryableError: network error: Read timed out."
DEFAULTS = Settings(_env_file=None)


def _run(
    s: Session, started: datetime, finished: datetime | None, error: str | None = None
) -> IngestRun:
    run = IngestRun(
        source="arxiv-daily", started_at=started, finished_at=finished,
        counts={} if error else {"cursor": (finished or started).isoformat()},
        errors={"error": error} if error else {},
    )  # fmt: skip
    s.add(run)
    s.commit()
    return run


def _morning_failed(s: Session, day_start: datetime) -> None:
    """Yesterday's success, then this morning's four failed attempts (as on 2026-10-29)."""
    _run(s, day_start - timedelta(hours=16), day_start - timedelta(hours=15, minutes=58))
    for minute, error in ((0, "will retry in 2 min (attempt 1 of 4): x"),
                          (2, "will retry in 7 min (attempt 2 of 4): x"),
                          (10, WILL_RETRY), (27, LAST_ATTEMPT)):  # fmt: skip
        start = day_start + timedelta(minutes=minute)
        _run(s, start, start + timedelta(minutes=1), error)


def test_a_failed_morning_starts_the_chain(migrated_session: Session) -> None:
    _morning_failed(migrated_session, SINCE)
    decision = catchup.decide(migrated_session, NOW, DEFAULTS)
    assert decision.start and decision.action == "started"
    assert decision.reason == (
        "no arxiv-daily run succeeded since 2026-10-29T04:00:00+00:00 (07:00 Istanbul)"
    )


def test_a_success_after_07_00_skips(migrated_session: Session) -> None:
    _morning_failed(migrated_session, SINCE)
    ok = _run(migrated_session, NOW - timedelta(hours=1), NOW - timedelta(minutes=58))
    decision = catchup.decide(migrated_session, NOW, DEFAULTS)
    assert not decision.start and decision.action == "skipped"
    assert decision.reason.startswith(f"arxiv-daily run #{ok.id} succeeded at 2026-10-29T09:02")


def test_a_success_exactly_at_07_00_counts_and_one_just_before_does_not(
    migrated_session: Session,
) -> None:
    s = migrated_session
    _run(s, SINCE - timedelta(minutes=2), SINCE - timedelta(seconds=1))  # 06:59:59 Istanbul
    assert catchup.decide(s, NOW, DEFAULTS).start
    _run(s, SINCE - timedelta(minutes=1), SINCE)  # finished at 07:00:00 exactly
    assert not catchup.decide(s, NOW, DEFAULTS).start


def test_a_run_still_going_skips_but_an_old_unfinished_one_does_not(
    migrated_session: Session,
) -> None:
    s = migrated_session
    _morning_failed(s, SINCE)
    old = _run(s, NOW - timedelta(hours=2, minutes=30), None)  # left by a stopped worker
    assert catchup.decide(s, NOW, DEFAULTS).start
    running = _run(s, NOW - timedelta(minutes=30), None)
    decision = catchup.decide(s, NOW, DEFAULTS)
    assert not decision.start
    assert decision.reason.startswith(
        f"a daily chain is in progress: arxiv-daily run #{running.id} started at"
    )
    assert old.id < running.id


def test_a_waiting_retry_skips_and_a_stale_one_does_not(migrated_session: Session) -> None:
    s = migrated_session
    _run(s, NOW - timedelta(hours=3, minutes=1), NOW - timedelta(hours=3), WILL_RETRY)
    assert catchup.decide(s, NOW, DEFAULTS).start  # its retry would have run long ago
    waiting = _run(s, NOW - timedelta(minutes=11), NOW - timedelta(minutes=10), WILL_RETRY)
    decision = catchup.decide(s, NOW, DEFAULTS)
    assert not decision.start
    assert decision.reason.startswith(
        f"a fetch_arxiv retry is waiting: arxiv-daily run #{waiting.id}"
    )


def test_the_last_attempt_is_not_a_waiting_retry(migrated_session: Session) -> None:
    s = migrated_session
    _run(s, NOW - timedelta(minutes=20), NOW - timedelta(minutes=19), WILL_RETRY)
    _run(s, NOW - timedelta(minutes=5), NOW - timedelta(minutes=4), LAST_ATTEMPT)
    assert catchup.decide(s, NOW, DEFAULTS).start


def test_off_when_the_setting_is_empty(migrated_session: Session) -> None:
    _morning_failed(migrated_session, SINCE)
    decision = catchup.decide(
        migrated_session, NOW, Settings(_env_file=None, daily_catchup_times="")
    )
    assert not decision.start
    assert decision.reason == "the catch-up is off (DAILY_CATCHUP_TIMES is empty)"


# --- the task ---------------------------------------------------------------------------------


class ChainRecorder:
    """Stands in for daily_chain(): records each send, runs nothing."""

    def __init__(self) -> None:
        self.sent = 0

    def __call__(self) -> "ChainRecorder":
        return self

    def apply_async(self) -> Any:
        self.sent += 1
        return type("Result", (), {"id": "chain-0001"})()


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> ChainRecorder:
    recorder = ChainRecorder()
    monkeypatch.setattr(ingest_tasks, "daily_chain", recorder)

    def no_arxiv() -> Any:
        raise AssertionError("the catch-up decision must not build an arXiv client")

    monkeypatch.setattr(ingest_tasks, "_arxiv", no_arxiv)
    get_settings.cache_clear()
    return recorder


def _catchup_rows(s: Session) -> list[IngestRun]:
    s.expire_all()
    query = select(IngestRun).where(IngestRun.source == "daily-catchup").order_by(IngestRun.id)
    return list(s.scalars(query))


def test_the_task_starts_the_chain_after_a_failed_morning(
    migrated_session: Session, chain: ChainRecorder
) -> None:
    s = migrated_session
    _morning_failed(s, catchup.latest_daily_start(datetime.now(UTC)))

    result = ingest_tasks.daily_catchup.apply().get()

    assert chain.sent == 1
    [row] = _catchup_rows(s)
    assert row.errors == {} and row.finished_at is not None
    assert row.counts == result
    assert row.counts["action"] == "started" and row.counts["chain_id"] == "chain-0001"
    assert row.counts["reason"].startswith("no arxiv-daily run succeeded since")


def test_the_task_skips_after_a_success_and_status_shows_both_lines(
    migrated_session: Session, chain: ChainRecorder
) -> None:
    s = migrated_session
    day_start = catchup.latest_daily_start(datetime.now(UTC))
    _morning_failed(s, day_start)
    now = datetime.now(UTC)  # a success after today's 07:00 whatever the clock says
    ok = _run(s, max(day_start, now - timedelta(minutes=2)), now - timedelta(milliseconds=1))

    ingest_tasks.daily_catchup.apply().get()

    assert chain.sent == 0  # no chain, so no external request
    [row] = _catchup_rows(s)
    assert set(row.counts) == {"action", "reason"} and row.counts["action"] == "skipped"
    assert row.counts["reason"].startswith(f"arxiv-daily run #{ok.id} succeeded at")
    # /status: the catch-up is its own step; the arxiv-daily line is still the real latest run
    by_source = {r.source: r for r in status.read_database().last_runs}
    assert by_source["daily-catchup"].ok and by_source["daily-catchup"].note == (
        f"skipped: {row.counts['reason']}"
    )
    arxiv = by_source["arxiv-daily"]
    assert arxiv.ok and arxiv.started_at == ok.started_at and not arxiv.retrying

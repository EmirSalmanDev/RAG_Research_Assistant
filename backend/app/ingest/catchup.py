"""The same-day catch-up of the daily chain (docs/adr/0003).

On two mornings in a row (2026-10-28 and 2026-10-29) arXiv refused the 07:00 run for its
whole retry window, while runs started later in the day were answered. Beat therefore sends
`ingest.daily_catchup` at the `daily_catchup_times` (13:00 and 19:00 Istanbul by default).
`decide()` says whether that catch-up starts the daily chain again:

1. not if an `arxiv-daily` run finished without an error at or after the latest 07:00
   Istanbul (the morning run, or an earlier catch-up, already got the day's papers);
2. not if a daily chain is still in progress: an `arxiv-daily` run that has not finished
   and started less than IN_PROGRESS_WINDOW ago, or a fetch_arxiv retry that waits for its
   countdown (the latest `arxiv-daily` row is a "will retry" row that finished less than
   IN_PROGRESS_WINDOW ago).

Otherwise it starts the chain. Deciding reads Postgres only: a skipped catch-up makes no
external request. The task in tasks.py records the decision as a `daily-catchup` run.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DAILY_INGEST_AT, SCHEDULE_TIMEZONE, Settings
from app.db.models import IngestRun
from app.ingest import retry

SOURCE = "daily-catchup"  # the catch-up's own ingest_runs rows
ARXIV_DAILY = "arxiv-daily"  # the chain's first step (app/ingest/daily.py)
# Longer than one fetch attempt and longer than any retry delay: both stay below the
# broker's 1-hour visibility timeout (app/config.py). An unfinished row older than this
# was left by a worker that stopped during the run (the health page calls it abandoned
# after 6 h); it does not hold the catch-up back.
IN_PROGRESS_WINDOW = timedelta(hours=2)


@dataclass(frozen=True)
class Decision:
    start: bool
    reason: str

    @property
    def action(self) -> str:
        return "started" if self.start else "skipped"


def latest_daily_start(now: datetime) -> datetime:
    """The latest 07:00 Istanbul at or before `now`, in UTC. For a catch-up time (always
    later than 07:00) that is today's 07:00; a catch-up that Beat sends late, after
    midnight, looks back at yesterday's."""
    zone = ZoneInfo(SCHEDULE_TIMEZONE)
    local = now.astimezone(zone)
    start = datetime.combine(local.date(), DAILY_INGEST_AT, tzinfo=zone)
    if start > local:
        start = datetime.combine(local.date() - timedelta(days=1), DAILY_INGEST_AT, tzinfo=zone)
    return start.astimezone(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def decide(session: Session, now: datetime, settings: Settings) -> Decision:
    if not settings.daily_catchup_at:
        return Decision(False, "the catch-up is off (DAILY_CATCHUP_TIMES is empty)")
    since = latest_daily_start(now)
    finished_since = session.scalars(
        select(IngestRun)
        .where(IngestRun.source == ARXIV_DAILY, IngestRun.finished_at >= since)
        .order_by(IngestRun.finished_at.desc())
    )
    for run in finished_since:
        if not run.errors:
            assert run.finished_at is not None  # the query asked for it
            return Decision(
                False,
                f"arxiv-daily run #{run.id} succeeded at {_iso(run.finished_at)}, "
                f"after {_iso(since)} (07:00 Istanbul)",
            )
    recent = now - IN_PROGRESS_WINDOW
    running = session.scalars(
        select(IngestRun)
        .where(
            IngestRun.source == ARXIV_DAILY,
            IngestRun.finished_at.is_(None),
            IngestRun.started_at > recent,
        )
        .order_by(IngestRun.id.desc())
    ).first()
    if running is not None:
        return Decision(
            False,
            f"a daily chain is in progress: arxiv-daily run #{running.id} started at "
            f"{_iso(running.started_at)} and has not finished",
        )
    latest = session.scalars(
        select(IngestRun).where(IngestRun.source == ARXIV_DAILY).order_by(IngestRun.id.desc())
    ).first()
    if (
        latest is not None
        and latest.finished_at is not None
        and latest.finished_at > recent
        and retry.is_retry_note((latest.errors or {}).get("error"))
    ):
        return Decision(
            False,
            f"a fetch_arxiv retry is waiting: arxiv-daily run #{latest.id} finished at "
            f"{_iso(latest.finished_at)} with a retry note",
        )
    return Decision(True, f"no arxiv-daily run succeeded since {_iso(since)} (07:00 Istanbul)")

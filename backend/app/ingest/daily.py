"""The daily ingest steps that work on many papers at once, plus run bookkeeping.

- `recorded_run()` writes one `ingest_runs` row per step run (source, started_at,
  finished_at, counts, errors); a run that raises is recorded with its error.
- The arXiv window: new submissions and new versions in the configured categories,
  from the last successful run's cursor minus a 4-day overlap until now. arXiv shows
  papers a day or more after submission (and not at weekends), so the overlap catches
  late arrivals; upserts make the repeats free, and `index_abstracts` skips papers that
  are already indexed, so they cost no embeddings either.
- `fetch_hf_daily()` merges Hugging Face's daily papers (upvotes, GitHub repo, project
  page) and adds trending papers we do not hold, even outside our categories.
- `backfill()` fetches a past window and indexes abstracts only (no full text).

The Celery tasks (`tasks.py`) and the CLI call these functions; they hold no logic.
"""

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import IngestRun, Paper
from app.ingest.paper import upsert_paper
from app.sources.arxiv import ArxivPaper
from app.sources.hf import HFPaper

logger = logging.getLogger(__name__)

ARXIV_SOURCES = ("arxiv-daily", "arxiv-backfill")  # runs whose cursor the next run uses
OVERLAP = timedelta(days=4)
FIRST_RUN_LOOKBACK = timedelta(days=1)  # no cursor yet: the last day (plus the overlap)
DEFAULT_MAX_PAPERS = 20_000


# --- ingest_runs bookkeeping ---------------------------------------------------------------


@contextmanager
def recorded_run(session: Session, source: str) -> Iterator[dict[str, Any]]:
    """Record one run in `ingest_runs`. The body fills the yielded `counts` dict.

    On success the row is committed with its counts and an empty `errors`. If the body
    raises, the work is rolled back, the row is written with the error, and the
    exception is re-raised (so Celery can retry).
    """
    run = IngestRun(source=source, counts={}, errors={})
    session.add(run)
    session.commit()
    counts: dict[str, Any] = {}
    try:
        yield counts
    except Exception as err:
        session.rollback()
        run.counts, run.errors = counts, {"error": f"{type(err).__name__}: {err}"[:500]}
        run.finished_at = datetime.now(UTC)
        session.commit()
        raise
    run.counts, run.finished_at = counts, datetime.now(UTC)
    session.commit()


# --- the arXiv window ----------------------------------------------------------------------


def last_cursor(session: Session) -> datetime | None:
    """`counts["cursor"]` of the latest successful arXiv run (daily or backfill)."""
    runs = session.scalars(
        select(IngestRun)
        .where(IngestRun.source.in_(ARXIV_SOURCES), IngestRun.finished_at.is_not(None))
        .order_by(IngestRun.finished_at.desc())
    )
    for run in runs:
        if not run.errors and run.counts.get("cursor"):
            return datetime.fromisoformat(run.counts["cursor"])
    return None


def daily_window(cursor: datetime | None, now: datetime) -> tuple[datetime, datetime]:
    """[cursor - 4 days, now]; without a cursor, [now - 1 day - 4 days, now]."""
    start = cursor if cursor is not None else now - FIRST_RUN_LOOKBACK
    return start - OVERLAP, now


class ArxivWindowSource(Protocol):
    def updated_papers(
        self, categories: Sequence[str], since: datetime, until: datetime | None = ...,
        page_size: int = ..., max_papers: int = ...,
    ) -> list[ArxivPaper]: ...  # fmt: skip
    def get_papers(self, arxiv_ids: Sequence[str]) -> list[ArxivPaper]: ...


def store_papers(session: Session, papers: Sequence[ArxivPaper]) -> dict[str, int]:
    """Upsert arXiv metadata. Counts: new papers, new versions of known papers, unchanged."""
    rows = session.execute(
        select(Paper.arxiv_id, Paper.version).where(
            Paper.arxiv_id.in_([p.arxiv_id for p in papers])
        )
    ).all()
    known = {arxiv_id: version for arxiv_id, version in rows}
    counts = {"new": 0, "new_version": 0, "unchanged": 0}
    for paper in papers:
        old = known.get(paper.arxiv_id)
        counts["new" if old is None else "new_version" if paper.version > old else "unchanged"] += 1
        upsert_paper(session, paper)
    return counts


def fetch_arxiv_window(
    session: Session,
    arxiv: ArxivWindowSource,
    since: datetime,
    until: datetime,
    *,
    max_papers: int = DEFAULT_MAX_PAPERS,
) -> dict[str, Any]:
    """Fetch and store one window. Returns counts, with the window and the new cursor."""
    papers = arxiv.updated_papers(
        get_settings().arxiv_categories, since, until, max_papers=max_papers
    )
    counts: dict[str, Any] = {"fetched": len(papers), **store_papers(session, papers)}
    counts.update(since=since.isoformat(), until=until.isoformat(), cursor=until.isoformat())
    counts["capped"] = len(papers) >= max_papers
    return counts


def fetch_arxiv_daily(
    session: Session, arxiv: ArxivWindowSource, now: datetime | None = None
) -> dict[str, Any]:
    """The daily step: fetch [cursor - 4 days, now] and record the run as 'arxiv-daily'."""
    now = now or datetime.now(UTC)
    since, until = daily_window(last_cursor(session), now)
    with recorded_run(session, "arxiv-daily") as counts:
        counts.update(fetch_arxiv_window(session, arxiv, since, until))
    return counts


# --- Hugging Face daily papers -------------------------------------------------------------


class HFDailySource(Protocol):
    def daily_papers(self, day: date) -> list[HFPaper]: ...


def fetch_hf_daily(
    session: Session, hf: HFDailySource, arxiv: ArxivWindowSource, today: date | None = None
) -> dict[str, Any]:
    """Yesterday's and today's Hugging Face daily papers: fetch the ones we do not hold
    from arXiv (50 ids per request), then merge upvotes, GitHub repo and project page.
    Recorded as an 'hf-daily' run."""
    today = today or datetime.now(UTC).date()
    with recorded_run(session, "hf-daily") as counts:
        listed: dict[str, HFPaper] = {}
        for day in (today - timedelta(days=1), today):
            for paper in hf.daily_papers(day):
                listed[paper.id] = paper
        held = set(session.scalars(select(Paper.arxiv_id).where(Paper.arxiv_id.in_(listed))))
        missing = [i for i in listed if i not in held]
        fetched = arxiv.get_papers(missing) if missing else []
        counts.update(listed=len(listed), missing=len(missing), **store_papers(session, fetched))
        merged = 0
        for arxiv_id, hf_paper in listed.items():
            result = session.execute(
                update(Paper)
                .where(Paper.arxiv_id == arxiv_id)
                .values(
                    hf_upvotes=hf_paper.upvotes,
                    # untrusted third-party URLs (rule 8): stored and shown as data only
                    github_url=hf_paper.github_repo or Paper.github_url,
                    project_url=hf_paper.project_page or Paper.project_url,
                )
            )
            merged += result.rowcount  # type: ignore[attr-defined]
        counts["merged"] = merged
    return counts


# --- backfill ---------------------------------------------------------------------------------


@dataclass
class BackfillPlan:
    days: int
    since: datetime
    until: datetime


def plan_backfill(days: int, now: datetime | None = None) -> BackfillPlan:
    now = now or datetime.now(UTC)
    return BackfillPlan(days=days, since=now - timedelta(days=days), until=now)


def backfill(
    session: Session,
    arxiv: ArxivWindowSource,
    plan: BackfillPlan,
    index: Callable[[Session], int],
    *,
    max_papers: int = DEFAULT_MAX_PAPERS,
    before_index: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Fetch a past window ('arxiv-backfill' run, which also sets the cursor), then index
    the abstracts of every paper not yet indexed ('index-abstracts' run). No full text.
    `before_index` sees the fetch counts first (the CLI prints its estimate there)."""
    with recorded_run(session, "arxiv-backfill") as fetched:
        fetched.update(
            fetch_arxiv_window(session, arxiv, plan.since, plan.until, max_papers=max_papers)
        )
    if before_index is not None:
        before_index(fetched)
    with recorded_run(session, "index-abstracts") as indexed:
        indexed["indexed"] = index(session)
    return {"fetch": fetched, "index": indexed}

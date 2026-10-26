"""GET /status: what the /health page of the app shows.

`/health` (app/api/health.py) is the cheap probe Docker uses: three stores, up or down.
This is the owner's view of the running system: the stores again, the last run of every
pipeline step, how many papers are in each full-text state, how many tasks wait in the
queue, how many points the Qdrant collection holds, and this month's spend against the cap.

Every part is read on its own and fails soft: a part that cannot be read is `null` and its
reason is in `errors`, so a dead Redis does not hide the database numbers, and the page can
say what is wrong instead of showing nothing. The route always answers 200; the status of
the stores is in the body. Nothing here writes, and nothing calls a paid or external API.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime

import redis.asyncio as aioredis
from fastapi import APIRouter
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import distinct_on

from app.api import health
from app.config import get_settings
from app.db.models import IngestRun, Paper
from app.db.session import session_scope
from app.features import budget

router = APIRouter()

TIMEOUT_S = 3.0
CELERY_QUEUE = "celery"  # Celery's default queue; its key in Redis is the queue's name


class RunOut(BaseModel):
    source: str
    started_at: datetime
    finished_at: datetime | None
    ok: bool  # finished without an error
    error: str | None
    skipped: str | None  # why the run did nothing (e.g. the budget cap), if it did not work


class BudgetOut(BaseModel):
    cap_usd: float | None
    spent_usd: float
    paused: bool
    month_start: datetime


class DatabaseStatus(BaseModel):
    papers_total: int
    papers_by_fulltext_status: dict[str, int]
    last_runs: list[RunOut]  # the latest run of each source, newest first
    budget: BudgetOut


class StatusResponse(BaseModel):
    stores: health.HealthResponse
    database: DatabaseStatus | None
    queue_length: int | None
    qdrant_points: int | None
    errors: dict[str, str]  # part name -> why it could not be read


def read_database() -> DatabaseStatus:
    with session_scope() as session:
        by_status = dict(
            session.execute(
                select(Paper.fulltext_status, func.count()).group_by(Paper.fulltext_status)
            ).all()
        )
        latest = session.scalars(
            select(IngestRun)
            .ext(distinct_on(IngestRun.source))  # Postgres: the first row of each source
            .order_by(IngestRun.source, IngestRun.id.desc())
        ).all()
        state = budget.budget_state(session)
        runs = [_run_out(run) for run in latest]
        return DatabaseStatus(
            papers_total=sum(by_status.values()),
            papers_by_fulltext_status=by_status,
            last_runs=sorted(runs, key=lambda r: r.started_at, reverse=True),
            budget=BudgetOut(
                cap_usd=None if state.cap_usd is None else float(state.cap_usd),
                spent_usd=float(state.spent_usd),
                paused=state.paused,
                month_start=state.month_start,
            ),
        )


def _run_out(run: IngestRun) -> RunOut:
    error = run.errors.get("error") if run.errors else None
    skipped = run.counts.get("skipped") if run.counts else None
    return RunOut(
        source=run.source,
        started_at=run.started_at,
        finished_at=run.finished_at,
        ok=run.finished_at is not None and not run.errors,
        error=None if error is None else str(error)[:300],
        skipped=None if skipped is None else str(skipped)[:200],
    )


async def read_queue_length() -> int:
    client = aioredis.from_url(get_settings().redis_url)
    try:
        return int(await client.llen(CELERY_QUEUE))
    finally:
        await client.aclose()


async def read_qdrant_points() -> int:
    client = AsyncQdrantClient(url=get_settings().qdrant_url, timeout=int(TIMEOUT_S))
    try:
        name = get_settings().qdrant_collection
        # The collection is created by the first ingest. Before that it is not a fault: a
        # fresh install has 0 points, and the page should not open with a warning.
        if not await client.collection_exists(name):
            return 0
        info = await client.get_collection(name)
        return int(info.points_count or 0)
    finally:
        await client.close()


async def _soft[T](name: str, read: Callable[[], Awaitable[T]], errors: dict[str, str]) -> T | None:
    """Run one read; on any failure or a timeout return None and note why."""
    try:
        return await asyncio.wait_for(read(), timeout=TIMEOUT_S)
    except TimeoutError:
        errors[name] = f"no answer within {TIMEOUT_S:g} s"
    except Exception as err:
        errors[name] = f"{type(err).__name__}: {err}"[:200]
    return None


@router.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    # Looked up at call time (module globals) so tests can monkeypatch the readers.
    errors: dict[str, str] = {}

    async def database() -> DatabaseStatus:
        return await asyncio.to_thread(read_database)

    pg, qd, rd, db, queue, points = await asyncio.gather(
        health._probe(health.check_postgres),
        health._probe(health.check_qdrant),
        health._probe(health.check_redis),
        _soft("database", database, errors),
        _soft("queue_length", read_queue_length, errors),
        _soft("qdrant_points", read_qdrant_points, errors),
    )
    return StatusResponse(
        stores=health.HealthResponse(postgres=pg, qdrant=qd, redis=rd),
        database=db,
        queue_length=queue,
        qdrant_points=points,
        errors=errors,
    )

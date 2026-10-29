"""Celery tasks: one per pipeline step, each a thin wrapper around a function from
`daily.py`, `enrich.py`, `fulltext.py`, `pipeline.py` or `index.py` (no logic is
duplicated here).

The daily chain (Beat, 07:00 Europe/Istanbul):
    fetch_arxiv -> fetch_hf_daily -> index_abstracts -> enrich_new -> score_relevance
    -> select_fulltext
(`score_relevance` is in app/features/tasks.py; the digest is built by its own Beat entry
at 08:00; `daily_catchup` sends the same chain again later the same day when the 07:00 run
did not get past arXiv) and `select_fulltext` starts, for every selected paper, its own chain:
    fetch_fulltext -> clean_paper -> chunk_paper -> embed_paper

Rules every task follows:
- Immutable signatures (`.si()`): a task never receives the previous task's result,
  only ids, and reads everything else from Postgres. So each task can be re-run alone.
- Idempotent: re-running a task changes nothing that is already done.
- Retries: any error except a ValueError (bad input, retrying cannot help) is retried
  up to 3 times with exponential backoff and jitter. After the last retry a per-paper
  task marks its paper 'failed' (shown on the health page; a manual request resets it).
  The chain's two fetch steps (fetch_arxiv, fetch_hf_daily) wait minutes between attempts
  instead of seconds, so a source that is down for a while does not cost the day; their
  failed runs say whether and when they will be retried (app/ingest/retry.py).
- Each run writes one `ingest_runs` row (source = the task name).
- The steps that spend money (embedding) do nothing while the monthly budget cap is reached
  (app/features/budget.py): the run records `skipped`, the paper keeps its status and is picked
  up again by `select_fulltext` once the cap allows (docs/adr/0013).
"""

import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from celery import Task, chain
from celery.canvas import Signature
from sqlalchemy import select

from app.config import get_settings
from app.db.models import Paper
from app.db.session import session_scope
from app.features import budget
from app.ingest import catchup, daily, pipeline, retry
from app.ingest.fulltext import default_sources, select_for_fulltext
from app.ingest.fulltext import fetch_fulltext as fetch_fulltext_step
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

RETRY_OPTIONS: dict[str, Any] = {
    "autoretry_for": (Exception,),
    "dont_autoretry_for": (ValueError,),
    "max_retries": 3,
    "retry_backoff": True,  # 1 s, 2 s, 4 s, ... between attempts
    "retry_backoff_max": 600,
    "retry_jitter": True,
}


class PaperTask(Task):  # type: ignore[misc]
    """A task whose first argument is an arXiv id. When it fails for good (no retries
    left, or an error that is not retried), the paper is marked 'failed'."""

    def on_failure(
        self, exc: BaseException, task_id: str, args: Any, kwargs: Any, einfo: Any
    ) -> None:
        arxiv_id = args[0] if args else kwargs.get("arxiv_id")
        logger.error("task %s failed for %s after retries: %s", self.name, arxiv_id, exc)
        if arxiv_id:
            mark_failed(str(arxiv_id))


def mark_failed(arxiv_id: str) -> None:
    with session_scope() as session:
        paper = session.get(Paper, arxiv_id)
        if paper is not None:
            paper.fulltext_status = "failed"


def _arxiv() -> Any:
    from app.sources.arxiv import ArxivClient

    return ArxivClient()


def _embedders() -> dict[str, Any]:
    from app.ingest.embed import get_dense_embedder, get_sparse_embedder

    return {"dense": get_dense_embedder(), "sparse": get_sparse_embedder()}


# --- the daily chain --------------------------------------------------------------------------


def run_with_fetch_retries(
    task: Task, step: Callable[[daily.ErrorNote], dict[str, Any]]
) -> dict[str, Any]:
    """Run one attempt of a daily fetch step under the minutes-scale retry rule
    (app/ingest/retry.py, docs/adr/0003). `step` gets the error note for its run row.

    A failure before the last attempt is retried after the delay drawn for it (the retry
    keeps the rest of the chain, so a later success continues the chain). The last
    attempt's failure is raised, which stops the chain as before. A ValueError is raised
    at once, as with RETRY_OPTIONS."""
    plan = retry.plan_attempt(task.request.retries, get_settings())
    try:
        return step(plan.error_note)
    except ValueError:
        raise
    except Exception as err:
        if plan.delay_s is None:
            raise
        logger.warning("%s failed (%s): %s", task.name, plan.note, err)
        # max_retries is given here, so the setting decides (not the decorator's default).
        raise task.retry(exc=err, countdown=plan.delay_s, max_retries=plan.attempts - 1) from err


@celery_app.task(name="ingest.fetch_arxiv", bind=True)
def fetch_arxiv(self: Task) -> dict[str, Any]:
    def step(error_note: daily.ErrorNote) -> dict[str, Any]:
        with session_scope() as session:
            return daily.fetch_arxiv_daily(session, _arxiv(), error_note=error_note)

    return run_with_fetch_retries(self, step)


@celery_app.task(name="ingest.fetch_hf_daily", bind=True)
def fetch_hf_daily(self: Task) -> dict[str, Any]:
    from app.sources.hf import HFClient

    def step(error_note: daily.ErrorNote) -> dict[str, Any]:
        with session_scope() as session:
            return daily.fetch_hf_daily(session, HFClient(), _arxiv(), error_note=error_note)

    return run_with_fetch_retries(self, step)


@celery_app.task(name="ingest.index_abstracts", **RETRY_OPTIONS)
def index_abstracts() -> dict[str, Any]:
    from app.ingest.index import ensure_collection
    from app.ingest.index import index_abstracts as index_all

    ensure_collection()
    with session_scope() as session, daily.recorded_run(session, "index-abstracts") as counts:
        if budget.is_paused(session):
            # Unindexed abstracts are found again by the next run, so nothing is lost.
            counts["skipped"] = budget.PAUSED_REASON
            return counts
        counts["indexed"] = index_all(session, **_embedders())
    return counts


@celery_app.task(name="ingest.enrich_new", **RETRY_OPTIONS)
def enrich_new() -> dict[str, Any]:
    """GitHub, Hugging Face Hub and Semantic Scholar data for the papers added since the
    last run (records its own 'enrich-daily' run; see app/ingest/enrich.py)."""
    from app.ingest.enrich import default_enrich_clients, enrich_daily

    with session_scope() as session:
        return enrich_daily(session, default_enrich_clients())


@celery_app.task(name="ingest.select_fulltext", **RETRY_OPTIONS)
def select_fulltext() -> dict[str, Any]:
    """Start one full-text chain per selected paper: in the library, requested by hand, or
    among the `fulltext_top_n` best by relevance."""
    from app.features import app_settings

    with session_scope() as session, daily.recorded_run(session, "select-fulltext") as counts:
        top_n = app_settings.load(session).fulltext_top_n
        selected = list(select_for_fulltext(session, top_n=top_n))
        counts.update(selected=len(selected), top_n=top_n, arxiv_ids=selected[:50])
        if budget.is_paused(session):
            counts["skipped"] = budget.PAUSED_REASON
            return counts
        for arxiv_id in selected:
            paper_chain(arxiv_id).apply_async()
        # Papers whose text is stored but not indexed (an embed step skipped under the
        # budget cap, or a worker that died): clean, chunk and embed them again. All three
        # steps are idempotent and start from the stored text.
        waiting = list(stuck_papers(session))
        for arxiv_id in waiting:
            resume_chain(arxiv_id).apply_async()
        counts["resumed"] = len(waiting)
    return counts


def paper_chain(arxiv_id: str) -> Signature:
    return chain(
        fetch_fulltext.si(arxiv_id),
        clean_paper.si(arxiv_id),
        chunk_paper.si(arxiv_id),
        embed_paper.si(arxiv_id),
    )


def resume_chain(arxiv_id: str) -> Signature:
    """The steps after the download, for a paper whose text is already stored."""
    return chain(clean_paper.si(arxiv_id), chunk_paper.si(arxiv_id), embed_paper.si(arxiv_id))


def stuck_papers(session: Any) -> Sequence[str]:
    """Papers with stored text that no chain has indexed ('fetched')."""
    return list(
        session.scalars(
            select(Paper.arxiv_id)
            .where(Paper.fulltext_status == "fetched")
            .order_by(Paper.arxiv_id)
        )
    )


def daily_chain() -> Signature:
    return chain(
        fetch_arxiv.si(),
        fetch_hf_daily.si(),
        index_abstracts.si(),
        enrich_new.si(),
        celery_app.signature("digest.score_relevance", immutable=True),  # by name: no import cycle
        select_fulltext.si(),
    )


@celery_app.task(name="ingest.daily_pipeline")
def daily_pipeline() -> str:
    """Started by Beat at 07:00 Europe/Istanbul (or by hand): queue the daily chain."""
    result = daily_chain().apply_async()
    return str(result.id)


@celery_app.task(name="ingest.daily_catchup")
def daily_catchup() -> dict[str, Any]:
    """Sent by Beat at the catch-up times (13:00 and 19:00 Istanbul by default): send the
    daily chain again if no arXiv run has succeeded since 07:00 and none is in progress
    (app/ingest/catchup.py). Writes one 'daily-catchup' run either way. Not retried: it
    only reads Postgres and sends one message, and the next catch-up decides again."""
    with session_scope() as session, daily.recorded_run(session, catchup.SOURCE) as counts:
        decision = catchup.decide(session, datetime.now(UTC), get_settings())
        counts.update(action=decision.action, reason=decision.reason)
        if decision.start:
            counts["chain_id"] = str(daily_chain().apply_async().id)
    return counts


# --- per-paper full-text steps ----------------------------------------------------------------


def _status(session: Any, arxiv_id: str) -> str | None:
    paper = session.get(Paper, arxiv_id)
    return paper.fulltext_status if paper is not None else None


@celery_app.task(name="ingest.fetch_fulltext", base=PaperTask, **RETRY_OPTIONS)
def fetch_fulltext(arxiv_id: str) -> dict[str, Any]:
    with session_scope() as session, daily.recorded_run(session, "fetch-fulltext") as counts:
        result = fetch_fulltext_step(session, arxiv_id, sources=default_sources())
        counts.update(arxiv_id=arxiv_id, source=result.source, status=result.status)
    return counts


@celery_app.task(name="ingest.clean_paper", base=PaperTask, **RETRY_OPTIONS)
def clean_paper(arxiv_id: str) -> dict[str, Any]:
    with session_scope() as session, daily.recorded_run(session, "clean-paper") as counts:
        if _status(session, arxiv_id) != "fetched":  # no new text (e.g. unavailable)
            counts.update(arxiv_id=arxiv_id, skipped=True)
            return counts
        counts.update(arxiv_id=arxiv_id, tokens=pipeline.clean_stored_text(session, arxiv_id))
    return counts


@celery_app.task(name="ingest.chunk_paper", base=PaperTask, **RETRY_OPTIONS)
def chunk_paper(arxiv_id: str) -> dict[str, Any]:
    with session_scope() as session, daily.recorded_run(session, "chunk-paper") as counts:
        if _status(session, arxiv_id) != "fetched":
            counts.update(arxiv_id=arxiv_id, skipped=True)
            return counts
        counts.update(arxiv_id=arxiv_id, chunks=len(pipeline.chunk_stored_text(session, arxiv_id)))
    return counts


@celery_app.task(name="ingest.embed_paper", base=PaperTask, **RETRY_OPTIONS)
def embed_paper(arxiv_id: str) -> dict[str, Any]:
    from app.ingest.index import ensure_collection, upsert_paper_chunks

    ensure_collection()
    with session_scope() as session, daily.recorded_run(session, "embed-paper") as counts:
        if _status(session, arxiv_id) != "fetched":
            counts.update(arxiv_id=arxiv_id, skipped=True)
            return counts
        if budget.is_paused(session):
            counts.update(arxiv_id=arxiv_id, skipped=budget.PAUSED_REASON)
            return counts  # status stays 'fetched': select_fulltext resumes it later
        counts.update(
            arxiv_id=arxiv_id, embedded=upsert_paper_chunks(session, arxiv_id, **_embedders())
        )
        paper = session.get(Paper, arxiv_id)
        assert paper is not None
        paper.fulltext_status = "indexed"
    return counts


# --- one paper end to end ("add by arXiv id") -------------------------------------------------


@celery_app.task(name="ingest.paper", base=PaperTask, **RETRY_OPTIONS)
def ingest_paper(arxiv_id: str, force_source: str | None = None) -> dict[str, Any]:
    from app.ingest.index import ensure_collection
    from app.ingest.paper import default_clients

    ensure_collection()
    with session_scope() as session:
        result = pipeline.run_paper(
            session, arxiv_id, clients=default_clients(), sources=default_sources(),
            force_source=force_source, **_embedders(),  # type: ignore[arg-type]
        )  # fmt: skip
        with daily.recorded_run(session, "ingest-paper") as counts:
            counts.update(arxiv_id=result.arxiv_id, status=result.fulltext_status,
                          source=result.fulltext_source, chunks=result.chunks,
                          embedded=result.embedded)  # fmt: skip
    return counts

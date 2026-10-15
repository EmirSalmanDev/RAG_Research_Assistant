"""Celery tasks: one per pipeline step, each a thin wrapper around a function from
`daily.py`, `enrich.py`, `fulltext.py`, `pipeline.py` or `index.py` (no logic is
duplicated here).

The daily chain (Beat, 07:00 Europe/Istanbul):
    fetch_arxiv -> fetch_hf_daily -> index_abstracts -> enrich_new -> select_fulltext
and `select_fulltext` starts, for every selected paper, its own chain:
    fetch_fulltext -> clean_paper -> chunk_paper -> embed_paper

Rules every task follows:
- Immutable signatures (`.si()`): a task never receives the previous task's result,
  only ids, and reads everything else from Postgres. So each task can be re-run alone.
- Idempotent: re-running a task changes nothing that is already done.
- Retries: any error except a ValueError (bad input, retrying cannot help) is retried
  up to 3 times with exponential backoff and jitter. After the last retry a per-paper
  task marks its paper 'failed' (shown on the health page; a manual request resets it).
- Each run writes one `ingest_runs` row (source = the task name).
"""

import logging
from typing import Any

from celery import Task, chain
from celery.canvas import Signature

from app.db.models import Paper
from app.db.session import session_scope
from app.ingest import daily, pipeline
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


@celery_app.task(name="ingest.fetch_arxiv", **RETRY_OPTIONS)
def fetch_arxiv() -> dict[str, Any]:
    with session_scope() as session:
        return daily.fetch_arxiv_daily(session, _arxiv())


@celery_app.task(name="ingest.fetch_hf_daily", **RETRY_OPTIONS)
def fetch_hf_daily() -> dict[str, Any]:
    from app.sources.hf import HFClient

    with session_scope() as session:
        return daily.fetch_hf_daily(session, HFClient(), _arxiv())


@celery_app.task(name="ingest.index_abstracts", **RETRY_OPTIONS)
def index_abstracts() -> dict[str, Any]:
    from app.ingest.index import ensure_collection
    from app.ingest.index import index_abstracts as index_all

    ensure_collection()
    with session_scope() as session, daily.recorded_run(session, "index-abstracts") as counts:
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
    """Start one full-text chain per selected paper (library or requested)."""
    with session_scope() as session, daily.recorded_run(session, "select-fulltext") as counts:
        selected = list(select_for_fulltext(session))
        for arxiv_id in selected:
            paper_chain(arxiv_id).apply_async()
        counts.update(selected=len(selected), arxiv_ids=selected[:50])
    return counts


def paper_chain(arxiv_id: str) -> Signature:
    return chain(
        fetch_fulltext.si(arxiv_id),
        clean_paper.si(arxiv_id),
        chunk_paper.si(arxiv_id),
        embed_paper.si(arxiv_id),
    )


def daily_chain() -> Signature:
    return chain(
        fetch_arxiv.si(),
        fetch_hf_daily.si(),
        index_abstracts.si(),
        enrich_new.si(),
        select_fulltext.si(),
    )


@celery_app.task(name="ingest.daily_pipeline")
def daily_pipeline() -> str:
    """Started by Beat at 07:00 Europe/Istanbul (or by hand): queue the daily chain."""
    result = daily_chain().apply_async()
    return str(result.id)


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

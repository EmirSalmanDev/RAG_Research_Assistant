"""Celery tasks for paper analyses (thin wrappers around app/llm/batch.py).

- `analysis.submit` (Beat, daily): one Batch API job for the papers that need an
  analysis. It spends money, so it only runs when `analysis_auto_submit` is on (off by
  default); otherwise it records that it skipped.
- `analysis.collect` (Beat, every 30 minutes): checks open batches and stores and
  indexes the results of finished ones. Without open batches it makes no API call and
  writes nothing.

Each run that does work writes one `ingest_runs` row; both are safe to re-run.
"""

from typing import Any

from app.config import get_settings
from app.db.session import session_scope
from app.ingest.daily import recorded_run
from app.ingest.tasks import RETRY_OPTIONS
from app.worker.celery_app import celery_app


def _openai() -> Any:
    from app.ingest.embed import default_openai_client

    return default_openai_client()


@celery_app.task(name="analysis.submit", **RETRY_OPTIONS)
def submit_analyses() -> dict[str, Any]:
    from app.llm.batch import submit

    with session_scope() as session, recorded_run(session, "analysis-submit") as counts:
        if not get_settings().analysis_auto_submit:
            counts["skipped"] = "analysis_auto_submit is off"
            return counts
        counts.update(submit(session, _openai()))
    return counts


@celery_app.task(name="analysis.collect", **RETRY_OPTIONS)
def collect_analyses() -> dict[str, Any]:
    from app.ingest.embed import get_dense_embedder, get_sparse_embedder
    from app.ingest.index import ensure_collection
    from app.llm.batch import collect, open_batch_count

    with session_scope() as session:
        if open_batch_count(session) == 0:
            return {"open": 0}  # nothing to ask OpenAI about, and nothing to record
        ensure_collection()
        with recorded_run(session, "analysis-collect") as counts:
            counts.update(
                collect(
                    session, _openai(), dense=get_dense_embedder(), sparse=get_sparse_embedder()
                )
            )
    return counts

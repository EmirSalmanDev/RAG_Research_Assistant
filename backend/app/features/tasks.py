"""Celery tasks of the daily digest (thin wrappers around app/features/digest.py).

- `digest.score_relevance` is step 4 of the nightly chain (after `enrich_new`, before
  `select_fulltext`, see app/ingest/tasks.py): it embeds interests whose text changed,
  then stores `relevance_score` for the papers of the window. The interest embedding is
  the only paid call (a few hundred tokens, once per edit); if it fails the scoring still
  runs with the vectors it has and the error is recorded in the run.
- `digest.build` (Beat, 08:00 Europe/Istanbul) writes the day's digest. A second run for
  the same day changes nothing.

Each run writes one `ingest_runs` row (source 'score-relevance' / 'build-digest').
"""

from datetime import date
from typing import Any

from app.db.session import session_scope
from app.features import digest
from app.ingest.daily import recorded_run
from app.ingest.tasks import RETRY_OPTIONS
from app.worker.celery_app import celery_app


def _dense() -> Any:
    from app.ingest.embed import get_dense_embedder

    return get_dense_embedder()


def _vectors() -> digest.AbstractVectors:
    return digest.QdrantAbstractVectors()


@celery_app.task(name="digest.score_relevance", **RETRY_OPTIONS)
def score_relevance() -> dict[str, Any]:
    with session_scope() as session, recorded_run(session, "score-relevance") as counts:
        try:
            counts["interests_embedded"] = digest.ensure_interest_embeddings(session, _dense())
            session.commit()  # keep the vectors that cost money even if scoring fails later
        except Exception as err:  # no key, API down: score with the vectors we have
            session.rollback()
            counts["embed_error"] = f"{type(err).__name__}: {err}"[:300]
        counts.update(digest.score_window(session, _vectors()))
    return counts


@celery_app.task(name="digest.build", **RETRY_OPTIONS)
def build_digest(digest_date: str | None = None, rebuild: bool = False) -> dict[str, Any]:
    """Build the digest of `digest_date` (ISO date; default today in Istanbul)."""
    day = date.fromisoformat(digest_date) if digest_date else None
    with session_scope() as session, recorded_run(session, "build-digest") as counts:
        counts.update(
            digest.build_digest(session, _vectors(), digest_date=day, rebuild=rebuild).counts()
        )
    return counts

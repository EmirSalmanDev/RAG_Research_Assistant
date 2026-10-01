"""Celery app: Redis broker, the ingest tasks, and the daily Beat schedule."""

from celery import Celery

from app.config import get_settings
from app.worker.schedule import BEAT_SCHEDULE

celery_app = Celery(
    "research_copilot",
    broker=get_settings().redis_url,
    backend=get_settings().redis_url,
    include=["app.ingest.tasks"],
)
celery_app.conf.update(
    worker_concurrency=1,  # 8 GB RAM machine: one worker process
    broker_connection_retry_on_startup=True,
    timezone="Europe/Istanbul",  # Beat reads crontab times in this zone
    enable_utc=True,  # timestamps are still stored in UTC
    # A task is acknowledged only after it finishes, so one killed mid-run (worker
    # restart, crash) is run again; safe because every task is idempotent.
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # take one task at a time (long tasks, one process)
    beat_schedule=BEAT_SCHEDULE,
)

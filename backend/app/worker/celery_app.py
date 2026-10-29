"""Celery app: Redis broker, the ingest, analysis and digest tasks, and the Beat schedule."""

from celery import Celery

from app.config import BROKER_VISIBILITY_TIMEOUT_S, SCHEDULE_TIMEZONE, get_settings
from app.worker.schedule import BEAT_SCHEDULE

celery_app = Celery(
    "research_copilot",
    broker=get_settings().redis_url,
    backend=get_settings().redis_url,
    include=["app.ingest.tasks", "app.llm.tasks", "app.features.tasks"],
)
celery_app.conf.update(
    worker_concurrency=1,  # 8 GB RAM machine: one worker process
    broker_connection_retry_on_startup=True,
    timezone=SCHEDULE_TIMEZONE,  # Beat reads crontab times in this zone (Europe/Istanbul)
    enable_utc=True,  # timestamps are still stored in UTC
    # A task is acknowledged only after it finishes, so one killed mid-run (worker
    # restart, crash) is run again; safe because every task is idempotent.
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # take one task at a time (long tasks, one process)
    # Redis delivers an unacknowledged message again after this many seconds. A retry that
    # waits for its countdown is held unacknowledged, so the daily fetch retries' longest
    # delay must stay below it (app/config.py refuses settings that break this).
    broker_transport_options={"visibility_timeout": BROKER_VISIBILITY_TIMEOUT_S},
    beat_schedule=BEAT_SCHEDULE,
)

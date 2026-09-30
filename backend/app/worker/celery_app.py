"""Minimal Celery app: Redis broker, no tasks yet (they arrive with the ingest milestone)."""

from celery import Celery

from app.config import get_settings

celery_app = Celery(
    "research_copilot",
    broker=get_settings().redis_url,
    backend=get_settings().redis_url,
)
celery_app.conf.update(
    worker_concurrency=1,  # 8 GB RAM machine: one worker process
    broker_connection_retry_on_startup=True,
)

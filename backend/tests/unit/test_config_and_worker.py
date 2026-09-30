from app.config import Settings
from app.worker.celery_app import celery_app


def test_settings_defaults_point_at_local_stores() -> None:
    s = Settings(_env_file=None)
    assert s.database_url.startswith("postgresql+psycopg://")
    assert s.qdrant_url.startswith("http")
    assert s.redis_url.startswith("redis://")
    assert s.openai_api_key is None


def test_celery_app_is_configured_for_low_memory() -> None:
    assert celery_app.conf.worker_concurrency == 1
    assert celery_app.conf.broker_url.startswith("redis://")

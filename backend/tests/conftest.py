"""Shared fixtures: a throwaway Postgres database per test, and cassette hygiene."""

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

from app.config import Settings


@pytest.fixture
def scratch_db_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[URL]:
    """Create an empty database `copilot_test_<random>`, yield its URL, drop it afterwards.

    `DATABASE_URL` is pointed at it (and the settings cache cleared) so Alembic's
    env.py, which reads the URL from app settings, migrates the scratch DB.
    """
    from app.config import get_settings
    from app.db.session import get_engine

    base = make_url(Settings(_env_file=None).database_url)
    name = f"copilot_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = base.set(database=name)
    monkeypatch.setenv("DATABASE_URL", url.render_as_string(hide_password=False))
    get_settings.cache_clear()
    get_engine.cache_clear()
    try:
        yield url
    finally:
        get_engine().dispose()
        get_settings.cache_clear()
        get_engine.cache_clear()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def migrated_session(scratch_db_url: URL) -> Iterator[Session]:
    """A session on a scratch database with all migrations applied."""
    from alembic import command
    from tests.helpers import alembic_config

    command.upgrade(alembic_config(), "head")
    engine = create_engine(scratch_db_url)
    with Session(engine) as session:
        yield session
    engine.dispose()

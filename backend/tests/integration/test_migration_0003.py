"""Migration 0003 (messages.meta) on a throwaway database holding a chat written before it.
Upgrade, downgrade, upgrade again; never runs on the real database (`scratch_db_url`)."""

import json
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL

from alembic import command
from tests.helpers import alembic_config

pytestmark = pytest.mark.integration

CONVERSATION = uuid.UUID("00000000-0000-4000-8000-000000000001")


@pytest.fixture
def engine(scratch_db_url: URL) -> Iterator[Engine]:
    eng = create_engine(scratch_db_url)
    yield eng
    eng.dispose()


def _message_columns(engine: Engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("messages")}


def test_0003_adds_a_nullable_meta_and_keeps_existing_messages(engine: Engine) -> None:
    command.upgrade(alembic_config(), "0002")
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO conversations (id, title) VALUES (:id, 'invented')"),
            {"id": CONVERSATION},
        )
        conn.execute(
            text(
                "INSERT INTO messages (conversation_id, role, content, citations)"
                " VALUES (:c, 'user', 'invented question', '[]'::jsonb),"
                " (:c, 'assistant', 'invented answer [1].', CAST(:cit AS jsonb))"
            ),
            {"c": CONVERSATION, "cit": json.dumps([{"n": 1, "chunk_type": "abstract"}])},
        )
    assert "meta" not in _message_columns(engine)

    command.upgrade(alembic_config(), "0003")
    meta_col = next(c for c in inspect(engine).get_columns("messages") if c["name"] == "meta")
    assert meta_col["nullable"] is True
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT role, content, meta FROM messages ORDER BY id")).all()
        assert [tuple(r) for r in rows] == [
            ("user", "invented question", None),
            ("assistant", "invented answer [1].", None),
        ]
        conn.execute(
            text("UPDATE messages SET meta = '{\"version\": 1}'::jsonb WHERE role = 'assistant'")
        )

    command.downgrade(alembic_config(), "0002")
    assert "meta" not in _message_columns(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM messages")).scalar_one() == 2

    command.upgrade(alembic_config(), "head")
    assert "meta" in _message_columns(engine)

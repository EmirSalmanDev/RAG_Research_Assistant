"""Migration 0005 (interest weights, app settings, digest entries) on a throwaway
database holding an interest written before it. Upgrade, downgrade, upgrade again;
never runs on the real database."""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

from alembic import command
from tests.helpers import alembic_config

pytestmark = pytest.mark.integration

PAPER = "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories, published_at) VALUES (:id, 1, 'Invented', 'Invented.', '[]', '{cs.CL}', now())"  # noqa: E501
ENTRY = "INSERT INTO digest_entries (digest_date, arxiv_id, rank, score) VALUES ('2026-10-19', :id, 1, 0.5)"  # noqa: E501
NEW_COLUMNS = {
    "weight", "embedding", "embedding_model", "embedding_hash", "is_active", "created_at",
    "updated_at",
}  # fmt: skip


@pytest.fixture
def engine(scratch_db_url: URL) -> Iterator[Engine]:
    eng = create_engine(scratch_db_url)
    yield eng
    eng.dispose()


def _columns(engine: Engine, table: str) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns(table)}


def test_0005_keeps_interests_and_adds_digest_tables(engine: Engine) -> None:
    command.upgrade(alembic_config(), "0004")
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO interests (name, description, keywords, categories) "
                "VALUES ('Invented', 'An invented topic.', '{a,b}', '{cs.CL}')"
            )
        )
        conn.execute(text(PAPER), {"id": "2601.00001"})

    command.upgrade(alembic_config(), "0005")
    assert NEW_COLUMNS <= _columns(engine, "interests")
    assert {"app_settings", "digest_entries"} <= set(inspect(engine).get_table_names())
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT name, keywords, weight, embedding, embedding_hash, is_active, "
                "created_at IS NOT NULL FROM interests"
            )  # fmt: skip
        ).one()
        # the existing row is kept; the new columns hold their defaults
        assert tuple(row) == ("Invented", ["a", "b"], 1.0, None, None, True, True)
        # a 1536-number embedding fits in real[]
        conn.execute(text("UPDATE interests SET embedding = :v"), {"v": [0.25] * 1536})
        assert conn.execute(text("SELECT cardinality(embedding) FROM interests")).scalar() == 1536

    with engine.begin() as conn:
        conn.execute(text(ENTRY), {"id": "2601.00001"})
        row = conn.execute(text("SELECT status, weight_delta, best_interest_id "
                                "FROM digest_entries")).one()  # fmt: skip
        assert tuple(row) == ("new", 0.0, None)
    with pytest.raises(IntegrityError), engine.begin() as conn:  # one row per paper and date
        conn.execute(text(ENTRY), {"id": "2601.00001"})
    with pytest.raises(IntegrityError), engine.begin() as conn:  # status is checked
        conn.execute(text("UPDATE digest_entries SET status = 'maybe'"))
    with engine.begin() as conn:  # deleting an interest keeps the entry
        conn.execute(text("UPDATE digest_entries SET best_interest_id = 1"))
        conn.execute(text("DELETE FROM interests"))
        assert conn.execute(text("SELECT best_interest_id FROM digest_entries")).scalar() is None
    with engine.begin() as conn:  # deleting a paper removes its entries
        conn.execute(text("DELETE FROM papers"))
        assert conn.execute(text("SELECT count(*) FROM digest_entries")).scalar() == 0

    command.downgrade(alembic_config(), "0004")
    assert not {"app_settings", "digest_entries"} & set(inspect(engine).get_table_names())
    assert not NEW_COLUMNS & _columns(engine, "interests")
    assert {"name", "description", "keywords", "categories"} <= _columns(engine, "interests")

    command.upgrade(alembic_config(), "head")  # and the upgrade works again
    assert "digest_entries" in inspect(engine).get_table_names()

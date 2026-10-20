"""Migration 0006 (library collections) on a throwaway database holding a library item
written before it. Upgrade, downgrade, upgrade again; never runs on the real database."""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

from alembic import command
from tests.helpers import alembic_config

pytestmark = pytest.mark.integration

PAPER = "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories, published_at) VALUES (:id, 1, 'Invented', 'Invented.', '[]', '{cs.CL}', now())"  # noqa: E501


@pytest.fixture
def engine(scratch_db_url: URL) -> Iterator[Engine]:
    eng = create_engine(scratch_db_url)
    yield eng
    eng.dispose()


def test_0006_adds_the_collections_table_and_keeps_the_items(engine: Engine) -> None:
    command.upgrade(alembic_config(), "0005")
    with engine.begin() as conn:
        conn.execute(text(PAPER), {"id": "2601.00001"})
        conn.execute(
            text(
                "INSERT INTO library_items (arxiv_id, collection, status) "
                "VALUES ('2601.00001', 'Saved', 'to_read')"
            )
        )

    command.upgrade(alembic_config(), "0006")
    assert "library_collections" in inspect(engine).get_table_names()
    indexes = {i["name"] for i in inspect(engine).get_indexes("library_items")}
    assert "ix_library_items_collection" in indexes
    with engine.begin() as conn:  # the existing item is untouched
        row = conn.execute(text("SELECT arxiv_id, collection, status FROM library_items")).one()
        assert tuple(row) == ("2601.00001", "Saved", "to_read")
        conn.execute(text("INSERT INTO library_collections (name) VALUES ('Thesis')"))
        row = conn.execute(
            text("SELECT name, description, created_at IS NOT NULL FROM library_collections")
        ).one()
        assert tuple(row) == ("Thesis", None, True)
    with pytest.raises(IntegrityError), engine.begin() as conn:  # the name is unique
        conn.execute(text("INSERT INTO library_collections (name) VALUES ('Thesis')"))
    with pytest.raises(IntegrityError), engine.begin() as conn:  # one item per paper+collection
        conn.execute(
            text("INSERT INTO library_items (arxiv_id, collection) VALUES ('2601.00001', 'Saved')")
        )

    command.downgrade(alembic_config(), "0005")
    assert "library_collections" not in inspect(engine).get_table_names()
    assert "ix_library_items_collection" not in {
        i["name"] for i in inspect(engine).get_indexes("library_items")
    }
    with engine.begin() as conn:  # the items stay
        assert conn.execute(text("SELECT count(*) FROM library_items")).scalar() == 1

    command.upgrade(alembic_config(), "head")  # and the upgrade works again
    assert "library_collections" in inspect(engine).get_table_names()

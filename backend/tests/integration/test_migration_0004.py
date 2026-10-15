"""Migration 0004 (analysis batches) on a throwaway database holding an analysis written
before it. Upgrade, downgrade, upgrade again; never runs on the real database."""

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


def _columns(engine: Engine, table: str) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns(table)}


def test_0004_adds_batch_tables_and_keeps_existing_analyses(engine: Engine) -> None:
    command.upgrade(alembic_config(), "0003")
    with engine.begin() as conn:
        conn.execute(text(PAPER), {"id": "2601.00001"})
        conn.execute(text("INSERT INTO paper_analyses (arxiv_id, tldr, model) "
                          "VALUES ('2601.00001', 'Invented.', 'm')"))  # fmt: skip

    command.upgrade(alembic_config(), "0004")
    assert {"analysis_batches", "analysis_batch_items"} <= set(inspect(engine).get_table_names())
    assert {"prompt_version", "batch_id"} <= _columns(engine, "paper_analyses")
    with engine.begin() as conn:
        row = conn.execute(text("SELECT tldr, prompt_version, batch_id FROM paper_analyses")).one()
        assert tuple(row) == ("Invented.", None, None)  # existing row kept, new columns NULL

    with engine.begin() as conn:
        conn.execute(text("INSERT INTO analysis_batches (status, model, prompt_version, "
                          "request_count) VALUES ('created', 'm', 'analyze_v1', 1), "
                          "('created', 'm', 'analyze_v1', 1)"))  # fmt: skip
        assert conn.execute(text("SELECT errors FROM analysis_batches LIMIT 1")).scalar() == {}
        conn.execute(text("INSERT INTO analysis_batch_items (batch_id, arxiv_id) "
                          "VALUES (1, '2601.00001')"))  # fmt: skip
    # At most one pending item per paper; a done or failed one does not count.
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("INSERT INTO analysis_batch_items (batch_id, arxiv_id) "
                          "VALUES (2, '2601.00001')"))  # fmt: skip
    with engine.begin() as conn:
        conn.execute(text("UPDATE analysis_batch_items SET status = 'done'"))
        conn.execute(text("INSERT INTO analysis_batch_items (batch_id, arxiv_id) "
                          "VALUES (2, '2601.00001')"))  # fmt: skip
    # One analysis per paper.
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("INSERT INTO paper_analyses (arxiv_id, model) "
                          "VALUES ('2601.00001', 'm')"))  # fmt: skip

    command.downgrade(alembic_config(), "0003")
    tables = set(inspect(engine).get_table_names())
    assert not {"analysis_batches", "analysis_batch_items"} & tables
    assert not {"prompt_version", "batch_id"} & _columns(engine, "paper_analyses")
    with engine.begin() as conn:  # the analysis itself survives the downgrade
        assert conn.execute(text("SELECT tldr FROM paper_analyses")).scalar() == "Invented."

    command.upgrade(alembic_config(), "head")  # and the upgrade works again
    assert "analysis_batches" in inspect(engine).get_table_names()

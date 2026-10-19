"""The migration builds every table on a fresh database and can be rolled back."""

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import URL

from alembic import command
from app.db import models  # noqa: F401  (registers tables)
from app.db.base import Base
from tests.helpers import alembic_config

pytestmark = pytest.mark.integration

EXPECTED_TABLES = {
    "papers", "paper_texts", "paper_analyses", "chunks", "repos", "hf_artifacts",
    "citations", "interests", "library_items", "conversations", "messages",
    "ingest_runs", "eval_items", "eval_runs", "eval_results", "llm_usage", "analysis_batches",
    "analysis_batch_items", "app_settings", "digest_entries",
}  # fmt: skip


def test_upgrade_creates_all_tables_and_downgrade_removes_them(scratch_db_url: URL) -> None:
    engine = create_engine(scratch_db_url)
    command.upgrade(alembic_config(), "head")
    insp = inspect(engine)
    assert EXPECTED_TABLES <= set(insp.get_table_names())

    # Every foreign key column is indexed (as an index or as the leading PK/unique column).
    for table in EXPECTED_TABLES:
        indexed = [ix["column_names"][0] for ix in insp.get_indexes(table)]
        pk = insp.get_pk_constraint(table)["constrained_columns"][:1]
        uniques = [u["column_names"][0] for u in insp.get_unique_constraints(table)]
        for fk in insp.get_foreign_keys(table):
            col = fk["constrained_columns"][0]
            assert col in indexed + pk + uniques, f"{table}.{col} has no index"

    assert "ix_papers_published_at" in {ix["name"] for ix in insp.get_indexes("papers")}

    command.downgrade(alembic_config(), "base")
    assert EXPECTED_TABLES.isdisjoint(inspect(engine).get_table_names())
    engine.dispose()


def test_models_match_migration(scratch_db_url: URL) -> None:
    """Autogenerate finds no difference between models.py and the migrated database."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    command.upgrade(alembic_config(), "head")
    engine = create_engine(scratch_db_url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    assert diff == []

"""ingest columns: primary category, project page, full-text versions and references

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-01

Non-destructive: only adds columns and one CHECK constraint. Existing rows are kept and
the new columns are filled from data already in the tables.

After this migration `paper_texts` holds two versions of a paper's text:
- `raw_markdown`: the text as fetched or converted (HF .md, arXiv HTML or PDF -> markdown);
- `markdown`: the CLEANED text, which is what gets chunked.
Before this migration only `markdown` existed, so an existing row is copied into
`raw_markdown` as well (cleaning is idempotent, so cleaning it again later is safe).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Same values as app.db.models.FULLTEXT_STATUSES. Copied on purpose: a migration must not
# import app code that may change after the migration was written.
FULLTEXT_STATUSES = ("none", "requested", "fetched", "indexed", "unavailable", "failed")
STATUS_LIST = ", ".join(f"'{status}'" for status in FULLTEXT_STATUSES)

# Naming convention in app/db/base.py: "ck_%(table_name)s_%(constraint_name)s", and the
# model names the constraint "fulltext_status". op.f() marks the name as final, so the
# convention is not applied a second time (same style as migration 0001).
CK_FULLTEXT_STATUS = "ck_papers_fulltext_status"


def upgrade() -> None:
    # --- papers ---------------------------------------------------------------------
    op.add_column("papers", sa.Column("primary_category", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("project_url", sa.Text(), nullable=True))

    # arXiv lists the primary category first and M1 stored the list in that order, so
    # existing rows can be filled from it. Postgres arrays are 1-based.
    op.execute(
        "UPDATE papers SET primary_category = categories[1] "
        "WHERE primary_category IS NULL AND cardinality(categories) > 0"
    )

    # M1 only ever wrote the default 'none'. Anything else is mapped to 'none' first, so
    # adding the CHECK constraint cannot fail on existing rows.
    op.execute(
        f"UPDATE papers SET fulltext_status = 'none' WHERE fulltext_status NOT IN ({STATUS_LIST})"
    )
    op.create_check_constraint(
        op.f(CK_FULLTEXT_STATUS), "papers", f"fulltext_status IN ({STATUS_LIST})"
    )

    # --- paper_texts ----------------------------------------------------------------
    # Added as nullable, filled from the existing rows, then made NOT NULL.
    op.add_column("paper_texts", sa.Column("version", sa.Integer(), nullable=True))
    op.add_column("paper_texts", sa.Column("raw_markdown", sa.Text(), nullable=True))
    op.add_column("paper_texts", sa.Column("references", sa.Text(), nullable=True))
    op.execute(
        "UPDATE paper_texts AS t SET version = p.version, raw_markdown = t.markdown "
        "FROM papers AS p WHERE p.arxiv_id = t.arxiv_id"
    )
    op.alter_column("paper_texts", "version", nullable=False)
    op.alter_column("paper_texts", "raw_markdown", nullable=False)


def downgrade() -> None:
    # Drops only what upgrade() added; the values stored in these columns are lost.
    op.drop_column("paper_texts", "references")
    op.drop_column("paper_texts", "raw_markdown")
    op.drop_column("paper_texts", "version")
    op.drop_constraint(op.f(CK_FULLTEXT_STATUS), "papers", type_="check")
    op.drop_column("papers", "project_url")
    op.drop_column("papers", "primary_category")

"""analysis batches: track paper analyses sent to the OpenAI Batch API

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-14

Non-destructive: two new tables, two nullable columns, one unique constraint and two
indexes. Nothing is dropped or rewritten (paper_analyses had no rows when this was
written; the unique constraint fails loudly if a database holds duplicates).

- `analysis_batches`: one row per Batch API job: its OpenAI ids (batch, input, output
  and error files), its status (`created` before OpenAI knows it, then OpenAI's own
  status, then `collected`), the model and prompt version, and the request count.
- `analysis_batch_items`: one row per paper in a batch. A paper's queued batch is the
  item with status `pending`; a partial unique index allows at most one such item per
  paper, so a paper is never queued twice.
- `paper_analyses`: `prompt_version` (rule 7) and `batch_id` (the batch that produced it)
  are added, and `arxiv_id` becomes unique: one analysis per paper, replaced when a
  newer prompt version analyses the paper again.

The downgrade removes exactly these additions (and with them any batch bookkeeping and
the provenance of analyses written since the upgrade; the analyses themselves stay).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analysis_batches",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("openai_batch_id", sa.Text(), unique=True),  # NULL until OpenAI created it
        sa.Column("input_file_id", sa.Text()),
        sa.Column("output_file_id", sa.Text()),
        sa.Column("error_file_id", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.Text(), nullable=False),
        sa.Column("request_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("submitted_at", sa.DateTime(timezone=True)),
        sa.Column("collected_at", sa.DateTime(timezone=True)),
        sa.Column(
            "errors",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.create_table(
        "analysis_batch_items",
        sa.Column(
            "batch_id",
            sa.BigInteger(),
            sa.ForeignKey("analysis_batches.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "arxiv_id",
            sa.Text(),
            sa.ForeignKey("papers.arxiv_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("error", sa.Text()),
    )
    op.create_index(
        "uq_batch_items_one_pending",
        "analysis_batch_items",
        ["arxiv_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.add_column("paper_analyses", sa.Column("prompt_version", sa.Text()))
    op.add_column(
        "paper_analyses",
        sa.Column(
            "batch_id",
            sa.BigInteger(),
            sa.ForeignKey(
                "analysis_batches.id",
                ondelete="SET NULL",
                name="paper_analyses_batch_id_fkey",
            ),
        ),
    )
    # Every foreign key column is indexed (a repo-wide rule, checked by test_migration).
    op.create_index("ix_paper_analyses_batch_id", "paper_analyses", ["batch_id"])
    op.create_unique_constraint("uq_paper_analyses_arxiv_id", "paper_analyses", ["arxiv_id"])


def downgrade() -> None:
    op.drop_constraint("uq_paper_analyses_arxiv_id", "paper_analyses", type_="unique")
    op.drop_index("ix_paper_analyses_batch_id", table_name="paper_analyses")
    op.drop_constraint("paper_analyses_batch_id_fkey", "paper_analyses", type_="foreignkey")
    op.drop_column("paper_analyses", "batch_id")
    op.drop_column("paper_analyses", "prompt_version")
    op.drop_index("uq_batch_items_one_pending", table_name="analysis_batch_items")
    op.drop_table("analysis_batch_items")
    op.drop_table("analysis_batches")

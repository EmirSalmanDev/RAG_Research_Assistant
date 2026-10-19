"""daily digest: interest weights and embeddings, app settings, digest entries

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-19

Non-destructive: seven new nullable-or-defaulted columns on `interests` and two new
tables. No row is dropped or rewritten.

- `interests`: `weight` (default 1.0; nudged by Save and Dismiss), `embedding` (real[],
  NULL until the scoring task embeds the interest), `embedding_model` and
  `embedding_hash` (hash of the embedded text, so an edit triggers a new embedding),
  `is_active` (default true) and `created_at` / `updated_at` (existing rows get the
  time of the migration).
- `app_settings`: key -> jsonb value for the settings page (digest size, categories
  override, budget cap, full-text top N). A missing key means the config default.
- `digest_entries`: one row per paper per day's digest, with its rank, score, best
  interest, status (new, saved, dismissed) and the exact weight change Save or Dismiss
  made, so that Undo can reverse it. Unique per (digest_date, arxiv_id); the foreign
  key columns are indexed.

The downgrade drops the two tables and the new columns (with the embeddings, weights
and digests created since the upgrade; the interests themselves stay).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0005"
down_revision: str | Sequence[str] | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "interests",
        sa.Column("weight", sa.Float(), server_default="1.0", nullable=False),
    )
    op.add_column("interests", sa.Column("embedding", postgresql.ARRAY(postgresql.REAL())))
    op.add_column("interests", sa.Column("embedding_model", sa.Text()))
    op.add_column("interests", sa.Column("embedding_hash", sa.Text()))
    op.add_column(
        "interests",
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
    )
    op.add_column(
        "interests",
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.add_column(
        "interests",
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "app_settings",
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_app_settings")),
    )
    op.create_table(
        "digest_entries",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("digest_date", sa.Date(), nullable=False),
        sa.Column("arxiv_id", sa.Text(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("best_interest_id", sa.Integer()),
        sa.Column("status", sa.Text(), server_default="new", nullable=False),
        sa.Column("weight_delta", sa.Float(), server_default="0", nullable=False),
        sa.Column("actioned_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('new', 'saved', 'dismissed')", name=op.f("ck_digest_entries_status")
        ),
        sa.ForeignKeyConstraint(
            ["arxiv_id"],
            ["papers.arxiv_id"],
            name=op.f("fk_digest_entries_arxiv_id_papers"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["best_interest_id"],
            ["interests.id"],
            name=op.f("fk_digest_entries_best_interest_id_interests"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_digest_entries")),
        sa.UniqueConstraint(
            "digest_date", "arxiv_id", name=op.f("uq_digest_entries_digest_date_arxiv_id")
        ),
    )
    op.create_index(op.f("ix_digest_entries_arxiv_id"), "digest_entries", ["arxiv_id"])
    op.create_index(
        op.f("ix_digest_entries_best_interest_id"), "digest_entries", ["best_interest_id"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_digest_entries_best_interest_id"), table_name="digest_entries")
    op.drop_index(op.f("ix_digest_entries_arxiv_id"), table_name="digest_entries")
    op.drop_table("digest_entries")
    op.drop_table("app_settings")
    for column in (
        "updated_at", "created_at", "is_active", "embedding_hash", "embedding_model",
        "embedding", "weight",
    ):  # fmt: skip
        op.drop_column("interests", column)

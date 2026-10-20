"""library collections: a table for the names (and descriptions) of collections

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-20

Non-destructive: one new table and one new index. No row is dropped or rewritten.

- `library_collections`: one row per named collection (unique name, optional description).
  A collection exists when it has a row here OR when `library_items` has an item with that
  collection name, so the "Saved" collection that the digest's Save button creates needs no
  seeding, and an empty collection can be created.
- `library_items` already has the primary key (arxiv_id, collection), i.e. one item per
  paper and collection, so no constraint is added there. Only an index on `collection`
  (listing and renaming a collection filters by it).

The downgrade drops the table and the index; the items stay.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006"
down_revision: str | Sequence[str] | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "library_collections",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_library_collections")),
        sa.UniqueConstraint("name", name=op.f("uq_library_collections_name")),
    )
    op.create_index(op.f("ix_library_items_collection"), "library_items", ["collection"])


def downgrade() -> None:
    op.drop_index(op.f("ix_library_items_collection"), table_name="library_items")
    op.drop_table("library_collections")

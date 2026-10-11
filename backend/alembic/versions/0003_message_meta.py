"""message meta: one nullable jsonb column on messages

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-03

Non-destructive: adds `messages.meta` (jsonb, NULL allowed). Existing rows keep NULL.

What it holds (assistant messages only; validated in the app by a pydantic model with
a `version` field, so older rows stay readable when the shape grows): the intent, the
checked search plan, the per-sentence verification, the stages that were skipped or
fell back after an error, the prompt versions and the standalone question.

Why one jsonb column and not typed columns: this shape will keep changing while the
chat features are built (verification output, filter chips from the plan, comparisons),
and a jsonb column avoids a migration for every change. The price is that type checking
moves from the database to the application. A field that becomes stable and is queried
often can move to its own typed column later.

The downgrade drops the column, and with it any meta written since the upgrade.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003"
down_revision: str | Sequence[str] | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("meta", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "meta")

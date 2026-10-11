"""record_chat_usage writes one committed llm_usage row (scratch database)."""

from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import LLMUsage
from app.llm.usage import record_chat_usage

pytestmark = pytest.mark.integration


def test_one_committed_row_with_both_token_counts(migrated_session: Session) -> None:
    cost = record_chat_usage("chat-generate", "gpt-4o-mini", 4000, 500)
    assert cost == Decimal("0.000900")
    row = migrated_session.scalars(select(LLMUsage)).one()  # visible: it was committed
    assert (row.feature, row.model, row.tokens_in, row.tokens_out, row.cost_usd) == (
        "chat-generate",
        "gpt-4o-mini",
        4000,
        500,
        Decimal("0.000900"),
    )

"""Record what every LLM / embedding call costs, in the `llm_usage` table (rule 9).

One row per API request. The row is written in its own short transaction and committed
at once: the money is spent when the request succeeds, even if the work that asked for
it fails later and rolls back, so the spend record must not roll back with it.
"""

from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import LLMUsage

# (feature, model, tokens_in) -> None. Embedding requests have no output tokens.
UsageRecorder = Callable[[str, str, int], None]

_MICRO_DOLLAR = Decimal("0.000001")  # llm_usage.cost_usd is NUMERIC(12, 6)


def embedding_cost(tokens: int) -> Decimal:
    """USD for `tokens` input tokens at the configured embedding price."""
    price_per_1m = Decimal(str(get_settings().embedding_price_per_1m))
    return (Decimal(tokens) * price_per_1m / 1_000_000).quantize(_MICRO_DOLLAR, ROUND_HALF_UP)


def log_usage(
    session: Session,
    *,
    feature: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: Decimal,
) -> LLMUsage:
    """Add one llm_usage row to `session` (the caller decides when to commit)."""
    row = LLMUsage(
        feature=feature,
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
    )
    session.add(row)
    session.flush()
    return row


def record_embedding_usage(feature: str, model: str, tokens_in: int) -> None:
    """The default UsageRecorder: one committed row per embedding request."""
    from app.db.session import session_scope

    with session_scope() as session:
        log_usage(
            session,
            feature=feature,
            model=model,
            tokens_in=tokens_in,
            tokens_out=0,
            cost_usd=embedding_cost(tokens_in),
        )


def session_recorder(session: Session) -> UsageRecorder:
    """A UsageRecorder that writes into an existing session (used by tests)."""

    def record(feature: str, model: str, tokens_in: int) -> None:
        log_usage(
            session,
            feature=feature,
            model=model,
            tokens_in=tokens_in,
            tokens_out=0,
            cost_usd=embedding_cost(tokens_in),
        )

    return record

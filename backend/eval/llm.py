"""The evaluation's model client: the eval model, temperature 0, every request logged.

Test-set generation and the judge both talk to the model through `eval_chat_client()`:
the same ChatClient as the chat (retries, Structured Outputs, usage logging), but with
the evaluation's own model, temperature and prices from the settings (`eval_model`,
`eval_temperature`, `eval_price_*`). Every successful request writes one `llm_usage` row
at those prices (rule 9), with a feature name starting with `eval-`.
"""

from decimal import ROUND_HALF_UP, Decimal

from app.config import get_settings
from app.llm.client import ChatClient
from app.llm.usage import log_usage

_MICRO_DOLLAR = Decimal("0.000001")  # llm_usage.cost_usd is NUMERIC(12, 6)


def eval_cost(tokens_in: int, tokens_out: int) -> Decimal:
    """USD for one request at the configured evaluation prices."""
    s = get_settings()
    cost = (
        Decimal(tokens_in) * Decimal(str(s.eval_price_in_per_1m))
        + Decimal(tokens_out) * Decimal(str(s.eval_price_out_per_1m))
    ) / 1_000_000
    return cost.quantize(_MICRO_DOLLAR, ROUND_HALF_UP)


def record_eval_usage(feature: str, model: str, tokens_in: int, tokens_out: int) -> Decimal:
    """One committed llm_usage row per evaluation request (a ChatUsageRecorder)."""
    from app.db.session import session_scope

    cost = eval_cost(tokens_in, tokens_out)
    with session_scope() as session:
        log_usage(
            session,
            feature=feature,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )
    return cost


def eval_chat_client() -> ChatClient:
    """The client for test-set generation and judging (no request is sent until used)."""
    s = get_settings()
    return ChatClient(
        record_usage=record_eval_usage, model=s.eval_model, temperature=s.eval_temperature
    )

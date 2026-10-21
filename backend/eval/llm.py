"""The evaluation's model client: the eval model, temperature 0, every request logged.

Test-set generation and the judge both talk to the model through `eval_chat_client()`:
the same ChatClient as the chat (retries, Structured Outputs, usage logging), but with
the evaluation's own model, temperature and prices from the settings (`eval_model`,
`eval_temperature`, `eval_price_*`). Every successful request writes one `llm_usage` row
at those prices (rule 9), with a feature name starting with `eval-`.
"""

from collections.abc import Callable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from app.config import get_settings
from app.ingest.embed import default_openai_client
from app.llm.client import ChatClient, _as_retryable
from app.llm.usage import log_usage
from app.sources.http import call_with_retry

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


# --- embeddings for response relevancy --------------------------------------------------------


def eval_embedding_cost(tokens: int) -> Decimal:
    """USD for `tokens` input tokens at the configured evaluation embedding price."""
    price = Decimal(str(get_settings().eval_embedding_price_per_1m))
    return (Decimal(tokens) * price / 1_000_000).quantize(_MICRO_DOLLAR, ROUND_HALF_UP)


def record_eval_embedding_usage(feature: str, model: str, tokens_in: int) -> None:
    from app.db.session import session_scope

    with session_scope() as session:
        log_usage(
            session,
            feature=feature,
            model=model,
            tokens_in=tokens_in,
            tokens_out=0,
            cost_usd=eval_embedding_cost(tokens_in),
        )


class EvalEmbedder:
    """Embeddings with the evaluation's own model (`eval_embedding_model`), not the
    retrieval's: an ablation that changes the retrieval embeddings must not change the
    yardstick of response relevancy. One request per call (a question and the few
    questions written back from an answer), retried like every chat request, logged to
    llm_usage at `eval_embedding_price_per_1m`. Tests pass a fake `client`."""

    MAX_TEXTS = 32  # far above what response relevancy sends; keeps one call one request

    def __init__(
        self,
        client: Any = None,
        record_usage: Callable[[str, str, int], None] | None = None,
        *,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._client = client
        self._record = record_usage or record_eval_embedding_usage
        self.model = get_settings().eval_embedding_model
        self._retry_kwargs: dict[str, Any] = {} if sleep is None else {"sleep": sleep}

    @property
    def client(self) -> Any:
        if self._client is None:  # created lazily: no key is needed until the first call
            self._client = default_openai_client()
        return self._client

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        if not texts or len(texts) > self.MAX_TEXTS or any(not t.strip() for t in texts):
            raise ValueError(f"expected 1 to {self.MAX_TEXTS} non-empty texts, got {len(texts)}")
        response = call_with_retry(
            _as_retryable(
                lambda: self.client.embeddings.create(input=list(texts), model=self.model)
            ),
            **self._retry_kwargs,
        )
        # Logged right after the request succeeded: this is when the money is spent.
        self._record(feature, self.model, int(response.usage.prompt_tokens))
        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            raise ValueError(f"asked for {len(texts)} vectors, got {len(items)}")
        return [list(item.embedding) for item in items]

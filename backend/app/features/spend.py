"""What one request of the compare or timeline feature has spent (rule 9).

Every API request already writes its own `llm_usage` row the moment it succeeds (the
embedding client and the chat client do that). This module only adds the numbers up for
the response, so the page can show tokens and cost without a second query.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.llm.client import ChatUsage
from app.llm.usage import UsageRecorder, embedding_cost, record_embedding_usage


@dataclass
class Spend:
    """Tokens and cost of the requests made so far (embeddings and chat together)."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Decimal = Decimal("0")
    embedding_requests: int = 0
    chat_requests: int = 0

    def add_chat(self, usage: ChatUsage) -> None:
        self.tokens_in += usage.tokens_in
        self.tokens_out += usage.tokens_out
        self.cost_usd += usage.cost_usd
        self.chat_requests += 1


class EmbeddingMeter:
    """A UsageRecorder that writes the llm_usage row as usual and keeps the total."""

    def __init__(self, spend: Spend, inner: UsageRecorder = record_embedding_usage) -> None:
        self._spend = spend
        self._inner = inner

    def __call__(self, feature: str, model: str, tokens_in: int) -> None:
        self._inner(feature, model, tokens_in)
        self._spend.tokens_in += tokens_in
        self._spend.cost_usd += embedding_cost(tokens_in)
        self._spend.embedding_requests += 1

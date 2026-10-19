"""What compare and timeline talk to, gathered so that tests can replace every part."""

from collections.abc import Callable
from dataclasses import dataclass

import openai
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from app.features.errors import UpstreamError
from app.features.paper_facts import FactsStore, SqlFactsStore
from app.ingest.embed import (
    DenseEmbedder,
    DenseEncoder,
    EmbeddingError,
    OpenAIKeyMissing,
    SparseEncoder,
    get_sparse_embedder,
)
from app.ingest.index import get_qdrant
from app.llm.client import ChatClient
from app.llm.usage import UsageRecorder, record_embedding_usage
from app.observability.tracing import Span, Tracer, get_tracer
from app.rag.search import SearchHit, search
from app.sources.http import RetryableError

SearchFn = Callable[..., list[SearchHit]]


def _dense_embedder(record: UsageRecorder) -> DenseEncoder:
    return DenseEmbedder(record_usage=record)


@dataclass
class ExploreDeps:
    """Everything a comparison or a timeline talks to. `dense` builds the embedder from a
    usage recorder, so the request can add up what its embeddings cost."""

    chat: ChatClient
    facts: FactsStore
    tracer: Tracer
    dense: Callable[[UsageRecorder], DenseEncoder] = _dense_embedder
    sparse: Callable[[], SparseEncoder] = get_sparse_embedder
    qdrant: Callable[[], QdrantClient] = get_qdrant
    search_fn: SearchFn = search
    record_embedding: UsageRecorder = record_embedding_usage

    @classmethod
    def default(cls) -> "ExploreDeps":
        return cls(chat=ChatClient(), facts=SqlFactsStore(), tracer=get_tracer())


def upstream(err: Exception) -> Exception:
    """The error to raise for a failure of OpenAI or Qdrant (HTTP 502). Anything else (a
    bug, a missing key) is passed through unchanged."""
    if isinstance(
        err,
        RetryableError
        | openai.OpenAIError
        | EmbeddingError
        | UnexpectedResponse
        | ResponseHandlingException,
    ) and not isinstance(err, OpenAIKeyMissing):
        return UpstreamError(f"{type(err).__name__}: {err}")
    return err


def end_span(span: Span, *, error: BaseException | None = None, **metadata: object) -> None:
    span.end(metadata=metadata, error=None if error is None else f"{type(error).__name__}: {error}")

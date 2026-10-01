"""Turn chunk texts into the two vectors each Qdrant point holds.

- Dense (`text-dense`): OpenAI embeddings. Texts are sent in batches (one request per
  `embedding_batch_size` texts); transient failures are retried with exponential backoff
  by our own retry layer (the SDK's built-in retries are switched off so attempts are
  not multiplied); every successful request is logged to `llm_usage`.
- Sparse (`text-sparse`): BM25 term weights from FastEmbed, computed locally. The IDF
  part is applied by Qdrant at query time (the collection's IDF modifier).

BM25 never touches the network: FastEmbed's own download/cache check for Qdrant/bm25
always goes online (it requires a file, `mock.file`, that the Hub repo does not have)
and fails offline. Instead the only file BM25 reads, the English stopword list, is
committed in `app/ingest/data/` and handed to FastEmbed via `specific_model_path`.

Both embedders sit behind Protocols, so tests inject fakes and make no API calls.
"""

import tempfile
from collections.abc import Sequence
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Protocol

import openai
from qdrant_client import models as m

from app.config import get_settings
from app.llm.usage import UsageRecorder, record_embedding_usage
from app.sources.http import RetryableError, call_with_retry

STOPWORDS_FILE = Path(__file__).parent / "data" / "bm25_stopwords_english.txt"
BM25_LANGUAGE = "english"
OPENAI_TIMEOUT_S = 60.0


class DenseEncoder(Protocol):
    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]: ...


class SparseEncoder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[m.SparseVector]: ...


# --- dense: OpenAI --------------------------------------------------------------------------


class _EmbeddingItem(Protocol):
    @property
    def embedding(self) -> list[float]: ...
    @property
    def index(self) -> int: ...


class _Usage(Protocol):
    @property
    def prompt_tokens(self) -> int: ...


class _EmbeddingResponse(Protocol):
    @property
    def data(self) -> Sequence[_EmbeddingItem]: ...
    @property
    def usage(self) -> _Usage: ...


class _EmbeddingsAPI(Protocol):
    def create(self, *, input: list[str], model: str) -> _EmbeddingResponse: ...


class OpenAIClient(Protocol):
    """The one part of `openai.OpenAI` we use (tests pass a fake with the same shape)."""

    @property
    def embeddings(self) -> _EmbeddingsAPI: ...


class EmbeddingError(RuntimeError):
    """The API answered, but not with what we asked for (wrong count or size)."""


def default_openai_client() -> OpenAIClient:
    key = get_settings().openai_api_key
    if key is None:
        raise RuntimeError("OPENAI_API_KEY is not set (put it in .env); cannot embed")
    client: Any = openai.OpenAI(
        api_key=key.get_secret_value(), max_retries=0, timeout=OPENAI_TIMEOUT_S
    )
    return client  # type: ignore[no-any-return]


def _retry_after(err: openai.APIStatusError) -> float | None:
    value = err.response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


class DenseEmbedder:
    def __init__(
        self,
        client: OpenAIClient | None = None,
        record_usage: UsageRecorder | None = None,
        *,
        sleep: Any = None,
    ) -> None:
        settings = get_settings()
        self._client = client
        self._record_usage = record_usage or record_embedding_usage
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim
        self._batch_size = settings.embedding_batch_size
        self._retry_kwargs: dict[str, Any] = {} if sleep is None else {"sleep": sleep}

    @property
    def client(self) -> OpenAIClient:
        if self._client is None:  # created lazily: no key is needed until we embed
            self._client = default_openai_client()
        return self._client

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        """One vector per text, in order. `feature` labels the llm_usage rows
        ('embed-abstract' or 'embed-fulltext')."""
        if any(not t.strip() for t in texts):
            raise ValueError("cannot embed an empty text")  # the API rejects them too
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = call_with_retry(partial(self._request, batch), **self._retry_kwargs)
            items = sorted(response.data, key=lambda item: item.index)
            if len(items) != len(batch) or any(len(i.embedding) != self._dim for i in items):
                raise EmbeddingError(
                    f"asked for {len(batch)} vectors of size {self._dim}, got "
                    f"{len(items)} of sizes {sorted({len(i.embedding) for i in items})}"
                )
            # Logged right after the request succeeded: this is when the money is spent.
            self._record_usage(feature, self._model, response.usage.prompt_tokens)
            vectors += [list(i.embedding) for i in items]
        return vectors

    def _request(self, batch: list[str]) -> _EmbeddingResponse:
        """One API request. Transient failures become RetryableError for call_with_retry;
        anything else (bad key, bad request) is raised at once."""
        try:
            return self.client.embeddings.create(input=batch, model=self._model)
        except openai.RateLimitError as err:
            raise RetryableError(str(err), status=429, retry_after=_retry_after(err)) from err
        except openai.InternalServerError as err:
            raise RetryableError(str(err), status=err.status_code) from err
        except openai.APIConnectionError as err:  # includes timeouts
            raise RetryableError(f"network error: {err}") from err


# --- sparse: FastEmbed BM25 ---------------------------------------------------------------------


def load_stopwords(path: Path = STOPWORDS_FILE) -> list[str]:
    """The committed stopword list, without its '#' header lines."""
    if not path.is_file():
        raise FileNotFoundError(
            f"BM25 stopword file missing: {path}. It is committed in the repo; restore it "
            "with git (the app never downloads it)."
        )
    words = [w for w in path.read_text().splitlines() if w and not w.startswith("#")]
    if not words:
        raise ValueError(f"BM25 stopword file is empty: {path}")
    return words


class SparseEmbedder:
    """FastEmbed BM25, loaded from local files only (see the module docstring)."""

    def __init__(self, stopwords_file: Path = STOPWORDS_FILE) -> None:
        from fastembed import (
            SparseTextEmbedding,
        )  # imported here: only processes that index need it

        self.stopwords = load_stopwords(stopwords_file)
        # FastEmbed reads '<language>.txt' from the model directory, so write the list
        # (without our header) into a private directory and point FastEmbed at it.
        self._model_dir = tempfile.TemporaryDirectory(prefix="bm25-")
        model_path = Path(self._model_dir.name)
        (model_path / f"{BM25_LANGUAGE}.txt").write_text("\n".join(self.stopwords) + "\n")
        self._model = SparseTextEmbedding(
            get_settings().sparse_model,
            cache_dir=str(model_path),
            specific_model_path=str(model_path),  # skips FastEmbed's download step entirely
            language=BM25_LANGUAGE,
        )

    def embed(self, texts: Sequence[str]) -> list[m.SparseVector]:
        return [
            m.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.embed(list(texts))
        ]


@lru_cache
def get_sparse_embedder() -> SparseEmbedder:
    """One BM25 model per process."""
    return SparseEmbedder()


@lru_cache
def get_dense_embedder() -> DenseEmbedder:
    return DenseEmbedder()

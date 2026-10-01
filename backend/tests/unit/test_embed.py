"""embed.py: DenseEmbedder against a fake OpenAI client (no API calls), and the real
BM25 model loaded from the committed stopword file (no Hub access, network blocked)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest

from app.ingest import embed as embed_module
from app.ingest.embed import (
    STOPWORDS_FILE,
    DenseEmbedder,
    EmbeddingError,
    SparseEmbedder,
    load_stopwords,
)
from app.sources.http import RetryableError

DIM = 1536
UsageLog = list[tuple[str, str, int]]  # (feature, model, tokens_in) per request


# --- a fake OpenAI client with the same shape as openai.OpenAI().embeddings ------------------


@dataclass
class _Item:
    embedding: list[float]
    index: int


@dataclass
class _Usage:
    prompt_tokens: int


@dataclass
class _Response:
    data: list[_Item]
    usage: _Usage


@dataclass
class FakeEmbeddings:
    """Returns one vector per input (in reverse order, to test re-sorting by index).
    `fail_with` holds errors to raise on the first calls."""

    fail_with: list[Exception] = field(default_factory=list)
    dim: int = DIM
    calls: list[list[str]] = field(default_factory=list)

    def create(self, *, input: list[str], model: str) -> _Response:
        self.calls.append(list(input))
        if self.fail_with:
            raise self.fail_with.pop(0)
        items = [
            _Item([float(len(t)), 1.0] + [0.0] * (self.dim - 2), i) for i, t in enumerate(input)
        ]
        return _Response(data=list(reversed(items)), usage=_Usage(prompt_tokens=10 * len(input)))


@dataclass
class FakeClient:
    embeddings: FakeEmbeddings


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.example/v1/embeddings")


def _status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    response = httpx2.Response(status, request=_request(), headers={"retry-after": "0"})
    return cls("invented error", response=response, body=None)


def _embedder(fake: FakeEmbeddings, usage: UsageLog) -> DenseEmbedder:
    def record(feature: str, model: str, tokens: int) -> None:
        usage.append((feature, model, tokens))

    return DenseEmbedder(FakeClient(fake), record, sleep=lambda s: None)


# --- dense ---------------------------------------------------------------------------------------


def test_batches_by_configured_size_and_keeps_order() -> None:
    fake = FakeEmbeddings()
    usage: UsageLog = []
    texts = [f"text number {i}" + "x" * i for i in range(250)]
    vectors = _embedder(fake, usage).embed(texts, "embed-abstract")
    assert [len(c) for c in fake.calls] == [100, 100, 50]  # embedding_batch_size = 100
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]  # order restored
    assert all(len(v) == DIM for v in vectors)


def test_one_usage_record_per_request() -> None:
    fake = FakeEmbeddings()
    usage: UsageLog = []
    _embedder(fake, usage).embed([f"t{i}" for i in range(150)], "embed-fulltext")
    assert usage == [
        ("embed-fulltext", "text-embedding-3-small", 1000),
        ("embed-fulltext", "text-embedding-3-small", 500),
    ]


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.RateLimitError, 429),
        _status_error(openai.InternalServerError, 503),
        openai.APIConnectionError(request=_request()),
        openai.APITimeoutError(request=_request()),
    ],
)
def test_transient_errors_are_retried(error: Exception) -> None:
    fake = FakeEmbeddings(fail_with=[error, error])
    usage: UsageLog = []
    vectors = _embedder(fake, usage).embed(["a", "b"], "embed-abstract")
    assert len(fake.calls) == 3 and len(vectors) == 2
    assert len(usage) == 1  # only the successful request is billed and logged


def test_gives_up_after_the_last_attempt() -> None:
    error = _status_error(openai.RateLimitError, 429)
    fake = FakeEmbeddings(fail_with=[error] * 5)
    usage: UsageLog = []
    with pytest.raises(RetryableError):
        _embedder(fake, usage).embed(["a"], "embed-abstract")
    assert len(fake.calls) == 5 and usage == []


def test_non_transient_errors_are_not_retried() -> None:
    error = _status_error(openai.AuthenticationError, 401)
    fake = FakeEmbeddings(fail_with=[error])
    usage: UsageLog = []
    with pytest.raises(openai.AuthenticationError):
        _embedder(fake, usage).embed(["a"], "embed-abstract")
    assert len(fake.calls) == 1 and usage == []


def test_wrong_vector_size_is_an_error() -> None:
    fake = FakeEmbeddings(dim=512)
    usage: UsageLog = []
    with pytest.raises(EmbeddingError, match="size 1536"):
        _embedder(fake, usage).embed(["a"], "embed-abstract")


def test_empty_text_is_rejected_before_any_request() -> None:
    fake = FakeEmbeddings()
    usage: UsageLog = []
    with pytest.raises(ValueError, match="empty"):
        _embedder(fake, usage).embed(["fine", "  "], "embed-abstract")
    assert fake.calls == []


def test_missing_key_fails_clearly_and_only_when_embedding() -> None:
    embedder = DenseEmbedder(record_usage=lambda f, m, t: None)  # tests blank OPENAI_API_KEY
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        embedder.embed(["a"], "embed-abstract")


# --- sparse (real BM25, local files only) ------------------------------------------------------


@pytest.fixture
def no_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any Hugging Face Hub call fails the test (on top of the blocked network)."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the Hugging Face Hub must not be called")

    import fastembed.common.model_management as mm

    for name in ("snapshot_download", "model_info", "list_repo_tree"):
        monkeypatch.setattr(mm, name, forbidden)


def test_committed_stopword_file_matches_the_source() -> None:
    import hashlib

    lines = STOPWORDS_FILE.read_text().splitlines()
    header = [line for line in lines if line.startswith("#")]
    assert any("Qdrant/bm25" in h for h in header) and any("Apache-2.0" in h for h in header)
    body = "\n".join(lines[len(header) :]) + "\n"
    # sha256 of the original english.txt, as recorded in the header
    expected = "019f104ba2ed07436d05f9cdd3383034ad66014edc27fc651f837e1a038b6451"
    assert hashlib.sha256(body.encode()).hexdigest() == expected
    assert len(load_stopwords()) == 179 and "the" in load_stopwords()


@pytest.mark.usefixtures("no_hub")
def test_bm25_loads_offline_and_drops_stopwords() -> None:
    sparse = SparseEmbedder()
    vectors = sparse.embed(["the attention of the transformer", "the the the", "attention"])
    assert len(vectors[0].indices) == 2  # 'attention', 'transformer'; stopwords dropped
    assert vectors[1].indices == []  # only stopwords
    assert set(vectors[2].indices) <= set(vectors[0].indices)  # same term, same index
    assert vectors == sparse.embed(["the attention of the transformer", "the the the", "attention"])
    assert all(v > 0 for v in vectors[0].values)


def test_missing_stopword_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="BM25 stopword file missing"):
        SparseEmbedder(stopwords_file=tmp_path / "nope.txt")


def test_header_only_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "words.txt"
    path.write_text("# only a header\n")
    with pytest.raises(ValueError, match="empty"):
        load_stopwords(path)


def test_sparse_embedder_is_loaded_once_per_process() -> None:
    assert embed_module.get_sparse_embedder() is embed_module.get_sparse_embedder()

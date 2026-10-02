"""`python -m eval.run` end to end, against a throwaway Postgres database and a
`chunks_test_<random>` Qdrant collection.

The corpus, the golden questions and every vector are invented. OpenAI is a fake behind
the real DenseEmbedder (so llm_usage logging is real), and the reranker is a fake
cross-encoder. The real collection is checked to be untouched after every test.
"""

import json
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import EvalItem, EvalResult, EvalRun, LLMUsage, Paper
from app.ingest import index as index_module
from app.ingest.chunk import Chunk, abstract_chunk, chunk_id, embedding_text
from app.ingest.embed import DenseEmbedder, get_sparse_embedder
from app.ingest.index import ensure_collection, save_chunks, upsert_paper_chunks
from app.rag.rerank import RerankerUnavailable
from eval import run as run_module
from eval.run import EMBED_FEATURE, main

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

DIM = 4
P1, P2, P3 = "2601.00001", "2601.00002", "2601.00003"
ABSTRACTS = {
    P1: "We study low-rank adaptation for fine-tuning language models.",
    P2: "A mixture of experts sends each token to two of sixteen experts.",
    P3: "Retrieval augmented generation adds retrieved passages to the prompt.",
}
SECTION = "LoRA freezes the pretrained weights and trains two small matrices per layer."
# One dense direction per chunk, so the dense ranking is known exactly.
CHUNK_VECTORS = {
    ABSTRACTS[P1]: [1.0, 0.0, 0.0, 0.0],
    SECTION: [0.0, 1.0, 0.0, 0.0],
    ABSTRACTS[P2]: [0.0, 0.0, 1.0, 0.0],
    ABSTRACTS[P3]: [0.0, 0.0, 0.0, 1.0],
}
Q1, Q2, Q3 = (
    "How does LoRA train a model?",
    "Which experts does each token go to?",
    "Which papers fold proteins?",
)
QUESTION_VECTORS = {
    Q1: [0.0, 1.0, 0.0, 0.0],
    Q2: [0.0, 0.0, 1.0, 0.0],
    Q3: [0.5, 0.5, 0.5, 0.5],
}
SECTION_ID = chunk_id(P1, 1, 1)
P2_ABSTRACT_ID = chunk_id(P2, 1, 0)


class IndexingDense:
    """Dense vectors for indexing: picked by which invented chunk text is embedded."""

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        return [next(v for k, v in CHUNK_VECTORS.items() if k in t) for t in texts]


# --- a fake OpenAI client, the same shape as openai.OpenAI().embeddings -----------------------


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
    calls: list[list[str]] = field(default_factory=list)

    def create(self, *, input: list[str], model: str) -> _Response:
        self.calls.append(list(input))
        items = [_Item(QUESTION_VECTORS[t], i) for i, t in enumerate(input)]
        return _Response(data=items, usage=_Usage(prompt_tokens=8 * len(input)))


@dataclass
class FakeClient:
    embeddings: FakeEmbeddings


class ConstantCrossEncoder:
    """Gives every chunk the same score, so the (stable) order stays the hybrid order."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
        self.calls += 1
        return [0.0 for _ in documents]


# --- fixtures ---------------------------------------------------------------------------------


@pytest.fixture
def corpus(
    migrated_session: Session,
    qdrant: QdrantClient,
    scratch_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Session]:
    """Three invented papers (4 chunks) in the scratch DB and collection, which are also
    set as the configured ones (the runner uses the configured stores)."""
    ensure_collection(qdrant, scratch_collection, DIM)
    monkeypatch.setenv("QDRANT_COLLECTION", scratch_collection)
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    get_settings.cache_clear()
    index_module.get_qdrant.cache_clear()

    for arxiv_id, abstract in ABSTRACTS.items():
        title = f"Invented Paper {arxiv_id}"
        migrated_session.add(
            Paper(
                arxiv_id=arxiv_id, version=1, title=title, abstract=abstract,
                authors=[{"name": "Ada Example"}], categories=["cs.CL"], primary_category="cs.CL",
                published_at=datetime(2026, 1, 2, tzinfo=UTC),
            )
        )  # fmt: skip
        migrated_session.flush()
        chunks = [abstract_chunk(arxiv_id, 1, title, abstract)]
        if arxiv_id == P1:
            chunks.append(
                Chunk(
                    id=SECTION_ID, ord=1, chunk_type="section", section_path="2 Method",
                    text=SECTION, token_count=16,
                    embedding_text=embedding_text(title, "2 Method", SECTION),
                )
            )  # fmt: skip
        save_chunks(migrated_session, arxiv_id, chunks)
    migrated_session.commit()
    for arxiv_id in ABSTRACTS:
        upsert_paper_chunks(
            migrated_session, arxiv_id, dense=IndexingDense(), sparse=get_sparse_embedder(),
            client=qdrant, collection=scratch_collection,
        )  # fmt: skip
    migrated_session.commit()
    yield migrated_session
    index_module.close_qdrant()
    get_settings.cache_clear()


@pytest.fixture
def golden_file(tmp_path: Path) -> Path:
    path = tmp_path / "golden.jsonl"
    items = [
        {"id": "q01", "question": Q1, "reference_chunk_ids": [str(SECTION_ID)], "type": "fact"},
        {"id": "q02", "question": Q2, "reference_chunk_ids": [str(P2_ABSTRACT_ID)]},
        {"id": "q03", "question": Q3, "type": "unanswerable"},
    ]
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    return path


@pytest.fixture
def openai_fake(monkeypatch: pytest.MonkeyPatch) -> FakeEmbeddings:
    """The real DenseEmbedder (it logs llm_usage) over a fake OpenAI client."""
    fake = FakeEmbeddings()
    monkeypatch.setattr(
        run_module, "get_dense_embedder", lambda: DenseEmbedder(FakeClient(fake), sleep=None)
    )
    return fake


@pytest.fixture
def cross_encoder(monkeypatch: pytest.MonkeyPatch) -> ConstantCrossEncoder:
    encoder = ConstantCrossEncoder()
    monkeypatch.setattr(run_module, "get_cross_encoder", lambda: encoder)
    return encoder


@pytest.fixture(autouse=True)
def vector_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "cache" / "query_embeddings.json"
    monkeypatch.setattr(run_module, "CACHE_PATH", path)
    return path


def _count(session: Session, model: Any) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def _stored(session: Session) -> tuple[int, int, int, int]:
    """(eval_items, eval_runs, eval_results, llm_usage) row counts."""
    return tuple(_count(session, m) for m in (EvalItem, EvalRun, EvalResult, LLMUsage))  # type: ignore[return-value]


# --- tests ------------------------------------------------------------------------------------


def test_uncached_questions_stop_the_run_before_any_call_or_write(
    corpus: Session,
    golden_file: Path,
    openai_fake: FakeEmbeddings,
    cross_encoder: ConstantCrossEncoder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--golden", str(golden_file)]) == 3
    err = capsys.readouterr().err
    assert "3 of 3 questions have no cached embedding" in err
    assert "1 request(s)" in err and "make eval-retrieval EMBED=1" in err
    assert openai_fake.calls == []
    assert _stored(corpus) == (0, 0, 0, 0)


def test_embed_once_then_every_run_is_free(
    corpus: Session,
    golden_file: Path,
    openai_fake: FakeEmbeddings,
    cross_encoder: ConstantCrossEncoder,
    vector_cache: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--golden", str(golden_file), "--embed"]) == 0
    # One batched request for the three questions, logged once to llm_usage.
    assert [sorted(c) for c in openai_fake.calls] == [sorted([Q1, Q2, Q3])]
    assert corpus.scalars(select(LLMUsage.feature)).all() == [EMBED_FEATURE]
    assert vector_cache.is_file()
    assert _stored(corpus) == (3, 4, 12, 1)
    assert cross_encoder.calls == 3  # hybrid-rerank only, once per question

    runs = {r.name: r for r in corpus.scalars(select(EvalRun))}
    assert list(runs) == ["dense", "bm25", "hybrid", "hybrid-rerank"]
    assert [(r.config["mode"], r.config["rerank"]) for r in runs.values()] == [
        ("dense", False), ("bm25", False), ("hybrid", False), ("hybrid", True),
    ]  # fmt: skip
    for run in runs.values():
        # Shared settings: the per-paper cap applies to all four (deviation 5, ADR 0005).
        assert (run.config["max_chunks_per_paper"], run.config["top_k"]) == (3, 10)
        assert (run.metrics["n_scored"], run.metrics["n_unscored"]) == (2, 1)
        assert run.config["golden_questions"] == 3 and run.git_sha
    assert runs["hybrid-rerank"].config["reranker_model"] == get_settings().reranker_model
    assert runs["dense"].config["reranker_model"] is None
    # The question vectors point exactly at their reference chunks.
    assert {k: runs["dense"].metrics[k] for k in ("recall@8", "mrr", "ndcg@10")} == {
        "recall@8": 1.0, "mrr": 1.0, "ndcg@10": 1.0,
    }  # fmt: skip
    # Equal reranker scores keep the hybrid order, so the metrics match.
    for key in ("recall@8", "mrr", "ndcg@10"):
        assert runs["hybrid-rerank"].metrics[key] == runs["hybrid"].metrics[key]

    results = corpus.scalars(
        select(EvalResult).where(EvalResult.run_id == runs["dense"].id).order_by(EvalResult.id)
    ).all()
    assert [r.metrics.get("golden_id") for r in results] == ["q01", "q02", "q03"]
    assert results[0].retrieved_chunk_ids[0] == SECTION_ID
    assert results[2].metrics == {"golden_id": "q03"}  # unanswerable: stored, not scored
    table = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| ")]
    assert len(table) == 6  # header, rule, 4 runs

    # A second run: no OpenAI request, no duplicate questions.
    assert main(["--golden", str(golden_file)]) == 0
    assert len(openai_fake.calls) == 1
    assert _stored(corpus) == (3, 8, 24, 1)


def test_bm25_alone_needs_no_embeddings(
    corpus: Session, golden_file: Path, openai_fake: FakeEmbeddings
) -> None:
    assert main(["--golden", str(golden_file), "--configs", "bm25"]) == 0
    assert openai_fake.calls == []
    assert _stored(corpus) == (3, 1, 3, 0)


def test_stale_reference_ids_stop_the_run(
    corpus: Session,
    tmp_path: Path,
    openai_fake: FakeEmbeddings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stale = uuid.uuid5(uuid.NAMESPACE_URL, "an older paper version")
    path = tmp_path / "stale.jsonl"
    path.write_text(json.dumps({"id": "q01", "question": Q1, "reference_chunk_ids": [str(stale)]}))
    assert main(["--golden", str(path), "--embed"]) == 2
    assert f"q01: {stale} is not in the chunks table" in capsys.readouterr().err
    assert openai_fake.calls == [] and _stored(corpus) == (0, 0, 0, 0)


def test_missing_reranker_stops_the_run_before_any_spend(
    corpus: Session,
    golden_file: Path,
    openai_fake: FakeEmbeddings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unavailable() -> Any:
        raise RerankerUnavailable("not in the cache; run make fetch-reranker")

    monkeypatch.setattr(run_module, "get_cross_encoder", unavailable)
    assert main(["--golden", str(golden_file), "--embed"]) == 4
    assert "make fetch-reranker" in capsys.readouterr().err
    assert openai_fake.calls == [] and _stored(corpus) == (0, 0, 0, 0)

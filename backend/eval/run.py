"""Retrieval evaluation (`make eval-retrieval`): the golden questions through four
configurations, with the results stored in eval_runs and eval_results.

    python -m eval.run [--configs dense,bm25,hybrid,hybrid-rerank] [--golden PATH] [--embed]

What a run does, in order (it stops at the first problem, before storing anything):
1. load and check golden.jsonl; every reference chunk id must exist in Postgres and in
   Qdrant (a new paper version changes chunk ids, see eval/golden.py);
2. load the reranker if a configuration uses it (from the local cache only);
3. find each question's dense vector in the local cache. Missing vectors stop the run,
   and the cost of embedding them is printed; `--embed` embeds just those, in batched
   OpenAI requests logged to llm_usage. Without `--embed` the run makes NO LLM call
   and costs nothing (the bm25 configuration needs no vectors at all);
4. store the questions in eval_items, matched by their exact text, so re-runs do not
   duplicate them;
5. for each configuration: retrieve the top 10 for every question, score them, and
   store one eval_runs row (config, git commit, averaged metrics, latency) plus one
   eval_results row per question;
6. print a table of the results.

All configurations share the same settings (candidate counts, RRF k, the per-paper cap)
and differ only in the search mode and the rerank switch, so each row isolates one
change. Questions are run without filters: nothing turns a question into filters yet.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk as ChunkRow
from app.db.models import EvalItem, EvalResult, EvalRun
from app.db.session import session_scope
from app.ingest.embed import DenseEncoder, SparseEncoder, get_dense_embedder, get_sparse_embedder
from app.ingest.index import close_qdrant, get_qdrant
from app.ingest.tokens import count_tokens
from app.llm.usage import embedding_cost
from app.rag.rerank import CrossEncoder, RerankerUnavailable, get_cross_encoder
from app.rag.retrieve import RetrievalConfig, retrieve
from eval.golden import GOLDEN_PATH, GoldenError, GoldenItem, load_golden
from eval.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank

BACKEND_DIR = Path(__file__).resolve().parents[1]
CACHE_PATH = Path(__file__).parent / ".cache" / "query_embeddings.json"  # gitignored
EMBED_FEATURE = "eval-embed-query"  # llm_usage.feature of the golden questions' embeddings
TOP_K = 10  # chunks retrieved per question
RECALL_K, NDCG_K = 8, 10
METRICS = ("recall@8", "mrr", "ndcg@10")

# Each configuration changes only these fields of the configured RetrievalConfig.
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "dense": {"mode": "dense", "rerank": False},
    "bm25": {"mode": "bm25", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid-rerank": {"mode": "hybrid", "rerank": True},
}


def build_configs(names: Sequence[str]) -> dict[str, RetrievalConfig]:
    """The named configurations: the configured settings (candidates, RRF k, the cap)
    with only the mode and the rerank switch replaced. Rerank is set explicitly, so the
    evaluation does not depend on RERANK_ENABLED (the GET /search default)."""
    base = RetrievalConfig.from_settings()
    return {name: replace(base, **CONFIGURATIONS[name]) for name in names}


# --- question embeddings: cached, so repeated runs cost nothing --------------------------------


class QueryVectorCache:
    """Question embeddings on disk, as {sha256(model|dim|question): vector}.

    The key includes the embedding model and its dimension, so a vector made by one model
    is never reused for another.
    """

    def __init__(self, path: Path, model: str, dim: int) -> None:
        self.path, self.model, self.dim = path, model, dim
        self._vectors: dict[str, list[float]] = (
            json.loads(path.read_text()) if path.is_file() else {}
        )

    def _key(self, question: str) -> str:
        return hashlib.sha256(f"{self.model}|{self.dim}|{question}".encode()).hexdigest()

    def get(self, question: str) -> list[float] | None:
        return self._vectors.get(self._key(question))

    def missing(self, questions: Sequence[str]) -> list[str]:
        return [q for q in questions if self._key(q) not in self._vectors]

    def put(self, question: str, vector: list[float]) -> None:
        self._vectors[self._key(question)] = vector

    def save(self) -> None:
        """Write to a temporary file, then rename: a crash never leaves half a file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._vectors))
        tmp.replace(self.path)


class CachedQueryEncoder:
    """A DenseEncoder that only answers from the cache: the evaluation itself can never
    make an OpenAI request."""

    def __init__(self, cache: QueryVectorCache) -> None:
        self.cache = cache

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = self.cache.get(text)
            if vector is None:
                raise RuntimeError(f"no cached embedding for {text!r} (run with --embed)")
            vectors.append(vector)
        return vectors


def embedding_estimate(questions: Sequence[str]) -> tuple[int, int, str]:
    """(tokens, requests, cost in USD) of embedding `questions`; counted locally."""
    tokens = sum(count_tokens(q) for q in questions)
    requests = math.ceil(len(questions) / get_settings().embedding_batch_size)
    return tokens, requests, f"{embedding_cost(tokens)}"


def embed_missing(cache: QueryVectorCache, missing: Sequence[str], encoder: DenseEncoder) -> None:
    """Embed the uncached questions (PAID: batched OpenAI requests, each logged to
    llm_usage) and save the cache at once, before anything else can fail."""
    for question, vector in zip(missing, encoder.embed(missing, EMBED_FEATURE), strict=True):
        cache.put(question, vector)
    cache.save()


# --- checks and storage ------------------------------------------------------------------------


def check_references(
    session: Session, client: QdrantClient, collection: str, golden: Sequence[GoldenItem]
) -> None:
    """Every reference chunk id must be a chunk row AND a Qdrant point; else retrieval
    could never find it and the scores would be wrong. Raises GoldenError listing them."""
    wanted = {cid for g in golden for cid in g.reference_chunk_ids}
    if not wanted:
        return
    in_postgres = set(session.scalars(select(ChunkRow.id).where(ChunkRow.id.in_(wanted))))
    points = client.retrieve(
        collection, ids=[str(i) for i in wanted], with_payload=False, with_vectors=False
    )
    in_qdrant = {uuid.UUID(str(p.id)) for p in points}
    problems: list[str] = []
    for g in golden:
        for cid in g.reference_chunk_ids:
            if cid not in in_postgres:
                problems.append(f"{g.id}: {cid} is not in the chunks table")
            elif cid not in in_qdrant:
                problems.append(f"{g.id}: {cid} has no Qdrant point")
    if problems:
        raise GoldenError(
            "reference chunk ids not found (a new paper version or re-chunking changes "
            "chunk ids; update golden.jsonl):\n  " + "\n  ".join(problems)
        )


def sync_items(session: Session, golden: Sequence[GoldenItem]) -> dict[str, int]:
    """Store the questions in eval_items; returns golden id -> eval_items.id.

    A question already stored with the same exact text is reused (its reference answer
    and chunk ids are updated), so running again adds no duplicates. A reworded question
    becomes a new item; the old one stays, because earlier results point to it.
    """
    stored = session.scalars(
        select(EvalItem)
        .where(EvalItem.question.in_([g.question for g in golden]))
        .order_by(EvalItem.id)
    ).all()
    by_question: dict[str, EvalItem] = {}
    for existing in stored:
        by_question.setdefault(existing.question, existing)  # the oldest row wins
    rows: dict[str, EvalItem] = {}
    for g in golden:
        row = by_question.get(g.question)
        if row is None:
            row = EvalItem(question=g.question)
            session.add(row)
        row.reference_answer = g.reference_answer
        row.reference_chunk_ids = list(g.reference_chunk_ids)
        rows[g.id] = row
    session.flush()  # assigns ids to the new rows
    return {golden_id: row.id for golden_id, row in rows.items()}


def score(retrieved: Sequence[uuid.UUID], item: GoldenItem) -> dict[str, float]:
    """One question's metrics. 'mrr' holds the question's reciprocal rank; its mean over
    the questions is the MRR."""
    refs = item.reference_chunk_ids
    return {
        "recall@8": recall_at_k(retrieved, refs, RECALL_K),
        "mrr": reciprocal_rank(retrieved, refs),
        "ndcg@10": ndcg_at_k(retrieved, refs, NDCG_K),
    }


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100): the smallest value with at least p% of
    the values at or below it."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=BACKEND_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_commit() -> str | None:
    """The current commit, with '-dirty' if tracked files have uncommitted changes
    (None outside a git checkout)."""
    try:
        sha = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


def run_configuration(
    session: Session,
    name: str,
    config: RetrievalConfig,
    golden: Sequence[GoldenItem],
    item_ids: dict[str, int],
    *,
    dense: DenseEncoder,
    sparse: SparseEncoder,
    cross_encoder: CrossEncoder | None,
    client: QdrantClient,
    collection: str,
    run_info: dict[str, Any],
) -> EvalRun:
    """Retrieve and score every question with one configuration; store the run.

    Latency is measured per question around retrieval. It excludes embedding the
    question, because the vectors come from the cache.
    """
    per_item: list[tuple[GoldenItem, list[uuid.UUID], dict[str, float]]] = []
    latencies_ms: list[float] = []
    for item in golden:
        start = time.perf_counter()
        hits = retrieve(
            item.question, None, limit=TOP_K, config=config, dense=dense, sparse=sparse,
            cross_encoder=cross_encoder, client=client, collection=collection,
        )  # fmt: skip
        latencies_ms.append((time.perf_counter() - start) * 1000)
        retrieved = [h.chunk_id for h in hits]
        per_item.append((item, retrieved, score(retrieved, item) if item.scored else {}))

    scored = [m for _, _, m in per_item if m]
    metrics: dict[str, Any] = {key: mean([m[key] for m in scored]) for key in METRICS}
    metrics |= {
        "n_scored": len(scored),
        "n_unscored": len(per_item) - len(scored),
        "latency_ms_p50": round(percentile(latencies_ms, 50), 1),
        "latency_ms_p95": round(percentile(latencies_ms, 95), 1),
    }
    settings = get_settings()
    run = EvalRun(
        name=name,
        git_sha=run_info.get("git_sha"),
        metrics=metrics,
        config={
            "name": name,
            **asdict(config),
            "top_k": TOP_K,
            "filters": None,
            "collection": collection,
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "reranker_model": settings.reranker_model if config.rerank else None,
            "golden_sha256": run_info.get("golden_sha256"),
            "golden_questions": len(golden),
        },
    )
    session.add(run)
    session.flush()
    for item, retrieved, item_metrics in per_item:
        session.add(
            EvalResult(
                run_id=run.id,
                item_id=item_ids[item.id],
                retrieved_chunk_ids=retrieved,
                metrics={**item_metrics, "golden_id": item.id},
            )
        )
    session.flush()
    return run


def results_table(rows: Sequence[tuple[int, str, dict[str, Any]]]) -> str:
    """A markdown table of the runs (for the report and the status notes)."""

    def fmt(value: Any) -> str:
        if value is None:
            return "n/a"  # e.g. no question had reference chunks
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = [
        "| run | config | Recall@8 | MRR | nDCG@10 | scored | p50 ms | p95 ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run_id, name, m in rows:
        cells = [m.get(k) for k in (*METRICS, "n_scored", "latency_ms_p50", "latency_ms_p95")]
        lines.append(f"| {run_id} | {name} | " + " | ".join(fmt(c) for c in cells) + " |")
    return "\n".join(lines)


# --- command line ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.run", description="Retrieval metrics on the golden set (stored in eval_runs)."
    )
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGURATIONS),
        help=f"comma-separated, from: {', '.join(CONFIGURATIONS)}",
    )
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument(
        "--embed",
        action="store_true",
        help="embed questions missing from the cache (PAID OpenAI requests; ask first)",
    )
    args = parser.parse_args(argv)
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGURATIONS]
    if unknown or not names:
        parser.error(f"unknown configurations {unknown}; choose from {list(CONFIGURATIONS)}")

    settings = get_settings()
    configs = build_configs(names)
    collection = settings.qdrant_collection
    client = get_qdrant()
    try:
        golden = load_golden(args.golden)
        with session_scope() as session:
            check_references(session, client, collection, golden)
        # Loaded before any embedding request, so a missing model never wastes money.
        cross_encoder = get_cross_encoder() if any(c.rerank for c in configs.values()) else None

        cache = QueryVectorCache(CACHE_PATH, settings.embedding_model, settings.embedding_dim)
        if any(c.mode != "bm25" for c in configs.values()):
            missing = cache.missing([g.question for g in golden])
            if missing and not args.embed:
                tokens, requests, cost = embedding_estimate(missing)
                print(
                    f"{len(missing)} of {len(golden)} questions have no cached embedding, "
                    "so this run would need OpenAI. Nothing was run or stored.\n"
                    f"Embedding them: {requests} request(s), {tokens} tokens, about ${cost} "
                    f"({settings.embedding_model}). After an ok: make eval-retrieval EMBED=1",
                    file=sys.stderr,
                )
                return 3
            if missing:
                embed_missing(cache, missing, get_dense_embedder())
                print(f"embedded {len(missing)} question(s); cached in {CACHE_PATH.name}")

        dense, sparse = CachedQueryEncoder(cache), get_sparse_embedder()
        with session_scope() as session:
            item_ids = sync_items(session, golden)

        run_info = {
            "git_sha": git_commit(),
            "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        }
        rows: list[tuple[int, str, dict[str, Any]]] = []
        for name, config in configs.items():
            with session_scope() as session:  # one transaction per configuration
                run = run_configuration(
                    session, name, config, golden, item_ids, dense=dense, sparse=sparse,
                    cross_encoder=cross_encoder, client=client, collection=collection,
                    run_info=run_info,
                )  # fmt: skip
                rows.append((run.id, name, dict(run.metrics)))
        print(results_table(rows))
        return 0
    except GoldenError as err:
        print(f"golden set: {err}", file=sys.stderr)
        return 2
    except RerankerUnavailable as err:
        print(f"reranker: {err}", file=sys.stderr)
        return 4
    finally:
        close_qdrant()


if __name__ == "__main__":
    raise SystemExit(main())

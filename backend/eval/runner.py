"""The ablation runner's core: one variant, one part, stored as one eval_runs row.

A variant (`eval/variants.py`) runs in up to two parts, each its own eval_runs row named
`<suite kind>:<variant>:<part>` ("ablation:a3-hybrid:answers", "smoke:configured:retrieval"):

- retrieval (free with cached question vectors): `retrieve()` on each question, as
  `make eval-retrieval` does, top 10. Recall@8, MRR and nDCG@10 against the reference
  chunk ids when the index has the configured chunking (ids mean the same chunks), and
  reference coverage@8 (share of the reference passages' text in the top 8) always, so
  chunk sizes can be compared. Latency per question and the rerank time apart.
- answers (PAID): every question through the chat pipeline exactly as POST /chat runs it,
  scored by the judge (`eval/answers.py`), plus the same id and text metrics on the
  sources the generator was given (`source_recall`, `source_coverage`).

How a configuration is applied, and checked (the risk this module exists for):
`run_chat_turn`, `search`, `DenseEmbedder` and `get_cross_encoder` read the process-wide
settings, so a part runs inside `eval.scope.applied_settings(...)`: every setting the
variant changes is applied for the whole process and restored afterwards, in every case.
What the run then actually did is observed where it happens (the collection and mode of
every search, the embedding model of every query encoder, the reranker that was loaded)
and compared with what the settings asked for; a mismatch stops the run as "stopped:
configuration mismatch" instead of storing numbers for the wrong configuration. The
configuration stored on the run is read back inside the scope, so it is what ran.

Storage: the run row is written before the first question (status "running"), each
question's eval_results row as soon as it is done, the metrics and the status at the end
("complete" or "stopped: <reason>"), with the run's real spend read back from llm_usage.
Per question the retrieved chunks are stored with their texts (the per-question view
shows them, and an ablation index's chunk ids mean other texts than the chunks table).

Memory: each run records the process's peak resident memory and which variants the
process ran before it; a peak is only the variant's own if it ran alone (the reranker
comparison runs each variant in its own process for that reason).
"""

import resource
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from qdrant_client import QdrantClient
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.models import Chunk as ChunkRow
from app.db.models import EvalResult, EvalRun
from app.ingest.embed import (
    DenseEmbedder,
    DenseEncoder,
    OpenAIClient,
    SparseEncoder,
    default_openai_client,
    get_sparse_embedder,
)
from app.ingest.index import get_qdrant
from app.ingest.tokens import count_tokens
from app.llm.client import ChatClient
from app.llm.usage import UsageRecorder, record_embedding_usage
from app.observability.tracing import NoopTracer
from app.rag.pipeline import ChatDeps, SearchFn, describe
from app.rag.rerank import (
    CrossEncoder,
    RerankerKey,
    get_cross_encoder,
    loaded_reranker,
    reranker_key,
)
from app.rag.retrieve import RetrievalConfig, retrieve
from app.rag.search import SearchHit, search
from eval.answers import (
    ANSWER_METRICS,
    NullChatStore,
    Row,
    aggregate,
    collect_sample,
    hit_record,
    result_metrics,
    run_config,
    score_sample,
)
from eval.budget import Budget, BudgetExhausted
from eval.golden import GENERATED_PATH, GOLDEN_PATH, GoldenItem, SetQuestion
from eval.judge import Judge, default_judge
from eval.llm import EvalEmbedder
from eval.metrics import mean, recall_at_k, reference_coverage
from eval.run import (
    CACHE_PATH,
    METRICS,
    TOP_K,
    CachedQueryEncoder,
    QueryVectorCache,
    check_references,
    embed_missing,
    embedding_estimate,
    git_commit,
    percentile,
    score,
)
from eval.scope import applied_settings
from eval.variants import IndexSpec, Part, Resolved

COVERAGE_K = 8  # the chat's number of sources
RETRIEVAL_METRICS = (*METRICS, "coverage@8")
SOURCE_METRICS = ("source_recall", "source_coverage", "source_tokens")
SPLITS = ("all", "heldout", "generated")


class ConfigMismatch(RuntimeError):
    """What a run did differs from what its settings asked for (a configuration leak)."""


class MissingQueryVectors(RuntimeError):
    """Questions without a cached embedding; the message says what embedding them costs."""


class IndexMissing(RuntimeError):
    """The variant's collection is not built (or empty)."""


# --- the suite a run belongs to ------------------------------------------------------------------


@dataclass(frozen=True)
class Suite:
    """Runs started together: one ablation step, a smoke or a full evaluation."""

    kind: str  # ablation | smoke | full
    step: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


_process_runs: list[str] = []  # "<variant>:<part>" of every run this process made, in order


def peak_rss_mib() -> float:
    """Peak resident memory of this process so far (ru_maxrss: bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


# --- where runs are stored -----------------------------------------------------------------------


class RunStore(Protocol):
    def check(self, questions: Sequence[SetQuestion], collection: str) -> None: ...
    def reference_texts(self, ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, str]: ...
    def sync_items(self, questions: Sequence[SetQuestion]) -> dict[str, int]: ...
    def start(self, name: str, config: dict[str, Any], git_sha: str | None) -> int: ...
    def add(
        self, run_id: int, item_id: int, retrieved: Sequence[uuid.UUID], answer: str | None,
        metrics: dict[str, Any],
    ) -> None: ...  # fmt: skip
    def finish(self, run_id: int, metrics: dict[str, Any], status: str) -> dict[str, Any]: ...


class SqlRunStore:
    """eval_runs / eval_results in Postgres, one short transaction per call, so a stopped
    run keeps every question stored before it stopped."""

    def __init__(self, qdrant: Callable[[], QdrantClient] = get_qdrant) -> None:
        self._qdrant = qdrant

    def check(self, questions: Sequence[SetQuestion], collection: str) -> None:
        from app.db.session import session_scope

        with session_scope() as session:
            check_references(session, self._qdrant(), collection, [q.item for q in questions])

    def reference_texts(self, ids: Iterable[uuid.UUID]) -> dict[uuid.UUID, str]:
        from app.db.session import session_scope

        wanted = list(set(ids))
        if not wanted:
            return {}
        with session_scope() as session:
            rows = session.execute(
                select(ChunkRow.id, ChunkRow.text).where(ChunkRow.id.in_(wanted))
            ).all()
        return {i: t for i, t in rows}

    def sync_items(self, questions: Sequence[SetQuestion]) -> dict[str, int]:
        from app.db.session import session_scope
        from eval.run import sync_items

        with session_scope() as session:
            return sync_items(session, [q.item for q in questions])

    def start(self, name: str, config: dict[str, Any], git_sha: str | None) -> int:
        from app.db.session import session_scope
        from eval.answers import last_usage_id

        with session_scope() as session:
            run = EvalRun(
                name=name, git_sha=git_sha, config=config,
                metrics={"status": "running", "llm_usage_after_id": last_usage_id(session)},
            )  # fmt: skip
            session.add(run)
            session.flush()
            return run.id

    def add(
        self, run_id: int, item_id: int, retrieved: Sequence[uuid.UUID], answer: str | None,
        metrics: dict[str, Any],
    ) -> None:  # fmt: skip
        from app.db.session import session_scope

        with session_scope() as session:
            session.add(EvalResult(run_id=run_id, item_id=item_id,
                                   retrieved_chunk_ids=list(retrieved), answer=answer,
                                   metrics=metrics))  # fmt: skip

    def finish(self, run_id: int, metrics: dict[str, Any], status: str) -> dict[str, Any]:
        from app.db.session import session_scope
        from eval.answers import last_usage_id, spend_since

        with session_scope() as session:
            run = session.get(EvalRun, run_id)
            if run is None:
                raise LookupError(f"eval_runs {run_id} does not exist")
            after = int(run.metrics.get("llm_usage_after_id", 0))
            run.metrics = {  # a new dict, so SQLAlchemy sees the JSONB change
                **metrics,
                "status": status,
                "llm_usage_after_id": after,
                "llm_usage_until_id": last_usage_id(session),
                "spend_usd": str(spend_since(session, after)),
            }
            return dict(run.metrics)


# --- what a run talks to (tests replace each part) --------------------------------------------


@dataclass
class RunnerDeps:
    store: RunStore
    chat_client: Callable[[], ChatClient] = ChatClient
    judge: Callable[[], Judge] = default_judge
    relevancy_embedder: Callable[[], DenseEncoder] = EvalEmbedder
    search_fn: SearchFn = search
    retrieve_fn: Callable[..., list[SearchHit]] = retrieve
    sparse: Callable[[], SparseEncoder] = get_sparse_embedder
    cross_encoder: Callable[[], CrossEncoder] = get_cross_encoder
    loaded_reranker: Callable[[], RerankerKey | None] = loaded_reranker
    # The OpenAI client for query embeddings; None: the app's own, created on first use.
    openai_client: Callable[[], OpenAIClient] | None = None
    record_embedding: UsageRecorder = record_embedding_usage
    qdrant: Callable[[], QdrantClient] = get_qdrant
    query_cache_path: Path = CACHE_PATH
    budget: Budget | None = None
    git_sha: Callable[[], str | None] = git_commit
    clock: Callable[[], float] = time.perf_counter
    progress: Callable[[str], None] = lambda line: None
    peak_rss: Callable[[], float] = peak_rss_mib


# --- query embeddings of an index's model -----------------------------------------------------


class ShortenedEmbeddings:
    """An OpenAI client whose embeddings requests carry `dimensions` (a text-embedding-3
    vector shortened by the API, ablation 8). Only `embeddings.create` is used."""

    def __init__(self, client: OpenAIClient, dimensions: int) -> None:
        self._client = client
        self._dimensions = dimensions

    @property
    def embeddings(self) -> "ShortenedEmbeddings":
        return self

    def create(self, *, input: list[str], model: str) -> Any:
        create: Any = self._client.embeddings.create
        return create(input=input, model=model, dimensions=self._dimensions)


def index_embedder(
    index: IndexSpec, record_usage: UsageRecorder, client: Callable[[], OpenAIClient] | None
) -> DenseEmbedder:
    """A DenseEmbedder for the index's model (read from the applied settings), shortened
    when the index asks for it. Without a `client` factory the app's own client is used;
    for the plain case it is created only at the first request (no key needed before)."""
    if index.request_dimensions is None:
        return DenseEmbedder(client=client() if client else None, record_usage=record_usage)
    base = client() if client else default_openai_client()
    shortened: Any = ShortenedEmbeddings(base, index.request_dimensions)
    return DenseEmbedder(client=shortened, record_usage=record_usage)


# --- observing what actually ran ---------------------------------------------------------------


@dataclass
class Observed:
    collections: set[str] = field(default_factory=set)
    modes: set[str] = field(default_factory=set)
    embeddings: set[tuple[str, int]] = field(default_factory=set)
    rerankers: set[RerankerKey] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        return {
            "collections": sorted(self.collections),
            "modes": sorted(self.modes),
            "embeddings": sorted([m, d] for m, d in self.embeddings),
            "rerankers": [asdict(k) for k in sorted(self.rerankers, key=repr)],
        }


def expected(s: Settings, config: RetrievalConfig) -> Observed:
    """What a run on these settings may do (a subset is fine: e.g. no search call)."""
    return Observed(
        collections={s.qdrant_collection},
        modes={config.mode},
        embeddings={(s.embedding_model, s.embedding_dim)} if config.mode != "bm25" else set(),
        rerankers={reranker_key()} if config.rerank else set(),
    )


def verify_observed(observed: Observed, allowed: Observed) -> None:
    problems = [
        f"{name}: did {sorted(map(str, getattr(observed, name)))}, "
        f"settings allow {sorted(map(str, getattr(allowed, name)))}"
        for name in ("collections", "modes", "embeddings", "rerankers")
        if not getattr(observed, name) <= getattr(allowed, name)
    ]
    if problems:
        raise ConfigMismatch("configuration mismatch: " + "; ".join(problems))


class ObservingDense:
    """A DenseEncoder that records the model and dimension the settings give it."""

    def __init__(self, inner: DenseEncoder, observed: Observed, s: Settings) -> None:
        self._inner = inner
        observed.embeddings.add((s.embedding_model, s.embedding_dim))

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        return self._inner.embed(texts, feature)


def observing_search(inner: SearchFn, observed: Observed) -> SearchFn:
    """The chat's search, recording the collection it searches (the one search() takes
    from the settings at call time) and the mode it is asked for."""

    def run(*args: Any, **kwargs: Any) -> list[SearchHit]:
        observed.collections.add(kwargs.get("collection") or get_settings().qdrant_collection)
        observed.modes.add(str(kwargs.get("mode")))
        return inner(*args, **kwargs)

    return run


class TimedReranker:
    """Hands out the configured cross-encoder, records which one was loaded, and times
    every rerank call. `load` loads it before the first question (and runs one warm-up
    pair), so the model load is never part of a question's rerank time."""

    def __init__(self, deps: RunnerDeps, observed: Observed) -> None:
        self._deps = deps
        self._observed = observed
        self.calls_ms: list[float] = []
        self.load_s: float | None = None

    def get(self) -> CrossEncoder:
        encoder = self._deps.cross_encoder()
        key = self._deps.loaded_reranker()
        if key is not None:
            self._observed.rerankers.add(key)
        return _Timed(encoder, self)

    def load(self) -> None:
        start = self._deps.clock()
        encoder = self.get()
        list(encoder.rerank("warm-up", ["warm-up"], batch_size=1))
        self.calls_ms.clear()
        self.load_s = round(self._deps.clock() - start, 2)

    def take_ms(self) -> float | None:
        """The rerank time since the last call (None: no rerank happened)."""
        total = sum(self.calls_ms) if self.calls_ms else None
        self.calls_ms.clear()
        return None if total is None else round(total, 1)


class _Timed:
    def __init__(self, inner: CrossEncoder, owner: TimedReranker) -> None:
        self._inner = inner
        self._owner = owner

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> list[float]:
        start = self._owner._deps.clock()
        scores = list(self._inner.rerank(query, documents, batch_size=batch_size))
        self._owner.calls_ms.append((self._owner._deps.clock() - start) * 1000)
        return scores


# --- per-question metrics ---------------------------------------------------------------------


def _reference_texts_of(item: GoldenItem, texts: dict[uuid.UUID, str]) -> list[str]:
    return [texts[c] for c in item.reference_chunk_ids if c in texts]


def retrieval_metrics(
    hits: Sequence[SearchHit], question: SetQuestion, texts: dict[uuid.UUID, str],
    same_chunking: bool,
) -> dict[str, Any]:  # fmt: skip
    """One question's retrieval metrics: ids (same chunking only) and text coverage."""
    item = question.item
    out: dict[str, Any] = {}
    if item.scored and same_chunking:
        out |= score([h.chunk_id for h in hits], item)
    references = _reference_texts_of(item, texts)
    if references:
        out["coverage@8"] = reference_coverage([h.text for h in hits], references, COVERAGE_K)
    return out


def source_metrics(
    sources: Sequence[SearchHit], question: SetQuestion, texts: dict[uuid.UUID, str],
    same_chunking: bool,
) -> dict[str, Any]:  # fmt: skip
    """The same on the sources the generator was given (all of them, at most chat_sources),
    and how many tokens of source text that was (larger chunks carry more text, which raises
    coverage and context recall by itself). No sources (the no-sources refusal) scores 0."""
    item = question.item
    out: dict[str, Any] = {"source_tokens": sum(count_tokens(h.text) for h in sources)}
    k = max(1, len(sources))
    if item.scored and same_chunking:
        out["source_recall"] = recall_at_k(
            [h.chunk_id for h in sources], item.reference_chunk_ids, k
        )
    references = _reference_texts_of(item, texts)
    if references:
        out["source_coverage"] = reference_coverage([h.text for h in sources], references, k)
    return out


def aggregate_retrieval(rows: Sequence[tuple[SetQuestion, dict[str, Any]]]) -> dict[str, Any]:
    """Per split: the mean of each retrieval metric where defined (with its count), and
    latency and rerank time percentiles."""
    out: dict[str, Any] = {}
    for split in SPLITS:
        chosen = [m for q, m in rows if split == "all" or q.split == split]
        if not chosen:
            continue
        agg: dict[str, Any] = {"n_questions": len(chosen)}
        for key in RETRIEVAL_METRICS:
            values = [float(v) for m in chosen if (v := m.get(key)) is not None]
            agg[key] = mean(values)
            agg[f"n_{key}"] = len(values)
        for key in ("retrieve_ms", "rerank_ms"):
            values = [float(v) for m in chosen if (v := m.get(key)) is not None]
            agg[f"{key}_p50"] = percentile(values, 50) if values else None
            agg[f"{key}_p95"] = percentile(values, 95) if values else None
        out[split] = agg
    return out


# --- the stored configuration -----------------------------------------------------------------


def _sha(path: Path) -> str | None:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def effective_config(s: Settings, resolved: Resolved, config: RetrievalConfig) -> dict[str, Any]:
    """The configuration as read back INSIDE the applied scope: what ran."""
    return {
        "retrieval": asdict(config),
        "collection": s.qdrant_collection,
        "index": asdict(resolved.index),
        "embedding": {
            "model": s.embedding_model,
            "dim": s.embedding_dim,
            "request_dimensions": resolved.index.request_dimensions,
            "price_per_1m": s.embedding_price_per_1m,
        },  # fmt: skip
        "reranker": (
            {
                "model": s.reranker_model,
                "threads": s.rerank_threads,
                "cpu_arena": s.rerank_cpu_arena,
                "batch_size": s.rerank_batch_size,
            }
            if config.rerank
            else None
        ),  # fmt: skip
        "stages": {
            "condense": s.condense_enabled,
            "plan": s.plan_enabled,
            "plan_rephrasings": s.plan_rephrasings_enabled,
            "verify": s.verify_enabled,
        },
        "chat_model": s.chat_model,
        "chat_sources": s.chat_sources,
    }


def base_config(
    suite: Suite, resolved: Resolved, part: Part, questions: Sequence[SetQuestion]
) -> dict[str, Any]:
    return {
        "kind": part,
        "suite": asdict(suite),
        "variant": resolved.describe(),
        "questions": {
            "heldout": sum(1 for q in questions if q.split == "heldout"),
            "generated": sum(1 for q in questions if q.split == "generated"),
            "ids": [q.item.id for q in questions],
        },
        "golden_sha256": _sha(GOLDEN_PATH),
        "generated_sha256": _sha(GENERATED_PATH),
        "process": {"runs_before": list(_process_runs)},
    }


def run_name(suite: Suite, resolved: Resolved, part: Part) -> str:
    return f"{suite.kind}:{resolved.variant.id}:{part}"


# --- running one part -------------------------------------------------------------------------


@dataclass(frozen=True)
class PartResult:
    run_id: int
    name: str
    status: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _Scoped:
    outside: Settings  # the configured settings (the configured index, the real collection)
    s: Settings  # the variant's settings, applied to the whole process


@contextmanager
def _scope(resolved: Resolved) -> Iterator[_Scoped]:
    outside = get_settings()
    with applied_settings(resolved.overrides(outside)) as s:
        yield _Scoped(outside, s)


def _check_index(deps: RunnerDeps, scoped: _Scoped, resolved: Resolved) -> None:
    """A variant index must be built before a run searches it."""
    if resolved.index == IndexSpec.configured(scoped.outside):
        return
    client = deps.qdrant()
    name = scoped.s.qdrant_collection
    if not client.collection_exists(name) or not client.count(name, exact=True).count:
        raise IndexMissing(
            f"collection {name!r} for {resolved.variant.id} is not built: "
            f"make eval-index VARIANT={resolved.variant.id} (ESTIMATE=1 first)"
        )


def _all_reference_ids(questions: Sequence[SetQuestion]) -> list[uuid.UUID]:
    return [c for q in questions for c in q.item.reference_chunk_ids]


def run_part(
    part: Part,
    resolved: Resolved,
    questions: Sequence[SetQuestion],
    deps: RunnerDeps,
    suite: Suite,
    *,
    embed_missing_vectors: bool = False,
) -> PartResult:
    """Run one part of one variant on `questions` and store it (see the module doc).
    The variant's settings apply to the whole process inside, and are restored after."""
    with _scope(resolved) as scoped:
        # Checked before anything is stored or paid: the reference ids exist in Postgres and
        # in the configured index, and a variant index is built.
        deps.store.check(questions, scoped.outside.qdrant_collection)
        _check_index(deps, scoped, resolved)
        if part == "retrieval":
            return _run_retrieval(resolved, questions, deps, suite, scoped, embed_missing_vectors)
        return _run_answers(resolved, questions, deps, suite, scoped)


def _start(
    deps: RunnerDeps, suite: Suite, resolved: Resolved, part: Part,
    questions: Sequence[SetQuestion], config: dict[str, Any],
) -> tuple[int, dict[str, int], str]:  # fmt: skip
    item_ids = deps.store.sync_items(questions)
    name = run_name(suite, resolved, part)
    run_id = deps.store.start(name, config, deps.git_sha())
    return run_id, item_ids, name


def _run_retrieval(
    resolved: Resolved, questions: Sequence[SetQuestion], deps: RunnerDeps, suite: Suite,
    scoped: _Scoped, embed_missing_vectors: bool,
) -> PartResult:  # fmt: skip
    s = scoped.s
    config = RetrievalConfig.from_settings()
    texts = deps.store.reference_texts(_all_reference_ids(questions))
    same_chunking = resolved.index.same_chunking(scoped.outside)
    observed = Observed()

    dense: DenseEncoder | None = None
    if config.mode != "bm25":
        cache = QueryVectorCache(deps.query_cache_path, s.embedding_model, s.embedding_dim)
        missing = cache.missing([q.item.question for q in questions])
        if missing and not embed_missing_vectors:
            tokens, requests, cost = embedding_estimate(missing)
            raise MissingQueryVectors(
                f"{len(missing)} of {len(questions)} questions have no cached "
                f"{s.embedding_model}/{s.embedding_dim} embedding: {requests} request(s), "
                f"{tokens} tokens, about ${cost}. Run with EMBED=1 (paid; ask first)."
            )
        if missing:
            embed_missing(cache, missing, index_embedder(resolved.index, deps.record_embedding,
                                                         deps.openai_client))  # fmt: skip
        dense = ObservingDense(CachedQueryEncoder(cache), observed, s)
    sparse = deps.sparse() if config.mode != "dense" else None
    timed = TimedReranker(deps, observed)
    encoder: CrossEncoder | None = None
    if config.rerank:
        timed.load()  # a missing model stops here, before anything is stored
        encoder = timed.get()

    run_config_ = base_config(suite, resolved, "retrieval", questions) | {
        "effective": effective_config(s, resolved, config),
        "top_k": TOP_K,
        "filters": None,
        "same_chunking": same_chunking,
    }
    run_id, item_ids, name = _start(deps, suite, resolved, "retrieval", questions, run_config_)
    rows: list[tuple[SetQuestion, dict[str, Any]]] = []
    status = "complete"
    try:
        client = deps.qdrant()
        for k, question in enumerate(questions, start=1):
            start = deps.clock()
            hits = deps.retrieve_fn(
                question.item.question, None, limit=TOP_K, config=config, dense=dense,
                sparse=sparse, cross_encoder=encoder, client=client,
                collection=s.qdrant_collection,
            )  # fmt: skip
            elapsed = (deps.clock() - start) * 1000
            observed.collections.add(s.qdrant_collection)
            observed.modes.add(config.mode)
            item = question.item
            metrics = {
                **retrieval_metrics(hits, question, texts, same_chunking),
                "golden_id": item.id, "split": question.split, "type": item.type,
                "retrieve_ms": round(elapsed, 1), "rerank_ms": timed.take_ms(),
                "hits": [hit_record(h) for h in hits],
            }  # fmt: skip
            deps.store.add(run_id, item_ids[item.id], [h.chunk_id for h in hits], None, metrics)
            rows.append((question, metrics))
            deps.progress(f"[{k}/{len(questions)}] {name} {item.id}")
        verify_observed(observed, expected(s, config))
    except BaseException as err:  # also Ctrl-C: record what was done, then stop
        status = f"stopped: {describe(err)}"
        raise
    finally:
        _process_runs.append(f"{resolved.variant.id}:retrieval")
        final = deps.store.finish(run_id, {
            **aggregate_retrieval(rows), "observed": observed.as_dict(),
            "reranker_load_s": timed.load_s, "peak_rss_mib": deps.peak_rss(),
        }, status)  # fmt: skip
    return PartResult(run_id, name, status, final)


def _run_answers(
    resolved: Resolved, questions: Sequence[SetQuestion], deps: RunnerDeps, suite: Suite,
    scoped: _Scoped,
) -> PartResult:  # fmt: skip
    s = scoped.s
    config = RetrievalConfig.from_settings()
    texts = deps.store.reference_texts(_all_reference_ids(questions))
    same_chunking = resolved.index.same_chunking(scoped.outside)
    observed = Observed()
    timed = TimedReranker(deps, observed)
    if config.rerank:
        timed.load()  # a missing model stops here, before anything is paid or stored

    def dense(record: UsageRecorder) -> DenseEncoder:
        return ObservingDense(index_embedder(resolved.index, record, deps.openai_client),
                              observed, get_settings())  # fmt: skip

    chat = ChatDeps(
        chat=deps.chat_client(), store=NullChatStore(), tracer=NoopTracer(),
        search_fn=observing_search(deps.search_fn, observed), dense=dense,
        sparse=deps.sparse, cross_encoder=timed.get, record_embedding=deps.record_embedding,
    )  # fmt: skip
    judge = deps.judge()
    relevancy = deps.relevancy_embedder()
    run_config_ = {
        **run_config(s, judge, questions),  # the judge, prompts and question files (part 1)
        **base_config(suite, resolved, "answers", questions),  # its "questions" lists the ids
        "effective": effective_config(s, resolved, config),
        "same_chunking": same_chunking,
    }
    run_id, item_ids, name = _start(deps, suite, resolved, "answers", questions, run_config_)
    rows: list[Row] = []
    rerank_ms: list[float] = []
    status = "complete"
    try:
        for k, question in enumerate(questions, start=1):
            if deps.budget is not None:
                deps.budget.before_question()
            sample = collect_sample(question, chat, s, clock=deps.clock)
            if deps.budget is not None:
                deps.budget.check()  # a refused request: this question is not stored
            scores = score_sample(sample, judge, relevancy)
            if deps.budget is not None:
                deps.budget.after_question()
            if sample.error is None:
                scores.metrics |= source_metrics([x.hit for x in sample.sources], question, texts,
                                                 same_chunking)  # fmt: skip
            took = timed.take_ms()
            if took is not None:
                rerank_ms.append(took)
            row: Row = (sample, scores)
            deps.store.add(run_id, item_ids[question.item.id],
                           [x.hit.chunk_id for x in sample.sources], sample.answer,
                           {**result_metrics(row), "rerank_ms": took})  # fmt: skip
            rows.append(row)
            state = sample.error or (
                f"refused ({sample.refusal})" if sample.refusal else "answered"
            )
            deps.progress(f"[{k}/{len(questions)}] {name} {question.item.id}: {state}")
        verify_observed(observed, expected(s, config))
    except BudgetExhausted as err:
        status = f"stopped: budget: {err}"
        raise
    except BaseException as err:  # also Ctrl-C: record what was done, then stop
        tripped = deps.budget.guard.tripped if deps.budget is not None else None
        if tripped is not None:
            # A refused request surfaces as the OpenAI client's connection error (for
            # example inside a judge step): it is the cap that stopped the run.
            status = f"stopped: budget: the egress guard stopped the run: {tripped}"
            raise BudgetExhausted(status) from err
        status = f"stopped: {describe(err)}"
        raise
    finally:
        _process_runs.append(f"{resolved.variant.id}:answers")
        judge_cost = sum((u.cost_usd for u in judge.spent), Decimal("0"))
        final = deps.store.finish(run_id, {
            **aggregate(rows, (*ANSWER_METRICS, *SOURCE_METRICS)),
            "judge_cost_usd": str(judge_cost), "observed": observed.as_dict(),
            "rerank_ms_p50": percentile(rerank_ms, 50) if rerank_ms else None,
            "rerank_ms_p95": percentile(rerank_ms, 95) if rerank_ms else None,
            "reranker_load_s": timed.load_s, "peak_rss_mib": deps.peak_rss(),
            "budget": (
                {"cap_usd": str(deps.budget.cap_usd), "bound_usd": str(deps.budget.bound_usd)}
                if deps.budget is not None else None
            ),
        }, status)  # fmt: skip
    return PartResult(run_id, name, status, final)

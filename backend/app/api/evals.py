"""Read-only evaluation results for the /evals page (docs/adr/0012).

    GET /evals/runs                          every run, newest first, with headline metrics
    GET /evals/runs/{run_id}                 one run: its configuration and every metric
    GET /evals/runs/{run_id}/results         its questions, one row of scores each
    GET /evals/runs/{run_id}/results/{item}  one question: the retrieved chunks with their
                                             text, the answer, every score, the verdicts

Everything is read from Postgres (eval_runs, eval_results, eval_items, and for runs that
stored no chunk text, chunks): no Qdrant, no OpenAI call, no cost, nothing written.

Runs come in several shapes, all shown:
- retrieval runs of `make eval-retrieval` (names like "hybrid"), flat metrics;
- answer runs of `python -m eval.answers` ("answers:<name>"), metrics per split;
- the ablation runner's and the suites' runs ("<suite>:<variant>:<part>"), per split;
- selections ("selection:<step>"), which have no per-question results.
The headline metrics of a run are its "all" split (or its flat metrics); the page draws
trends and comparisons from them. Each question's chunk texts come from what the run
stored with it (a run on an ablation index searched other chunks than the chunks table
holds); only runs that stored none, on the app's own collection, are looked up in chunks.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db.models import Chunk, EvalItem, EvalResult, EvalRun, Paper
from app.db.session import session_scope
from app.ids import abs_url

router = APIRouter()

RunKind = Literal["retrieval", "answers", "selection", "other"]
SPLITS = ("all", "heldout", "generated")


class EvalRunSummaryOut(BaseModel):
    id: int
    name: str | None
    kind: RunKind
    suite: str | None  # ablation | smoke | full | None (older runs)
    suite_id: str | None
    step: str | None
    variant: str | None
    label: str | None
    created_at: datetime
    git_sha: str | None
    status: str | None
    n_questions: int | None
    splits: list[str]
    metrics: dict[str, float | None]  # the headline numbers: the "all" split


class EvalRunsOut(BaseModel):
    runs: list[EvalRunSummaryOut]


class EvalRunOut(EvalRunSummaryOut):
    config: dict[str, Any]
    all_metrics: dict[str, Any]  # every stored metric, splits included


class EvalResultRowOut(BaseModel):
    item_id: int
    question_id: str | None  # the golden or generated id ("q01", "gen-001")
    question: str
    split: str | None  # heldout | generated
    type: str | None
    outcome: Literal["answered", "refused", "error", "retrieved"]  # see Outcome below
    refusal: str | None
    scores: dict[str, float | None]


class EvalResultsOut(BaseModel):
    run_id: int
    kind: RunKind
    results: list[EvalResultRowOut]


class RetrievedChunkOut(BaseModel):
    rank: int
    n: int | None  # the source number [n] the answer cites (answer runs)
    chunk_id: uuid.UUID
    arxiv_id: str | None
    title: str | None
    section_path: str | None
    chunk_type: str | None
    score: float | None
    text: str | None  # None only when neither the run nor the chunks table has it
    is_reference: bool
    abs_url: str | None


class EvalResultDetailOut(EvalResultRowOut):
    run_id: int
    reference_answer: str | None
    reference_chunk_ids: list[uuid.UUID]
    answer: str | None
    error: str | None
    latency_ms: float | None
    cost_usd: float | None
    retrieved: list[RetrievedChunkOut]
    judge: dict[str, Any]  # the judge's verdicts per metric (statements, sources, citations)
    judge_errors: dict[str, str]


# --- the rows as stored ---------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    id: int
    name: str | None
    config: Mapping[str, Any]
    metrics: Mapping[str, Any]
    git_sha: str | None
    created_at: datetime


@dataclass(frozen=True)
class ResultRecord:
    item_id: int
    question: str
    reference_answer: str | None
    reference_chunk_ids: list[uuid.UUID]
    retrieved_chunk_ids: list[uuid.UUID]
    answer: str | None
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class ChunkInfo:
    arxiv_id: str
    title: str
    section_path: str | None
    chunk_type: str
    text: str


class EvalStore(Protocol):
    def list_runs(self, limit: int) -> list[RunRecord]: ...
    def get_run(self, run_id: int) -> RunRecord | None: ...
    def results(self, run_id: int) -> list[ResultRecord]: ...
    def result(self, run_id: int, item_id: int) -> ResultRecord | None: ...
    def chunks(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, ChunkInfo]: ...


class SqlEvalStore:
    def list_runs(self, limit: int) -> list[RunRecord]:
        with session_scope() as session:
            rows = session.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit)).all()
            return [_run_record(r) for r in rows]

    def get_run(self, run_id: int) -> RunRecord | None:
        with session_scope() as session:
            run = session.get(EvalRun, run_id)
            return None if run is None else _run_record(run)

    def results(self, run_id: int) -> list[ResultRecord]:
        with session_scope() as session:
            rows = session.execute(
                select(EvalResult, EvalItem)
                .join(EvalItem, EvalItem.id == EvalResult.item_id)
                .where(EvalResult.run_id == run_id)
                .order_by(EvalResult.id)
            ).all()
            return [_result_record(r, i) for r, i in rows]

    def result(self, run_id: int, item_id: int) -> ResultRecord | None:
        with session_scope() as session:
            row = session.execute(
                select(EvalResult, EvalItem)
                .join(EvalItem, EvalItem.id == EvalResult.item_id)
                .where(EvalResult.run_id == run_id, EvalResult.item_id == item_id)
            ).first()
            return None if row is None else _result_record(row[0], row[1])

    def chunks(self, ids: list[uuid.UUID]) -> dict[uuid.UUID, ChunkInfo]:
        if not ids:
            return {}
        with session_scope() as session:
            rows = session.execute(
                select(Chunk, Paper.title)
                .join(Paper, Paper.arxiv_id == Chunk.arxiv_id)
                .where(Chunk.id.in_(ids))
            ).all()
            return {
                c.id: ChunkInfo(c.arxiv_id, title, c.section_path, c.chunk_type, c.text)
                for c, title in rows
            }


def _run_record(run: EvalRun) -> RunRecord:
    return RunRecord(run.id, run.name, dict(run.config or {}), dict(run.metrics or {}),
                     run.git_sha, run.created_at)  # fmt: skip


def _result_record(result: EvalResult, item: EvalItem) -> ResultRecord:
    return ResultRecord(
        item_id=item.id, question=item.question, reference_answer=item.reference_answer,
        reference_chunk_ids=list(item.reference_chunk_ids or []),
        retrieved_chunk_ids=list(result.retrieved_chunk_ids or []), answer=result.answer,
        metrics=dict(result.metrics or {}),
    )  # fmt: skip


# --- shaping --------------------------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A stored number (costs are stored as decimal strings); None for anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def run_kind(run: RunRecord) -> RunKind:
    name = run.name or ""
    if name.startswith("selection:"):
        return "selection"
    kind = run.config.get("kind")
    if kind in ("retrieval", "answers"):
        return kind  # type: ignore[no-any-return]
    if name.startswith("answers:"):
        return "answers"
    if ":" not in name and "mode" in run.config:
        return "retrieval"  # make eval-retrieval
    return "other"


def headline(run: RunRecord) -> tuple[dict[str, float | None], list[str], int | None]:
    """The run's "all" split as numbers (or its flat metrics), the splits it has, and its
    number of questions."""
    splits = [s for s in SPLITS if isinstance(run.metrics.get(s), dict)]
    source: Mapping[str, Any] = run.metrics["all"] if "all" in splits else run.metrics
    numbers = {k: _number(v) for k, v in source.items() if _number(v) is not None}
    n = source.get("n_questions")
    if n is None and "n_scored" in run.metrics:
        n = int(run.metrics.get("n_scored", 0)) + int(run.metrics.get("n_unscored", 0))
    return numbers, splits, int(n) if isinstance(n, int | float) else None


def summary(run: RunRecord) -> EvalRunSummaryOut:
    suite = run.config.get("suite") if isinstance(run.config.get("suite"), dict) else None
    variant = run.config.get("variant") if isinstance(run.config.get("variant"), dict) else None
    metrics, splits, n = headline(run)
    kind = run_kind(run)
    status_ = run.metrics.get("status") or ("complete" if kind == "retrieval" else None)
    return EvalRunSummaryOut(
        id=run.id, name=run.name, kind=kind,
        suite=suite.get("kind") if suite else None, suite_id=suite.get("id") if suite else None,
        step=(suite.get("step") if suite else None) or run.config.get("step"),
        variant=variant.get("variant") if variant else None,
        label=variant.get("label") if variant else None,
        created_at=run.created_at, git_sha=run.git_sha,
        status=None if status_ is None else str(status_),
        n_questions=n, splits=splits, metrics=metrics,
    )  # fmt: skip


Outcome = Literal["answered", "refused", "error", "retrieved"]


def _outcome(kind: RunKind, metrics: Mapping[str, Any]) -> Outcome:
    if metrics.get("error"):
        return "error"
    if metrics.get("refusal"):
        return "refused"
    return "answered" if kind == "answers" else "retrieved"


def scores(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    """A result's numeric per-question values (scores, timings, cost), by name."""
    return {k: n for k, v in metrics.items() if (n := _number(v)) is not None and k != "golden_id"}


def row(kind: RunKind, record: ResultRecord) -> EvalResultRowOut:
    m = record.metrics
    golden_id = m.get("golden_id")
    return EvalResultRowOut(
        item_id=record.item_id, question_id=str(golden_id) if golden_id is not None else None,
        question=record.question, split=m.get("split"), type=m.get("type"),
        outcome=_outcome(kind, m), refusal=m.get("refusal"), scores=scores(m),
    )  # fmt: skip


def retrieved(
    run: RunRecord, record: ResultRecord, lookup: Mapping[uuid.UUID, ChunkInfo]
) -> list[RetrievedChunkOut]:
    """The retrieved chunks in rank order, with their text: from what the run stored with
    the result ("sources" of an answer, "hits" of a retrieval run), else from `lookup`."""
    stored = record.metrics.get("sources") or record.metrics.get("hits") or []
    by_id = {str(h.get("chunk_id")): h for h in stored if isinstance(h, dict)}
    # Under another chunking (an ablation index) a chunk id names another text than the
    # reference chunk with the same id (ids are uuid5(paper, version, ord) under any
    # chunking), so no chunk is marked as the reference there.
    same_chunking = run.config.get("same_chunking", True) is not False
    references = set(record.reference_chunk_ids) if same_chunking else set()
    out: list[RetrievedChunkOut] = []
    for rank, chunk_id in enumerate(record.retrieved_chunk_ids, start=1):
        hit = by_id.get(str(chunk_id), {})
        info = lookup.get(chunk_id)
        arxiv_id = hit.get("arxiv_id") or (info.arxiv_id if info else None)
        out.append(RetrievedChunkOut(
            rank=rank, n=hit.get("n"), chunk_id=chunk_id, arxiv_id=arxiv_id,
            title=hit.get("title") or (info.title if info else None),
            section_path=hit.get("section_path") if hit else (info.section_path if info else None),
            chunk_type=hit.get("chunk_type") or (info.chunk_type if info else None),
            score=_number(hit.get("score")), text=hit.get("text") or (info.text if info else None),
            is_reference=chunk_id in references,
            abs_url=abs_url(arxiv_id) if arxiv_id else None,
        ))  # fmt: skip
    return out


def needs_lookup(run: RunRecord, record: ResultRecord) -> list[uuid.UUID]:
    """Chunk ids to read from the chunks table: only for a result that stored no texts,
    and only when the run searched the app's own collection (else the ids mean other
    chunks than the table's)."""
    if record.metrics.get("sources") or record.metrics.get("hits"):
        return []
    collection = run.config.get("collection") or (run.config.get("effective") or {}).get(
        "collection"
    )
    if collection not in (None, get_settings().qdrant_collection):
        return []
    return list(record.retrieved_chunk_ids)


def detail(
    run: RunRecord, record: ResultRecord, lookup: Mapping[uuid.UUID, ChunkInfo]
) -> EvalResultDetailOut:
    kind = run_kind(run)
    m = record.metrics
    base = row(kind, record)
    details, errors_ = m.get("details"), m.get("judge_errors")
    judge: dict[str, Any] = details if isinstance(details, dict) else {}
    errors: dict[str, Any] = errors_ if isinstance(errors_, dict) else {}
    return EvalResultDetailOut(
        **base.model_dump(), run_id=run.id, reference_answer=record.reference_answer,
        reference_chunk_ids=record.reference_chunk_ids, answer=record.answer,
        error=m.get("error"), latency_ms=_number(m.get("latency_ms") or m.get("retrieve_ms")),
        cost_usd=_number(m.get("cost_usd")), retrieved=retrieved(run, record, lookup),
        judge=dict(judge), judge_errors={str(k): str(v) for k, v in errors.items()},
    )  # fmt: skip


# --- routes ---------------------------------------------------------------------------------


def get_eval_store() -> EvalStore:
    """The real store (tests override this dependency)."""
    return SqlEvalStore()


Store = Annotated[EvalStore, Depends(get_eval_store)]


async def _run_or_404(store: EvalStore, run_id: int) -> RunRecord:
    run = await run_in_threadpool(store.get_run, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown evaluation run")
    return run


@router.get("/evals/runs", response_model=EvalRunsOut)
async def list_runs(store: Store, limit: Annotated[int, Query(ge=1, le=1000)] = 500) -> EvalRunsOut:
    """Every run, newest first, with its headline metrics (Postgres only)."""
    runs = await run_in_threadpool(store.list_runs, limit)
    return EvalRunsOut(runs=[summary(r) for r in runs])


@router.get("/evals/runs/{run_id}", response_model=EvalRunOut)
async def get_run(run_id: int, store: Store) -> EvalRunOut:
    run = await _run_or_404(store, run_id)
    return EvalRunOut(**summary(run).model_dump(), config=dict(run.config),
                      all_metrics=dict(run.metrics))  # fmt: skip


@router.get("/evals/runs/{run_id}/results", response_model=EvalResultsOut)
async def run_results(run_id: int, store: Store) -> EvalResultsOut:
    run = await _run_or_404(store, run_id)
    kind = run_kind(run)
    results = await run_in_threadpool(store.results, run_id)
    return EvalResultsOut(run_id=run_id, kind=kind, results=[row(kind, r) for r in results])


@router.get("/evals/runs/{run_id}/results/{item_id}", response_model=EvalResultDetailOut)
async def run_result(run_id: int, item_id: int, store: Store) -> EvalResultDetailOut:
    run = await _run_or_404(store, run_id)
    record = await run_in_threadpool(store.result, run_id, item_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no result for this question in the run")
    lookup = await run_in_threadpool(store.chunks, needs_lookup(run, record))
    return detail(run, record, lookup)

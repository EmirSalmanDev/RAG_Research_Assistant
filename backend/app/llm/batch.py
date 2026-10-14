"""Paper analyses through the OpenAI Batch API: submit now, collect within 24 hours.

Nothing here is urgent, and the Batch API costs half the chat price. Two steps:

1. `submit()` picks full-text papers that have no analysis from the current prompt
   version (at most `analysis_max_papers`), writes our `analysis_batches` row and one
   `analysis_batch_items` row per paper ('pending') and commits, then uploads a JSONL
   file (one chat request per paper, Structured Outputs with a strict JSON schema) and
   creates the batch. Our row is committed before OpenAI hears of it, so a crash leaves
   a 'created' row that the next `submit()` finishes instead of starting over.
2. `collect()` (periodic) asks OpenAI about every open batch. When one has ended, it
   reads the output file: each answer is validated against the schema and stored in
   `paper_analyses` (one per paper, replacing an older one), together with its
   'analysis' chunks and one `llm_usage` row at Batch prices, in one transaction per
   paper. Then the analysis chunks are embedded and indexed, and the batch is marked
   'collected'. Re-running `collect()` stores nothing twice: a paper whose item is no
   longer 'pending' is skipped, and unchanged chunks are not embedded again.

The paper text is untrusted third-party input (rule 8): it goes inside one delimited
<source> block and the prompt tells the model to ignore instructions found in it.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import and_, delete, exists, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import (
    AnalysisBatch,
    AnalysisBatchItem,
    Paper,
    PaperAnalysis,
    PaperText,
)
from app.db.models import Chunk as ChunkRow
from app.ingest.chunk import ANALYSIS_SECTIONS, FIRST_ANALYSIS_ORD, analysis_chunks
from app.ingest.embed import DenseEncoder, SparseEncoder
from app.ingest.tokens import count_tokens, get_encoding
from app.llm.usage import batch_cost, log_usage
from app.rag.prompts import Prompt, escape_block, load_prompt
from app.sources.http import call_with_retry

logger = logging.getLogger(__name__)

FEATURE = "paper-analysis-batch"  # llm_usage.feature
ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"  # the only window the Batch API offers
ENDED = frozenset({"completed", "failed", "expired", "cancelled"})  # OpenAI's final states
CREATED, COLLECTED, SUBMIT_FAILED = "created", "collected", "submit_failed"  # our states
PENDING, DONE, FAILED = "pending", "done", "failed"  # item states


class PaperAnalysisOut(BaseModel):
    """The JSON schema every answer must follow (sent as a strict json_schema)."""

    tldr: str
    contributions: list[str]
    method: str
    results: str
    limitations: str

    def fields(self) -> dict[str, str]:
        """As stored in `paper_analyses`: contributions one per line."""
        return {
            "tldr": self.tldr.strip(),
            "contributions": "\n".join(c.strip() for c in self.contributions if c.strip()),
            "method": self.method.strip(),
            "results": self.results.strip(),
            "limitations": self.limitations.strip(),
        }


def response_format() -> dict[str, Any]:
    """The `response_format` of every request: what `chat.completions.parse` sends for
    the same model (the SDK's own conversion to a strict schema; checked by a test)."""
    from openai.lib._parsing._completions import type_to_response_format_param

    return dict(type_to_response_format_param(PaperAnalysisOut))  # type: ignore[arg-type]


# --- the requests ---------------------------------------------------------------------------


def custom_id(arxiv_id: str) -> str:
    return f"paper-{arxiv_id}"


def _attr(value: str) -> str:
    return escape_block(value).replace('"', "&quot;").replace("\n", " ")


def paper_block(paper: Paper, markdown: str, max_tokens: int) -> str:
    """The user message: the paper inside one <source> block, its text cut to
    `max_tokens` tokens (the block says when it was cut)."""
    ids = get_encoding().encode_ordinary(markdown)
    cut = len(ids) > max_tokens
    text = get_encoding().decode(ids[:max_tokens]) if cut else markdown
    return (
        f'<source arxiv_id="{_attr(paper.arxiv_id)}" title="{_attr(paper.title)}" '
        f'cut_off="{"yes" if cut else "no"}">\n{escape_block(text.strip())}\n</source>'
    )


def request_line(paper: Paper, markdown: str, prompt: Prompt) -> dict[str, Any]:
    """One line of the input JSONL file: a chat request for one paper."""
    s = get_settings()
    return {
        "custom_id": custom_id(paper.arxiv_id),
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": s.analysis_model,
            "messages": [
                {"role": "system", "content": prompt.text},
                {
                    "role": "user",
                    "content": paper_block(paper, markdown, s.analysis_max_input_tokens),
                },
            ],
            "response_format": response_format(),
            "temperature": s.chat_temperature,
            "max_completion_tokens": s.analysis_max_output_tokens,
        },
    }


@dataclass
class Estimate:
    """Upper bound of what a submission can cost (input counted with the embedding
    tokenizer, which counts English text at least as high as the chat model's)."""

    papers: int
    tokens_in: int
    max_tokens_out: int

    @property
    def max_cost_usd(self) -> Decimal:
        return batch_cost(self.tokens_in, self.max_tokens_out)


def estimate(lines: Sequence[dict[str, Any]]) -> Estimate:
    tokens_in = sum(count_tokens(m["content"]) for line in lines for m in line["body"]["messages"])
    max_out = sum(int(line["body"]["max_completion_tokens"]) for line in lines)
    return Estimate(papers=len(lines), tokens_in=tokens_in, max_tokens_out=max_out)


# --- which papers -------------------------------------------------------------------------------


def select_papers(
    session: Session,
    prompt_version: str,
    *,
    max_papers: int,
    max_attempts: int,
    arxiv_ids: Sequence[str] | None = None,
) -> list[str]:
    """Full-text papers (indexed, text stored) without an analysis from `prompt_version`,
    not queued already, and with fewer than `max_attempts` failed tries with this prompt
    version. Newest first."""
    analysed = exists().where(
        PaperAnalysis.arxiv_id == Paper.arxiv_id, PaperAnalysis.prompt_version == prompt_version
    )
    queued = exists().where(
        AnalysisBatchItem.arxiv_id == Paper.arxiv_id, AnalysisBatchItem.status == PENDING
    )
    failures = (
        select(func.count())
        .select_from(AnalysisBatchItem)
        .join(AnalysisBatch, AnalysisBatch.id == AnalysisBatchItem.batch_id)
        .where(
            AnalysisBatchItem.arxiv_id == Paper.arxiv_id,
            AnalysisBatchItem.status == FAILED,
            AnalysisBatch.prompt_version == prompt_version,
        )
        .scalar_subquery()
    )
    query = (
        select(Paper.arxiv_id)
        .join(PaperText, PaperText.arxiv_id == Paper.arxiv_id)
        .where(Paper.fulltext_status == "indexed", ~analysed, ~queued, failures < max_attempts)
        .order_by(Paper.published_at.desc(), Paper.arxiv_id)
        .limit(max_papers)
    )
    if arxiv_ids is not None:
        query = query.where(Paper.arxiv_id.in_(list(arxiv_ids)))
    return list(session.scalars(query))


def build_lines(session: Session, arxiv_ids: Sequence[str], prompt: Prompt) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Paper, PaperText.markdown)
        .join(PaperText, PaperText.arxiv_id == Paper.arxiv_id)
        .where(Paper.arxiv_id.in_(list(arxiv_ids)))
        .order_by(Paper.arxiv_id)
    ).all()
    return [request_line(paper, markdown, prompt) for paper, markdown in rows]


def jsonl(lines: Sequence[dict[str, Any]]) -> bytes:
    """The input file exactly as uploaded: one JSON request per line."""
    return "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines).encode()


# --- submit -----------------------------------------------------------------------------------


def _call[T](fn: Callable[[], T]) -> T:
    """One OpenAI request with the chat client's retry rules (429, 5xx, network)."""
    from app.llm.client import _as_retryable

    return call_with_retry(_as_retryable(fn))


def _send(session: Session, client: Any, batch: AnalysisBatch, prompt: Prompt) -> None:
    """Upload the batch's input file (unless done before a crash), then create the
    OpenAI batch. Each id is committed as soon as OpenAI returns it."""
    if batch.input_file_id is None:
        ids = [i.arxiv_id for i in batch_items(session, batch.id, PENDING)]
        data = jsonl(build_lines(session, ids, prompt))
        uploaded = _call(
            lambda: client.files.create(file=(f"analyses-{batch.id}.jsonl", data), purpose="batch")
        )
        batch.input_file_id = uploaded.id
        session.commit()
    created = _call(
        lambda: client.batches.create(
            input_file_id=batch.input_file_id, endpoint=ENDPOINT,
            completion_window=COMPLETION_WINDOW,
            metadata={"feature": FEATURE, "batch_row": str(batch.id)},
        )
    )  # fmt: skip
    batch.openai_batch_id, batch.status = created.id, created.status
    batch.submitted_at = datetime.now(UTC)
    session.commit()


def batch_items(
    session: Session, batch_id: int, status: str | None = None
) -> list[AnalysisBatchItem]:
    query = select(AnalysisBatchItem).where(AnalysisBatchItem.batch_id == batch_id)
    if status is not None:
        query = query.where(AnalysisBatchItem.status == status)
    return list(session.scalars(query.order_by(AnalysisBatchItem.arxiv_id)))


def _submit_failed(session: Session, batch: AnalysisBatch, err: Exception) -> None:
    session.rollback()
    message = f"{type(err).__name__}: {err}"[:500]
    batch.status, batch.errors = SUBMIT_FAILED, {"submit": message}
    for item in batch_items(session, batch.id, PENDING):
        item.status, item.error = FAILED, message  # free to be tried again (max_attempts)
    session.commit()


def submit(
    session: Session,
    client: Any,
    *,
    arxiv_ids: Sequence[str] | None = None,
    max_papers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Finish batches a crash left 'created', then submit one new batch (if any paper
    needs an analysis). `dry_run` only selects and estimates: no write, no API call."""
    s = get_settings()
    prompt = load_prompt(s.analysis_prompt)
    out: dict[str, Any] = {"prompt_version": prompt.id, "resumed": 0}
    if not dry_run:
        for stale in session.scalars(select(AnalysisBatch).where(AnalysisBatch.status == CREATED)):
            try:
                _send(session, client, stale, prompt)
                out["resumed"] += 1
            except Exception as err:
                _submit_failed(session, stale, err)
    ids = select_papers(
        session, prompt.id, max_papers=max_papers or s.analysis_max_papers,
        max_attempts=s.analysis_max_attempts, arxiv_ids=arxiv_ids,
    )  # fmt: skip
    lines = build_lines(session, ids, prompt) if ids else []
    est = estimate(lines)
    out.update(
        selected=ids, file_bytes=len(jsonl(lines)) if lines else 0, tokens_in=est.tokens_in,
        max_tokens_out=est.max_tokens_out, max_cost_usd=str(est.max_cost_usd),
    )  # fmt: skip
    if dry_run or not ids:
        return out
    batch = AnalysisBatch(
        status=CREATED, model=s.analysis_model, prompt_version=prompt.id,
        request_count=len(ids), errors={},
    )  # fmt: skip
    session.add(batch)
    session.flush()
    session.add_all(AnalysisBatchItem(batch_id=batch.id, arxiv_id=i, status=PENDING) for i in ids)
    session.commit()  # our record first: a crash after this is finished by the next submit
    try:
        _send(session, client, batch, prompt)
    except Exception as err:
        _submit_failed(session, batch, err)
        raise
    out.update(batch_id=batch.id, openai_batch_id=batch.openai_batch_id, status=batch.status)
    return out


# --- collect ---------------------------------------------------------------------------------


@dataclass
class Outcome:
    """One paper's result from the output or error file."""

    analysis: PaperAnalysisOut | None
    error: str | None
    tokens_in: int = 0
    tokens_out: int = 0
    billed: bool = False  # the request produced a completion (usage is reported)


def parse_line(line: dict[str, Any]) -> Outcome:
    """One line of the output (or error) file -> the analysis, or why there is none."""
    response = line.get("response") or {}
    if line.get("error") or response.get("status_code") != 200:
        err = line.get("error") or {}
        reason = err.get("message") or f"HTTP {response.get('status_code')}"
        return Outcome(None, f"request failed: {reason}"[:500])
    body = response.get("body") or {}
    usage = body.get("usage") or {}
    billed = Outcome(
        None, None, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
        billed=bool(usage),
    )  # fmt: skip
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    if message.get("refusal"):
        billed.error = f"refused: {message['refusal']}"[:500]
    elif choice.get("finish_reason") != "stop":
        billed.error = f"no complete answer (finish_reason {choice.get('finish_reason')})"
    else:
        try:
            billed.analysis = PaperAnalysisOut.model_validate_json(message.get("content") or "")
        except ValidationError as err:
            billed.error = f"answer does not match the schema: {err}"[:500]
    return billed


def _download(client: Any, file_id: str) -> str:
    content: Any = _call(lambda: client.files.content(file_id))
    return str(content.text)


def _retrieve(client: Any, openai_batch_id: str | None) -> Any:
    return _call(lambda: client.batches.retrieve(openai_batch_id))


def read_results(client: Any, *file_ids: str | None) -> dict[str, Outcome]:
    """Outcomes by custom_id from the output and error files (either may be missing)."""
    results: dict[str, Outcome] = {}
    for file_id in file_ids:
        if not file_id:
            continue
        text = _download(client, file_id)
        for raw in text.splitlines():
            if raw.strip():
                line = json.loads(raw)
                results[line["custom_id"]] = parse_line(line)
    return results


def store_analysis(
    session: Session, paper: Paper, analysis: PaperAnalysisOut, batch: AnalysisBatch
) -> int:
    """Replace the paper's analysis and its 'analysis' chunk rows (Postgres only).
    Returns how many chunks the analysis has."""
    from app.ingest.index import save_chunks

    fields = analysis.fields()
    values = {
        "arxiv_id": paper.arxiv_id, **fields, "model": batch.model,
        "prompt_version": batch.prompt_version, "batch_id": batch.id,
    }  # fmt: skip
    stmt = pg_insert(PaperAnalysis).values(**values)
    session.execute(
        stmt.on_conflict_do_update(
            constraint="uq_paper_analyses_arxiv_id",
            set_={
                **{k: stmt.excluded[k] for k in values if k != "arxiv_id"},
                "created_at": func.now(),
            },
        )
    )
    chunks = analysis_chunks(paper.arxiv_id, paper.version, paper.title, dict(fields))
    # A section left empty this time must not keep the previous analysis's chunk.
    last_ord = FIRST_ANALYSIS_ORD + len(ANALYSIS_SECTIONS) - 1
    session.execute(
        delete(ChunkRow).where(
            ChunkRow.arxiv_id == paper.arxiv_id,
            ChunkRow.chunk_type == "analysis",
            and_(ChunkRow.ord >= FIRST_ANALYSIS_ORD, ChunkRow.ord <= last_ord),
            ChunkRow.ord.not_in([c.ord for c in chunks]),
        )
    )
    save_chunks(session, paper.arxiv_id, chunks)
    return len(chunks)


def _finish_item(
    session: Session, batch: AnalysisBatch, item: AnalysisBatchItem, outcome: Outcome | None
) -> bool:
    """Store one paper's outcome and its usage in one transaction. True if stored."""
    if outcome is None:
        item.status, item.error = FAILED, f"no result in the batch (status {batch.status})"
        session.commit()
        return False
    if outcome.billed:  # billed even when the answer is unusable (rule 9)
        log_usage(
            session, feature=FEATURE, model=batch.model, tokens_in=outcome.tokens_in,
            tokens_out=outcome.tokens_out,
            cost_usd=batch_cost(outcome.tokens_in, outcome.tokens_out),
        )  # fmt: skip
    paper = session.get(Paper, item.arxiv_id)
    if outcome.analysis is None or paper is None:
        item.status, item.error = FAILED, outcome.error or "the paper is no longer stored"
        session.commit()
        return False
    store_analysis(session, paper, outcome.analysis, batch)
    item.status, item.error = DONE, None
    session.commit()
    return True


def _open_batches() -> Any:
    return select(AnalysisBatch).where(
        AnalysisBatch.openai_batch_id.is_not(None),
        AnalysisBatch.status.not_in([COLLECTED, SUBMIT_FAILED]),
    )


def open_batch_count(session: Session) -> int:
    """Batches OpenAI holds whose results are not stored yet."""
    return session.scalar(select(func.count()).select_from(_open_batches().subquery())) or 0


def collect(
    session: Session,
    client: Any,
    *,
    dense: DenseEncoder,
    sparse: SparseEncoder,
    index: Callable[..., int] | None = None,
) -> dict[str, Any]:
    """Check every open batch; store and index the results of the ones that ended.
    `index` defaults to `upsert_analysis_chunks` (tests pass a stand-in)."""
    from app.ingest.index import upsert_analysis_chunks

    index_chunks = index or upsert_analysis_chunks
    counts: dict[str, Any] = {"checked": 0, "ended": 0, "stored": 0, "failed": 0, "embedded": 0}
    open_batches = session.scalars(_open_batches().order_by(AnalysisBatch.id)).all()
    for batch in open_batches:
        counts["checked"] += 1
        remote = _retrieve(client, batch.openai_batch_id)
        batch.status = remote.status
        batch.output_file_id, batch.error_file_id = remote.output_file_id, remote.error_file_id
        session.commit()
        if remote.status not in ENDED:
            continue
        counts["ended"] += 1
        results = read_results(client, batch.output_file_id, batch.error_file_id)
        for item in batch_items(session, batch.id, PENDING):
            stored = _finish_item(session, batch, item, results.get(custom_id(item.arxiv_id)))
            counts["stored" if stored else "failed"] += 1
        # Index after storing (a re-run embeds nothing that is already indexed).
        for item in batch_items(session, batch.id, DONE):
            counts["embedded"] += index_chunks(session, item.arxiv_id, dense=dense, sparse=sparse)
            session.commit()
        batch.status, batch.collected_at = COLLECTED, datetime.now(UTC)
        session.commit()
    return counts


def pending_batch_id(session: Session, arxiv_id: str) -> int | None:
    """The batch a paper is queued in, if any (for the paper page's 'queued' state)."""
    return session.scalar(
        select(AnalysisBatchItem.batch_id).where(
            AnalysisBatchItem.arxiv_id == arxiv_id, AnalysisBatchItem.status == PENDING
        )
    )

"""app/llm/batch.py against a throwaway Postgres database and a `chunks_test_<random>`
Qdrant collection, with a fake OpenAI client (files + batches) and fake embedders.
No network, no OpenAI; the real collection is checked to be untouched."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import (
    AnalysisBatch,
    AnalysisBatchItem,
    LLMUsage,
    Paper,
    PaperAnalysis,
    PaperText,
)
from app.db.models import Chunk as ChunkRow
from app.ingest.chunk import FIRST_ANALYSIS_ORD, abstract_chunk
from app.ingest.index import save_chunks, upsert_analysis_chunks, upsert_paper_chunks
from app.llm import batch
from app.llm.usage import batch_cost
from tests.integration.test_indexing import FakeDense, FakeSparse, collection  # noqa: F401

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

FIELDS = {
    "tldr": "An invented model does an invented thing.",
    "contributions": ["First invented idea.", "Second invented idea."],
    "method": "An invented method.",
    "results": "Invented results.",
    "limitations": "Only one invented dataset.",
}


# --- a fake OpenAI client: files and batches -------------------------------------------------


@dataclass
class Obj:
    id: str
    status: str = "validating"
    output_file_id: str | None = None
    error_file_id: str | None = None


@dataclass
class Content:
    text: str


@dataclass
class FakeOpenAI:
    """The parts of `openai.OpenAI()` that batch.py uses. `uploads` keeps every input
    file; `outputs` maps a file id to the JSONL text OpenAI would return."""

    uploads: dict[str, bytes] = field(default_factory=dict)
    batches_made: list[dict[str, Any]] = field(default_factory=list)
    remote: dict[str, Obj] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    fail_create: Exception | None = None
    calls: list[str] = field(default_factory=list)

    @property
    def files(self) -> "FakeOpenAI":
        return self

    @property
    def batches(self) -> "FakeOpenAI":
        return self

    # files.create / files.content
    def create(self, **kw: Any) -> Obj:
        if "purpose" in kw:
            self.calls.append("files.create")
            name, data = kw["file"]
            file_id = f"file-in-{len(self.uploads) + 1}"
            self.uploads[file_id] = data
            return Obj(id=file_id)
        self.calls.append("batches.create")
        if self.fail_create is not None:
            raise self.fail_create
        self.batches_made.append(kw)
        made = Obj(id=f"batch_{len(self.batches_made)}")
        self.remote[made.id] = made
        return made

    def content(self, file_id: str) -> Content:
        self.calls.append(f"files.content {file_id}")
        return Content(self.outputs[file_id])

    def retrieve(self, batch_id: str) -> Obj:
        self.calls.append(f"batches.retrieve {batch_id}")
        return self.remote[batch_id]

    def lines(self, file_id: str = "file-in-1") -> list[dict[str, Any]]:
        return [json.loads(raw) for raw in self.uploads[file_id].decode().splitlines()]

    def finish(self, batch_id: str, out: list[dict[str, Any]], err: list[dict[str, Any]]) -> None:
        """OpenAI ran the batch: these output and error lines."""
        remote = self.remote[batch_id]
        remote.status = "completed"
        if out:
            remote.output_file_id = f"file-out-{batch_id}"
            self.outputs[remote.output_file_id] = "".join(json.dumps(x) + "\n" for x in out)
        if err:
            remote.error_file_id = f"file-err-{batch_id}"
            self.outputs[remote.error_file_id] = "".join(json.dumps(x) + "\n" for x in err)


def answer(arxiv_id: str, fields: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
    content = json.dumps(fields or FIELDS)
    body = {
        "choices": [{"index": 0, "finish_reason": kw.get("finish", "stop"),
                     "message": {"role": "assistant", "content": content,
                                 "refusal": kw.get("refusal")}}],
        "usage": {"prompt_tokens": 2000, "completion_tokens": 300},
    }  # fmt: skip
    return {"custom_id": f"paper-{arxiv_id}", "error": None,
            "response": {"status_code": 200, "body": body}}  # fmt: skip


def failed(arxiv_id: str) -> dict[str, Any]:
    return {"custom_id": f"paper-{arxiv_id}", "response": None,
            "error": {"code": "invalid_request", "message": "invented failure"}}  # fmt: skip


# --- papers ---------------------------------------------------------------------------------


def _paper(s: Session, n: int, *, status: str = "indexed", text: bool = True) -> str:
    arxiv_id = f"2601.0000{n}"
    s.add(
        Paper(
            arxiv_id=arxiv_id, version=1, title=f"Invented {n}", abstract=f"Invented abstract {n}.",
            authors=[], categories=["cs.CL"], published_at=datetime(2026, 1, n, tzinfo=UTC),
            fulltext_status=status,
        )
    )  # fmt: skip
    s.flush()
    if text:
        s.add(PaperText(arxiv_id=arxiv_id, version=1, raw_markdown="raw",
                        markdown=f"# Intro\n\nInvented full text {n}.", token_count=8))  # fmt: skip
    s.commit()
    return arxiv_id


def _count(s: Session, model: Any, *where: Any) -> int:
    return s.scalar(select(func.count()).select_from(model).where(*where)) or 0


def _no_index(session: Session, arxiv_id: str, **kw: Any) -> int:
    return 0


# --- select and submit ----------------------------------------------------------------------


def test_selection_takes_full_text_papers_without_a_current_analysis(
    migrated_session: Session,
) -> None:
    s = migrated_session
    a, b, c = _paper(s, 1), _paper(s, 2), _paper(s, 3)
    _paper(s, 4, status="none", text=False)  # abstract only
    s.add(PaperAnalysis(arxiv_id=b, model="m", prompt_version="analyze_v1"))  # current
    s.add(PaperAnalysis(arxiv_id=c, model="m", prompt_version="analyze_v0"))  # older prompt
    s.commit()
    pick = batch.select_papers(s, "analyze_v1", max_papers=10, max_attempts=2)
    assert pick == [c, a]  # newest first; c is analysed again for the newer prompt
    assert batch.select_papers(s, "analyze_v1", max_papers=1, max_attempts=2) == [c]
    assert batch.select_papers(s, "analyze_v1", max_papers=10, max_attempts=2,
                               arxiv_ids=[a]) == [a]  # fmt: skip


def test_dry_run_estimates_without_writing_or_calling(migrated_session: Session) -> None:
    s = migrated_session
    a = _paper(s, 1)
    out = batch.submit(s, None, dry_run=True)
    assert out["selected"] == [a] and out["tokens_in"] > 0
    assert Decimal(out["max_cost_usd"]) > 0
    assert _count(s, AnalysisBatch) == 0  # nothing written
    client = FakeOpenAI()
    batch.submit(s, client)  # the real submission uploads exactly the bytes measured
    assert out["file_bytes"] == len(client.uploads["file-in-1"])


def test_submit_uploads_one_file_and_creates_one_batch(migrated_session: Session) -> None:
    s = migrated_session
    a, b = _paper(s, 1), _paper(s, 2)
    client = FakeOpenAI()
    out = batch.submit(s, client)
    assert client.calls == ["files.create", "batches.create"]
    assert [line["custom_id"] for line in client.lines()] == [f"paper-{a}", f"paper-{b}"]
    made = client.batches_made[0]
    assert made["endpoint"] == "/v1/chat/completions" and made["completion_window"] == "24h"
    assert made["input_file_id"] == "file-in-1"
    row = s.scalars(select(AnalysisBatch)).one()
    assert (row.openai_batch_id, row.status, row.request_count) == ("batch_1", "validating", 2)
    assert (row.prompt_version, row.input_file_id) == ("analyze_v1", "file-in-1")
    assert out["batch_id"] == row.id and row.submitted_at is not None
    assert {i.status for i in batch.batch_items(s, row.id)} == {"pending"}
    assert batch.pending_batch_id(s, a) == row.id  # the paper page shows "queued"
    # Nothing is queued twice: a second submit finds no paper.
    again = batch.submit(s, client)
    assert again["selected"] == [] and client.calls == ["files.create", "batches.create"]


def test_a_failed_submission_frees_the_papers_until_max_attempts(
    migrated_session: Session,
) -> None:
    s = migrated_session
    a = _paper(s, 1)
    client = FakeOpenAI(fail_create=ValueError("invented 400"))
    for _ in range(2):  # analysis_max_attempts = 2
        with pytest.raises(ValueError):
            batch.submit(s, client)
    rows = s.scalars(select(AnalysisBatch).order_by(AnalysisBatch.id)).all()
    assert [r.status for r in rows] == ["submit_failed", "submit_failed"]
    assert rows[0].input_file_id == "file-in-1" and "invented 400" in rows[0].errors["submit"]
    assert _count(s, AnalysisBatchItem, AnalysisBatchItem.status == "failed") == 2
    assert batch.submit(s, client)["selected"] == []  # given up on after two failures
    assert batch.pending_batch_id(s, a) is None


def test_a_created_row_is_reported_and_never_resubmitted(migrated_session: Session) -> None:
    s = migrated_session
    a = _paper(s, 1)
    # A crash (or an ambiguous creation) left a row with the input file but no batch.
    row = AnalysisBatch(status="created", model="gpt-4o-mini", prompt_version="analyze_v1",
                        request_count=1, input_file_id="file-in-9", errors={})  # fmt: skip
    s.add(row)
    s.flush()
    s.add(AnalysisBatchItem(batch_id=row.id, arxiv_id=a))
    s.commit()
    client = FakeOpenAI()
    out = batch.submit(s, client)
    assert out["unresolved_created"] == [row.id] and out["selected"] == []
    assert client.calls == []  # nothing resubmitted, and its paper is still queued
    s.refresh(row)
    assert row.status == "created" and batch.pending_batch_id(s, a) == row.id


def _openai_error(kind: str) -> Exception:
    """A real SDK exception, as the client would raise it."""
    import httpx2
    import openai

    from app.egress_guard import EgressRefused

    request = httpx2.Request("POST", "https://api.openai.com/v1/batches")
    if kind == "timeout":
        return openai.APITimeoutError(request=request)
    if kind == "guard":  # the SDK wraps a transport exception, keeping it as the cause
        err = openai.APIConnectionError(request=request)
        err.__cause__ = EgressRefused("egress guard refused POST api.openai.com/v1/batches")
        return err
    status = {"502": 502, "400": 400, "429": 429}[kind]
    cls = {502: openai.InternalServerError, 400: openai.BadRequestError,
           429: openai.RateLimitError}[status]  # fmt: skip
    return cls(f"invented {status}", response=httpx2.Response(status, request=request), body=None)


@pytest.mark.parametrize("kind", ["timeout", "502"])
def test_an_ambiguous_creation_leaves_the_row_created_and_stops(
    migrated_session: Session, kind: str
) -> None:
    s = migrated_session
    a, b = _paper(s, 1), _paper(s, 2)
    client = FakeOpenAI(fail_create=_openai_error(kind))
    with pytest.raises(batch.AmbiguousSubmitError) as raised:
        batch.submit(s, client)
    row = s.scalars(select(AnalysisBatch)).one()
    assert raised.value.batch_row_id == row.id and f"row {row.id}" in str(raised.value)
    assert client.calls == ["files.create", "batches.create"]  # created once, no retry
    s.refresh(row)
    assert (row.status, row.openai_batch_id, row.input_file_id) == ("created", None, "file-in-1")
    assert row.errors["create"].startswith("ambiguous: ")
    assert {i.status for i in batch.batch_items(s, row.id)} == {"pending"}
    # The next submit does not resubmit it, and its papers stay queued.
    again = batch.submit(s, FakeOpenAI())
    assert again["unresolved_created"] == [row.id] and again["selected"] == []
    assert batch.pending_batch_id(s, a) == batch.pending_batch_id(s, b) == row.id


@pytest.mark.parametrize("kind", ["400", "429", "guard"])
def test_a_clear_refusal_marks_the_batch_failed_without_retry(
    migrated_session: Session, kind: str
) -> None:
    s = migrated_session
    a = _paper(s, 1)
    client = FakeOpenAI(fail_create=_openai_error(kind))
    with pytest.raises(Exception) as raised:
        batch.submit(s, client)
    assert not isinstance(raised.value, batch.AmbiguousSubmitError)
    assert client.calls == ["files.create", "batches.create"]  # no retry, also for 429
    row = s.scalars(select(AnalysisBatch)).one()
    assert row.status == "submit_failed" and batch.pending_batch_id(s, a) is None
    assert batch.batch_items(s, row.id)[0].status == "failed"


# --- collect --------------------------------------------------------------------------------


def _submitted(s: Session, *ns: int) -> tuple[FakeOpenAI, list[str]]:
    ids = [_paper(s, n) for n in ns]
    client = FakeOpenAI()
    batch.submit(s, client)
    return client, ids


def test_collect_waits_while_the_batch_runs(migrated_session: Session) -> None:
    s = migrated_session
    client, _ = _submitted(s, 1)
    client.remote["batch_1"].status = "in_progress"
    counts = batch.collect(s, client, dense=FakeDense(), sparse=FakeSparse(), index=_no_index)
    assert (counts["checked"], counts["ended"]) == (1, 0)
    assert s.scalars(select(AnalysisBatch)).one().status == "in_progress"
    assert _count(s, PaperAnalysis) == 0 and batch.open_batch_count(s) == 1


def test_collect_stores_analyses_chunks_and_usage_once(migrated_session: Session) -> None:
    s = migrated_session
    client, (a, b, c) = _submitted(s, 1, 2, 3)
    client.finish(
        "batch_1",
        out=[answer(a), answer(b, refusal="I can't help with that.")],
        err=[failed(c)],
    )
    indexed: list[str] = []

    def index(session: Session, arxiv_id: str, **kw: Any) -> int:
        indexed.append(arxiv_id)
        return 4

    counts = batch.collect(s, client, dense=FakeDense(), sparse=FakeSparse(), index=index)
    assert counts == {"checked": 1, "ended": 1, "stored": 1, "failed": 2, "embedded": 4}
    stored = s.scalars(select(PaperAnalysis)).one()
    assert (stored.arxiv_id, stored.tldr, stored.prompt_version) == (
        a,
        FIELDS["tldr"],
        "analyze_v1",
    )
    assert stored.contributions == "First invented idea.\nSecond invented idea."
    row = s.scalars(select(AnalysisBatch)).one()
    assert stored.batch_id == row.id and row.status == "collected" and row.collected_at
    ords = s.scalars(select(ChunkRow.ord).where(ChunkRow.arxiv_id == a).order_by(ChunkRow.ord))
    assert list(ords) == [FIRST_ANALYSIS_ORD + n for n in range(4)]
    assert indexed == [a]
    # Usage: the stored answer and the refused one were billed; the failed request not.
    usage = s.scalars(select(LLMUsage)).all()
    assert [(u.feature, u.tokens_in, u.tokens_out) for u in usage] == [
        ("paper-analysis-batch", 2000, 300), ("paper-analysis-batch", 2000, 300),
    ]  # fmt: skip
    assert all(u.cost_usd == batch_cost(2000, 300) == Decimal("0.000240") for u in usage)
    items = {i.arxiv_id: i for i in batch.batch_items(s, row.id)}
    assert items[a].status == "done" and items[b].status == "failed"
    assert "refused" in (items[b].error or "") and "invented failure" in (items[c].error or "")
    # Running again stores and logs nothing more (the batch is collected).
    again = batch.collect(s, client, dense=FakeDense(), sparse=FakeSparse(), index=index)
    assert again["checked"] == 0 and _count(s, LLMUsage) == 2


def test_a_newer_analysis_replaces_the_old_one_and_its_chunks(migrated_session: Session) -> None:
    s = migrated_session
    client, (a,) = _submitted(s, 1)
    client.finish("batch_1", out=[answer(a)], err=[])
    batch.collect(s, client, dense=FakeDense(), sparse=FakeSparse(), index=_no_index)
    first = s.scalars(select(PaperAnalysis)).one()
    # A newer prompt version analyses the paper again; this time without limitations.
    first.prompt_version = "analyze_v0"
    s.commit()
    batch.submit(s, client)
    client.finish("batch_2", out=[answer(a, {**FIELDS, "tldr": "Newer.", "limitations": ""})],
                  err=[])  # fmt: skip
    batch.collect(s, client, dense=FakeDense(), sparse=FakeSparse(), index=_no_index)
    s.expire_all()
    newest = s.scalars(select(PaperAnalysis)).one()  # still one row per paper
    assert newest.id == first.id and (newest.tldr, newest.limitations) == ("Newer.", "")
    ords = s.scalars(select(ChunkRow.ord).where(ChunkRow.chunk_type == "analysis"))
    assert sorted(ords) == [FIRST_ANALYSIS_ORD + n for n in range(3)]  # limitations gone


def test_a_batch_that_expired_marks_its_papers_failed(migrated_session: Session) -> None:
    s = migrated_session
    client, (a,) = _submitted(s, 1)
    client.remote["batch_1"].status = "expired"
    counts = batch.collect(s, client, dense=FakeDense(), sparse=FakeSparse(), index=_no_index)
    assert counts["failed"] == 1
    item = s.scalars(select(AnalysisBatchItem)).one()
    assert item.status == "failed" and "status expired" in (item.error or "")


# --- indexing the analysis chunks -------------------------------------------------------------


def test_analysis_chunks_are_indexed_once_and_survive_a_reindex_of_the_paper(
    migrated_session: Session,
    qdrant: QdrantClient,
    collection: str,  # noqa: F811
) -> None:
    s = migrated_session
    client, (a,) = _submitted(s, 1)
    paper = s.get(Paper, a)
    assert paper is not None
    save_chunks(s, a, [abstract_chunk(a, 1, paper.title, paper.abstract)])
    s.commit()
    client.finish("batch_1", out=[answer(a)], err=[])
    dense = FakeDense()
    kw: dict[str, Any] = {"dense": dense, "sparse": FakeSparse(), "client": qdrant,
                          "collection": collection}  # fmt: skip

    def index(session: Session, arxiv_id: str, **_: Any) -> int:
        return upsert_analysis_chunks(session, arxiv_id, **kw)

    counts = batch.collect(s, client, dense=dense, sparse=FakeSparse(), index=index)
    assert counts["embedded"] == 4
    assert dense.calls[0][0] == "embed-analysis" and len(dense.calls[0][1]) == 4
    assert upsert_analysis_chunks(s, a, **kw) == 0  # unchanged: nothing embedded again

    upsert_paper_chunks(s, a, **kw)  # re-indexing abstract/section chunks
    s.commit()
    points, _ = qdrant.scroll(collection, with_payload=True, limit=20)
    types = sorted(p.payload["chunk_type"] for p in points if p.payload)
    assert types == ["abstract"] + ["analysis"] * 4

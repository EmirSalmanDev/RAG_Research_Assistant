"""Add a PDF that is not on arXiv (a journal paper) through the ingestion steps (docs/adr/0011).

    bytes -> PDF to markdown (in memory) -> clean -> chunk -> embed -> Postgres and Qdrant

The steps are the ones the arXiv path uses (`pdf_to_markdown`, `clean_stored_text`,
`chunk_stored_text`, `upsert_paper_chunks`), so chunk ids are `uuid5(id, version, ord)` and
the document is searched, cited and compared like any other paper. Differences:
- the id is `local-<first 12 hex digits of the sha256 of the PDF bytes>` (app/ids.py), so the
  same bytes always give the same id;
- the PDF itself is never stored and never served: only the parsed text is kept;
- there is no arXiv record: the title (and optionally the year and the abstract) come from
  the uploader. `fulltext_source` is `upload`, there are no categories and no GitHub link;
- the abstract chunk (ord 0) holds the abstract the uploader gave, else the first
  `upload_abstract_chars` characters of the cleaned body.

Idempotent: a second upload of the same bytes finds the paper and changes nothing (the
title you send is ignored; delete the upload to re-add it under another title), and makes no
embedding request. An upload that stopped half-way (the embedding failed) is finished by
uploading the same bytes again: Postgres already holds the text, and only missing points are
embedded.

Parsing runs in the calling process (PyMuPDF4LLM peaks at about 430 MiB for a long paper), one
upload at a time (`upload_slot`). Embeddings are logged to `llm_usage` as `embed-upload`.
"""

import hashlib
import logging
import re
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk as ChunkRow
from app.db.models import LibraryItem, Paper, PaperText
from app.features.spend import EmbeddingMeter, Spend
from app.ids import LOCAL_HASH_CHARS, LOCAL_PREFIX, is_local_id
from app.ingest.embed import DenseEncoder, SparseEncoder
from app.ingest.fulltext import _store_raw, pdf_to_markdown
from app.ingest.index import get_qdrant, upsert_paper_chunks
from app.ingest.pipeline import chunk_stored_text, clean_stored_text
from app.llm.usage import UsageRecorder, record_embedding_usage

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
EMBED_FEATURE = "embed-upload"  # llm_usage.feature of the embeddings of an upload
UPLOAD_SOURCE = "upload"  # papers.fulltext_source
TITLE_MAX_CHARS = 300
ABSTRACT_MAX_CHARS = 3000
FIRST_YEAR, LAST_YEAR = 1900, 2100


class UploadError(ValueError):
    """The upload cannot be accepted. Nothing was stored and nothing was spent."""


class EmptyUpload(UploadError):
    """The body is empty."""


class TooLarge(UploadError):
    """The body is over `upload_max_bytes`."""


class NotAPdf(UploadError):
    """The body does not start with the PDF magic bytes."""


class NoTextLayer(UploadError):
    """The PDF has (almost) no text: a scan. OCR is not part of this project."""


class InvalidMetadata(UploadError):
    """The title, year or abstract breaks a rule."""


class UploadBusy(RuntimeError):
    """Another upload is being processed (one at a time keeps the parser's memory bounded)."""


class UploadNotFound(LookupError):
    """No uploaded document with that id."""


class NotAnUpload(ValueError):
    """The id is an arXiv id: only uploaded documents (`local-` ids) can be deleted here."""


_slot = threading.Lock()


@contextmanager
def upload_slot() -> Iterator[None]:
    """The one place an upload runs. Raises UploadBusy at once instead of queueing: a waiting
    request would hold its whole body in memory."""
    if not _slot.acquire(blocking=False):
        raise UploadBusy("another upload is being processed; try again in a moment")
    try:
        yield
    finally:
        _slot.release()


def local_id_for(pdf: bytes) -> str:
    """`local-` + the first 12 hex digits of the sha256 of the PDF bytes."""
    return LOCAL_PREFIX + hashlib.sha256(pdf).hexdigest()[:LOCAL_HASH_CHARS]


# --- checks before anything is parsed -----------------------------------------------------------


def check_body(pdf: bytes) -> None:
    """Empty, too large, not a PDF: refused before parsing."""
    if not pdf:
        raise EmptyUpload("the upload is empty")
    limit = get_settings().upload_max_bytes
    if len(pdf) > limit:
        raise TooLarge(f"the upload is {len(pdf):,} bytes; the limit is {limit:,}")
    if not pdf.startswith(PDF_MAGIC):
        raise NotAPdf("the file is not a PDF (it does not start with %PDF-)")


def check_metadata(
    title: str, year: int | None, abstract: str | None
) -> tuple[str, int, str | None]:
    """The cleaned title, the year (the upload year when not given) and the abstract."""
    clean_title = " ".join(title.split())
    if not clean_title or len(clean_title) > TITLE_MAX_CHARS:
        raise InvalidMetadata(f"the title is required and has at most {TITLE_MAX_CHARS} characters")
    if year is None:
        year = datetime.now(UTC).year
    if not FIRST_YEAR <= year <= LAST_YEAR:
        raise InvalidMetadata(f"the year is between {FIRST_YEAR} and {LAST_YEAR}")
    clean_abstract = " ".join((abstract or "").split()) or None
    if clean_abstract is not None and len(clean_abstract) > ABSTRACT_MAX_CHARS:
        raise InvalidMetadata(f"the abstract has at most {ABSTRACT_MAX_CHARS} characters")
    return clean_title, year, clean_abstract


def abstract_from_body(markdown: str, max_chars: int) -> str:
    """The first `max_chars` characters of the body as plain running text, cut at a word."""
    flat = re.sub(r"[#*_`>|]+", " ", markdown)
    flat = " ".join(flat.split())
    if len(flat) <= max_chars:
        return flat
    return flat[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"


# --- the pipeline ---------------------------------------------------------------------------------


class _Labelled:
    """A DenseEncoder that files every request under one llm_usage feature name."""

    def __init__(self, inner: DenseEncoder, feature: str) -> None:
        self._inner = inner
        self._feature = feature

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        return self._inner.embed(texts, self._feature)


@dataclass
class UploadDeps:
    """What an upload talks to, so that tests can replace every part."""

    dense: Callable[[UsageRecorder], DenseEncoder]
    sparse: Callable[[], SparseEncoder]
    qdrant: Callable[[], QdrantClient] = get_qdrant
    qdrant_collection: str | None = None  # None: the configured collection
    parse: Callable[[bytes], str] = pdf_to_markdown
    record_embedding: UsageRecorder = record_embedding_usage

    @classmethod
    def default(cls) -> "UploadDeps":
        from app.ingest.embed import DenseEmbedder, get_sparse_embedder

        return cls(
            dense=lambda record: DenseEmbedder(record_usage=record), sparse=get_sparse_embedder
        )


@dataclass
class UploadResult:
    arxiv_id: str
    title: str
    year: int
    created: bool  # False: the same bytes were uploaded before; nothing was changed
    chunks: dict[str, int] = field(default_factory=dict)  # Postgres rows by chunk_type
    embedded: int = 0  # points embedded in this call
    spend: Spend = field(default_factory=Spend)


def _chunk_counts(session: Session, arxiv_id: str) -> dict[str, int]:
    rows = session.execute(
        select(ChunkRow.chunk_type, func.count())
        .where(ChunkRow.arxiv_id == arxiv_id)
        .group_by(ChunkRow.chunk_type)
    ).all()
    return {chunk_type: int(n) for chunk_type, n in rows}


def ingest_pdf(
    session: Session,
    pdf: bytes,
    *,
    title: str,
    year: int | None = None,
    abstract: str | None = None,
    deps: UploadDeps,
    collection: str | None = None,
) -> UploadResult:
    """Parse, clean, chunk, embed and index one PDF. Raises an UploadError before anything is
    stored or spent when the upload cannot be accepted. Commits."""
    settings = get_settings()
    check_body(pdf)
    clean_title, upload_year, clean_abstract = check_metadata(title, year, abstract)
    arxiv_id = local_id_for(pdf)
    spend = Spend()
    dense = _Labelled(deps.dense(EmbeddingMeter(spend, deps.record_embedding)), EMBED_FEATURE)
    paper = session.get(Paper, arxiv_id)
    created = paper is None

    if paper is None:
        raw = deps.parse(pdf)
        if len(raw.strip()) < settings.upload_min_text_chars:
            raise NoTextLayer(
                "the PDF has no text layer (a scan?); OCR is not supported. "
                f"Only {len(raw.strip())} characters could be read."
            )
        paper = Paper(
            arxiv_id=arxiv_id,
            version=1,
            title=clean_title,
            abstract=clean_abstract or "(pending)",
            authors=[],
            categories=[],
            primary_category=None,
            published_at=datetime(upload_year, 1, 1, tzinfo=UTC),
            fulltext_status="fetched",
            fulltext_source=UPLOAD_SOURCE,
        )
        session.add(paper)
        session.flush()
        _store_raw(session, paper, raw)
        tokens = clean_stored_text(session, arxiv_id)
        if tokens == 0:
            session.rollback()
            raise NoTextLayer("nothing was left of the PDF's text after cleaning")
        if clean_abstract is None:
            text = session.get(PaperText, arxiv_id)
            assert text is not None
            paper.abstract = abstract_from_body(text.markdown, settings.upload_abstract_chars)
        chunk_stored_text(session, arxiv_id)
        session.commit()  # Postgres first, before any money is spent
    elif session.get(PaperText, arxiv_id) is None:
        raise UploadError(f"{arxiv_id} exists without its text; delete the upload and add it again")
    else:
        chunk_stored_text(session, arxiv_id)  # finish an upload that stopped half-way (idempotent)
        session.commit()

    try:
        embedded = upsert_paper_chunks(
            session,
            arxiv_id,
            dense=dense,
            sparse=deps.sparse(),
            client=deps.qdrant(),
            collection=deps.qdrant_collection,
        )
    except Exception:
        session.rollback()
        raise
    if paper.fulltext_status != "indexed":
        paper.fulltext_status = "indexed"
    if collection:
        session.execute(
            pg_insert(LibraryItem)
            .values(arxiv_id=arxiv_id, collection=collection, status="to_read")
            .on_conflict_do_nothing()
        )
    session.commit()
    return UploadResult(
        arxiv_id=arxiv_id,
        title=paper.title,
        year=paper.published_at.year,
        created=created,
        chunks=_chunk_counts(session, arxiv_id),
        embedded=embedded,
        spend=spend,
    )


# --- delete ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeleteResult:
    arxiv_id: str
    chunks: int
    points: int
    library_items: int


def delete_upload(
    session: Session,
    arxiv_id: str,
    *,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> DeleteResult:
    """Remove an uploaded document: its Qdrant points first (a failure leaves the paper in
    place, so a retry finishes the job), then the paper row, whose chunks, text, library
    items and digest entries go with it (foreign keys cascade). Only `local-` ids; an arXiv
    paper is never deleted here. Commits."""
    if not is_local_id(arxiv_id):
        raise NotAnUpload(f"{arxiv_id} is not an uploaded document")
    paper = session.get(Paper, arxiv_id)
    if paper is None:
        raise UploadNotFound(f"no uploaded document {arxiv_id}")
    client = client or get_qdrant()
    collection = collection or get_settings().qdrant_collection
    chunks = int(session.scalar(select(func.count()).where(ChunkRow.arxiv_id == arxiv_id)) or 0)
    items = int(session.scalar(select(func.count()).where(LibraryItem.arxiv_id == arxiv_id)) or 0)
    by_paper = m.Filter(must=[m.FieldCondition(key="arxiv_id", match=m.MatchValue(value=arxiv_id))])
    points = client.count(collection, count_filter=by_paper, exact=True).count
    client.delete(collection, points_selector=m.FilterSelector(filter=by_paper), wait=True)
    session.delete(paper)
    session.commit()
    return DeleteResult(arxiv_id=arxiv_id, chunks=chunks, points=points, library_items=items)

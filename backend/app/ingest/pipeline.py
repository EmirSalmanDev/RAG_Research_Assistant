"""The per-paper pipeline: metadata, abstract, full text, clean, chunk, embed, index.

Each step reads its input from Postgres and writes its output there (Qdrant is derived
from the `chunks` rows), so every step can be re-run on its own and the Celery tasks
of the daily chain can call the same functions. `run_paper()` runs all of them for one
paper (used by `make ingest-one` and, later, "add by arXiv id" in the UI).

Idempotent: a second run on an unchanged paper re-reads the stored full text instead of
downloading it again, and makes no embedding calls (unchanged points are skipped).
"""

import logging
from collections import Counter
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import Chunk as ChunkRow
from app.db.models import Paper, PaperText
from app.ingest.chunk import Chunk, abstract_chunk, chunk_paper
from app.ingest.clean import clean_markdown
from app.ingest.embed import DenseEncoder, SparseEncoder
from app.ingest.fulltext import (
    FulltextSource,
    FulltextSources,
    fetch_fulltext,
    request_fulltext,
)
from app.ingest.index import index_abstracts, save_chunks, upsert_paper_chunks
from app.ingest.paper import Clients, IngestResult, ingest_one
from app.ingest.tokens import count_tokens

logger = logging.getLogger(__name__)


def clean_stored_text(session: Session, arxiv_id: str) -> int:
    """Clean `paper_texts.raw_markdown` into `markdown` and `references`.
    Returns the cleaned text's token count. The caller commits."""
    text = session.get(PaperText, arxiv_id)
    if text is None:
        raise ValueError(f"no stored full text for {arxiv_id}")
    cleaned = clean_markdown(text.raw_markdown)
    text.markdown, text.references = cleaned.markdown, cleaned.references
    text.token_count = count_tokens(cleaned.markdown)
    return text.token_count


def chunk_stored_text(session: Session, arxiv_id: str) -> list[Chunk]:
    """Chunk the cleaned text and save the rows (the abstract chunk too).

    Section rows from an earlier chunking that are not in the new set are deleted here
    (re-chunking the same version can give fewer chunks); their Qdrant points are
    removed by `upsert_paper_chunks`. Analysis chunks are never touched. The caller
    commits.
    """
    paper = session.get(Paper, arxiv_id)
    text = session.get(PaperText, arxiv_id)
    if paper is None or text is None:
        raise ValueError(f"no paper or no stored full text for {arxiv_id}")
    if text.version < paper.version:
        raise ValueError(f"{arxiv_id}: stored text is v{text.version}, paper is v{paper.version}")
    chunks = [
        abstract_chunk(arxiv_id, paper.version, paper.title, paper.abstract),
        *chunk_paper(arxiv_id, paper.version, paper.title, text.markdown),
    ]
    session.execute(
        delete(ChunkRow).where(
            ChunkRow.arxiv_id == arxiv_id,
            ChunkRow.chunk_type == "section",
            ChunkRow.id.not_in([c.id for c in chunks]),
        )
    )
    save_chunks(session, arxiv_id, chunks)
    return chunks


@dataclass
class PaperRunResult:
    arxiv_id: str
    metadata: IngestResult
    fulltext_status: str
    fulltext_source: str | None = None
    fulltext_reused: bool = False  # True: stored text of this version reused, no download
    tried: dict[str, str] = field(default_factory=dict)
    chunks: dict[str, int] = field(default_factory=dict)  # Postgres rows by chunk_type
    embedded: int = 0  # points embedded in this run (abstract + sections)


def run_paper(
    session: Session,
    arxiv_id: str,
    *,
    clients: Clients,
    sources: FulltextSources,
    dense: DenseEncoder,
    sparse: SparseEncoder,
    force_source: FulltextSource | None = None,
    qdrant: QdrantClient | None = None,
    collection: str | None = None,
) -> PaperRunResult:
    """Ingest one paper end to end: metadata (and enrichment), abstract index, mark the
    paper requested, full text, clean, chunk, embed and index, status 'indexed'.

    If a full-text step fails, the paper is marked 'failed' (a later manual request
    resets it) and the error is raised. If no source has a full text, the paper is
    'unavailable' and only its abstract is searchable.
    """
    metadata = ingest_one(session, clients, arxiv_id)  # commits; raises if arXiv fails
    arxiv_id = metadata.arxiv_id  # normalised (no version suffix)
    embedded = index_abstracts(
        session, [arxiv_id], dense=dense, sparse=sparse, client=qdrant, collection=collection
    )
    request_fulltext(session, arxiv_id)
    session.commit()

    paper = session.get(Paper, arxiv_id)
    assert paper is not None  # ingest_one stored it
    result = PaperRunResult(arxiv_id=arxiv_id, metadata=metadata, fulltext_status="requested")
    try:
        stored = session.get(PaperText, arxiv_id)
        reuse = (
            stored is not None
            and stored.version >= paper.version
            and paper.fulltext_source is not None
            and force_source in (None, paper.fulltext_source)
        )
        if reuse:
            result.fulltext_reused = True
        else:
            fetched = fetch_fulltext(session, arxiv_id, sources=sources, force_source=force_source)
            result.tried = fetched.tried
            if fetched.source is None:  # no source has a full text
                session.commit()
                result.fulltext_status = "unavailable"
                result.chunks = _chunk_counts(session, arxiv_id)
                result.embedded = embedded
                return result
            paper.fulltext_status = "fetched"
            session.commit()
        clean_stored_text(session, arxiv_id)
        chunk_stored_text(session, arxiv_id)
        session.commit()
        embedded += upsert_paper_chunks(
            session, arxiv_id, dense=dense, sparse=sparse, client=qdrant, collection=collection
        )
        paper.fulltext_status = "indexed"
        session.commit()
    except Exception:
        session.rollback()
        failed = session.get(Paper, arxiv_id)
        if failed is not None:
            failed.fulltext_status = "failed"
            session.commit()
        raise
    result.fulltext_status = "indexed"
    result.fulltext_source = paper.fulltext_source
    result.chunks = _chunk_counts(session, arxiv_id)
    result.embedded = embedded
    return result


def _chunk_counts(session: Session, arxiv_id: str) -> dict[str, int]:
    types = session.scalars(select(ChunkRow.chunk_type).where(ChunkRow.arxiv_id == arxiv_id))
    return dict(Counter(types))

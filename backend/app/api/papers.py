"""Read-only lookups the frontend needs: one paper, a paper's chunks, one chunk.

    GET /papers/{arxiv_id}           the paper's metadata and abstract
    GET /papers/{arxiv_id}/chunks    its chunks in reading order (Postgres only)
    GET /papers/{arxiv_id}/related   similar papers, papers it cites, papers citing it
    GET /chunks/{chunk_id}           one chunk with its text: what a citation chip opens

Nothing here calls OpenAI or costs anything. Everything reads Postgres only, except
"similar" in /related, which asks Qdrant for the neighbours of the paper's stored
abstract vector (app/features/related.py). That is why the
paper page lists chunks here instead of calling /search (which embeds the query).
The chat stream's `sources` events carry no chunk text, so the side panel fetches it
from /chunks/{chunk_id}. Results link to the arXiv abstract page, never to a PDF.

The queries sit behind a small store (like ChatStore in app/rag/conversations.py) so
the route tests run without a database.
"""

import logging
import uuid
from datetime import datetime
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db.models import Chunk, Paper
from app.db.session import session_scope
from app.features import related as related_papers
from app.features.related import RelatedPaper, SimilarStatus

logger = logging.getLogger(__name__)

router = APIRouter()

CITATION_LIST_LIMIT = 200  # "cites" and "cited by" are capped at this many papers

ABS_URL = "https://arxiv.org/abs/{arxiv_id}"


class PaperOut(BaseModel):
    arxiv_id: str
    version: int
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str | None
    published_at: datetime
    hf_upvotes: int | None
    citation_count: int | None
    github_url: str | None
    project_url: str | None
    fulltext_status: str
    abs_url: str


class ChunkOut(BaseModel):
    chunk_id: uuid.UUID
    arxiv_id: str
    title: str
    chunk_type: str  # abstract | section | analysis
    section_path: str | None
    ord: int
    text: str
    published_at: datetime
    abs_url: str


class PaperChunksOut(BaseModel):
    arxiv_id: str
    chunks: list[ChunkOut]


class RelatedPaperOut(BaseModel):
    arxiv_id: str
    title: str | None  # None for a referenced paper that is not in the database
    published_at: datetime | None
    held: bool  # the paper is in the database (it has a /paper page)
    score: float | None  # similarity, for "similar" only
    abs_url: str


class RelatedOut(BaseModel):
    arxiv_id: str
    # ok; not_indexed (the paper's abstract has no vector yet); unavailable (Qdrant failed)
    similar_status: SimilarStatus
    similar: list[RelatedPaperOut]
    cites: list[RelatedPaperOut]
    cited_by: list[RelatedPaperOut]


class PaperStore(Protocol):
    def get_paper(self, arxiv_id: str) -> PaperOut | None: ...

    def list_chunks(self, arxiv_id: str, limit: int) -> PaperChunksOut | None:
        """The paper's chunks by `ord`; None when the paper does not exist."""
        ...

    def get_chunk(self, chunk_id: uuid.UUID) -> ChunkOut | None: ...

    def related(self, arxiv_id: str, limit: int) -> RelatedOut | None:
        """Similar (at most `limit`), cites and cited by; None when the paper does not exist."""
        ...


def _related_out(paper: RelatedPaper) -> RelatedPaperOut:
    return RelatedPaperOut(
        arxiv_id=paper.arxiv_id, title=paper.title, published_at=paper.published_at,
        held=paper.held, score=paper.score, abs_url=ABS_URL.format(arxiv_id=paper.arxiv_id),
    )  # fmt: skip


def _author_names(authors: list[object]) -> list[str]:
    """`papers.authors` holds [{"name": ...}] objects; the API only needs the names."""
    names = []
    for author in authors:
        name = author.get("name") if isinstance(author, dict) else author
        if isinstance(name, str):
            names.append(name)
    return names


def _chunk_out(chunk: Chunk, paper: Paper) -> ChunkOut:
    return ChunkOut(
        chunk_id=chunk.id,
        arxiv_id=chunk.arxiv_id,
        title=paper.title,
        chunk_type=chunk.chunk_type,
        section_path=chunk.section_path,
        ord=chunk.ord,
        text=chunk.text,
        published_at=paper.published_at,
        abs_url=ABS_URL.format(arxiv_id=chunk.arxiv_id),
    )


class SqlPaperStore:
    def get_paper(self, arxiv_id: str) -> PaperOut | None:
        with session_scope() as session:
            paper = session.get(Paper, arxiv_id)
            if paper is None:
                return None
            return PaperOut(
                arxiv_id=paper.arxiv_id,
                version=paper.version,
                title=paper.title,
                abstract=paper.abstract,
                authors=_author_names(paper.authors),
                categories=list(paper.categories),
                primary_category=paper.primary_category,
                published_at=paper.published_at,
                hf_upvotes=paper.hf_upvotes,
                citation_count=paper.citation_count,
                github_url=paper.github_url,
                project_url=paper.project_url,
                fulltext_status=paper.fulltext_status,
                abs_url=ABS_URL.format(arxiv_id=paper.arxiv_id),
            )

    def list_chunks(self, arxiv_id: str, limit: int) -> PaperChunksOut | None:
        with session_scope() as session:
            paper = session.get(Paper, arxiv_id)
            if paper is None:
                return None
            rows = session.scalars(
                select(Chunk).where(Chunk.arxiv_id == arxiv_id).order_by(Chunk.ord).limit(limit)
            ).all()
            return PaperChunksOut(arxiv_id=arxiv_id, chunks=[_chunk_out(c, paper) for c in rows])

    def get_chunk(self, chunk_id: uuid.UUID) -> ChunkOut | None:
        with session_scope() as session:
            chunk = session.get(Chunk, chunk_id)
            if chunk is None:
                return None
            paper = session.get(Paper, chunk.arxiv_id)
            assert paper is not None  # chunks.arxiv_id is a foreign key to papers
            return _chunk_out(chunk, paper)

    def related(self, arxiv_id: str, limit: int) -> RelatedOut | None:
        from app.ingest.index import get_qdrant

        with session_scope() as session:
            paper = session.get(Paper, arxiv_id)
            if paper is None:
                return None
            cites = related_papers.cites(session, arxiv_id, CITATION_LIST_LIMIT)
            cited_by = related_papers.cited_by(session, arxiv_id, CITATION_LIST_LIMIT)
            version = paper.version
        similar: list[RelatedPaper] = []
        similar_status: SimilarStatus
        try:
            similar_status, similar = related_papers.similar_papers(
                get_qdrant(), get_settings().qdrant_collection, arxiv_id, version, limit
            )
        except Exception:  # Qdrant down: the citation lists still work
            logger.warning("similar papers for %s: Qdrant failed", arxiv_id, exc_info=True)
            similar_status = "unavailable"
        return RelatedOut(
            arxiv_id=arxiv_id, similar_status=similar_status,
            similar=[_related_out(p) for p in similar], cites=[_related_out(p) for p in cites],
            cited_by=[_related_out(p) for p in cited_by],
        )  # fmt: skip


def get_paper_store() -> PaperStore:
    """The real store (tests override this dependency)."""
    return SqlPaperStore()


Store = Annotated[PaperStore, Depends(get_paper_store)]


@router.get("/papers/{arxiv_id:path}/chunks", response_model=PaperChunksOut)
async def paper_chunks(
    arxiv_id: str,
    store: Store,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> PaperChunksOut:
    # `:path` because old arXiv ids contain a slash (hep-th/9901001).
    result = await run_in_threadpool(store.list_chunks, arxiv_id, limit)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown paper")
    return result


@router.get("/papers/{arxiv_id:path}/related", response_model=RelatedOut)
async def paper_related(
    arxiv_id: str,
    store: Store,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RelatedOut:
    """The paper page's three related-paper tabs. `limit` caps "similar"; the citation
    lists hold up to 200 papers each."""
    result = await run_in_threadpool(store.related, arxiv_id, limit)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown paper")
    return result


@router.get("/papers/{arxiv_id:path}", response_model=PaperOut)
async def get_paper(arxiv_id: str, store: Store) -> PaperOut:
    paper = await run_in_threadpool(store.get_paper, arxiv_id)
    if paper is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown paper")
    return paper


@router.get("/chunks/{chunk_id}", response_model=ChunkOut)
async def get_chunk(chunk_id: uuid.UUID, store: Store) -> ChunkOut:
    chunk = await run_in_threadpool(store.get_chunk, chunk_id)
    if chunk is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown chunk")
    return chunk

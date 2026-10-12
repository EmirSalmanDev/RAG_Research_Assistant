"""Read-only lookups the frontend needs: one paper, a paper's chunks, one chunk.

    GET /papers/{arxiv_id}           the paper's metadata and abstract
    GET /papers/{arxiv_id}/chunks    its chunks in reading order (Postgres only)
    GET /chunks/{chunk_id}           one chunk with its text: what a citation chip opens

Everything here reads Postgres only: no Qdrant, no OpenAI, no cost. That is why the
paper page lists chunks here instead of calling /search (which embeds the query).
The chat stream's `sources` events carry no chunk text, so the side panel fetches it
from /chunks/{chunk_id}. Results link to the arXiv abstract page, never to a PDF.

The queries sit behind a small store (like ChatStore in app/rag/conversations.py) so
the route tests run without a database.
"""

import uuid
from datetime import datetime
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.db.models import Chunk, Paper
from app.db.session import session_scope

router = APIRouter()

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


class PaperStore(Protocol):
    def get_paper(self, arxiv_id: str) -> PaperOut | None: ...

    def list_chunks(self, arxiv_id: str, limit: int) -> PaperChunksOut | None:
        """The paper's chunks by `ord`; None when the paper does not exist."""
        ...

    def get_chunk(self, chunk_id: uuid.UUID) -> ChunkOut | None: ...


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

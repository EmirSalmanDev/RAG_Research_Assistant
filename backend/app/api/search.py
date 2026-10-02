"""GET /search: the retrieval pipeline over HTTP (chunks only; answers come later).

Example: /search?q=how+does+lora+work&category=cs.CL&published_from=2026-09-01&limit=5

The pipeline (app/rag/retrieve.py) runs with the configured stages. It is synchronous
(qdrant-client, OpenAI and the reranker are blocking calls), so it runs in FastAPI's
thread pool and the event loop stays free for other requests. In the dense and hybrid
modes, every call embeds the question: one paid OpenAI request, logged to llm_usage.
"""

import uuid
from datetime import UTC, date, datetime, time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.ingest.embed import OpenAIKeyMissing
from app.rag.rerank import RerankerUnavailable
from app.rag.retrieve import RetrievalConfig, retrieve
from app.rag.search import SearchFilters, SearchHit

router = APIRouter()

# Results link to the arXiv abstract page, never to a PDF (we do not re-serve papers).
ABS_URL = "https://arxiv.org/abs/{arxiv_id}"


class SearchHitOut(BaseModel):
    chunk_id: uuid.UUID
    arxiv_id: str
    title: str
    section_path: str | None
    chunk_type: str
    published_at: datetime
    score: float  # the reranker's score when reranked, else the search score
    text: str
    abs_url: str

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "SearchHitOut":
        return cls(
            chunk_id=hit.chunk_id,
            arxiv_id=hit.arxiv_id,
            title=hit.title,
            section_path=hit.section_path,
            chunk_type=hit.chunk_type,
            published_at=datetime.fromisoformat(hit.published_at),
            score=hit.score,
            text=hit.text,
            abs_url=ABS_URL.format(arxiv_id=hit.arxiv_id),
        )


class SearchResponse(BaseModel):
    query: str
    mode: str  # dense | bm25 | hybrid
    reranked: bool
    hits: list[SearchHitOut]


def _filters(
    categories: list[str] | None,
    arxiv_ids: list[str] | None,
    published_from: date | None,
    published_to: date | None,
) -> SearchFilters:
    """Query parameters -> SearchFilters. Dates are whole days in UTC, both inclusive:
    'from' starts at 00:00 and 'to' ends at 23:59:59.999999."""
    if published_from and published_to and published_from > published_to:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "published_from is after published_to"
        )
    return SearchFilters(
        arxiv_ids=tuple(arxiv_ids or ()),
        categories=tuple(categories or ()),
        published_from=datetime.combine(published_from, time.min, UTC) if published_from else None,
        published_to=datetime.combine(published_to, time.max, UTC) if published_to else None,
    )


@router.get("/search", response_model=SearchResponse)
async def search_chunks(
    q: Annotated[str, Query(min_length=1, max_length=1000, description="the question")],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    category: Annotated[list[str] | None, Query(description="repeatable; any of")] = None,
    arxiv_id: Annotated[list[str] | None, Query(description="repeatable; any of")] = None,
    published_from: date | None = None,
    published_to: date | None = None,
) -> SearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "q is blank")
    filters = _filters(category, arxiv_id, published_from, published_to)
    config = RetrievalConfig.from_settings()
    try:
        hits = await run_in_threadpool(retrieve, query, filters, limit=limit, config=config)
    except (RerankerUnavailable, OpenAIKeyMissing) as err:
        # Setup problems, not request problems: say what is missing and how to fix it.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(err)) from err
    return SearchResponse(
        query=query,
        mode=config.mode,
        reranked=config.rerank,
        hits=[SearchHitOut.from_hit(h) for h in hits],
    )

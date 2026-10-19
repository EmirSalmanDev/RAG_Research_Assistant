"""Paper comparison and topic timeline over HTTP (docs/adr/0010).

    POST /compare   {"arxiv_ids": ["1706.03762", ...], "dimensions": ["method", ...]}
    POST /timeline  {"topic": "speculative decoding", "date_from": "2026-01-01", ...}

Both spend money (one embeddings request and one chat request each), only when called:
no page makes either call on load. Everything that can be refused is checked BEFORE the
first paid call: 422 for a request that breaks a rule (count, duplicates, unknown
dimension, bad dates), 404 for an unknown paper, 503 when OPENAI_API_KEY is missing.
502 means OpenAI or Qdrant failed after the retries; whatever was billed before the
failure is in llm_usage. Both routes are synchronous underneath (blocking clients), so
they run in FastAPI's thread pool. Nothing is stored.
"""

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.features.compare import (
    DIMENSIONS,
    MAX_PAPERS,
    MIN_PAPERS,
    CompareOut,
    Dimension,
    run_compare,
)
from app.features.deps import ExploreDeps
from app.features.errors import InvalidRequest, UnknownPapers, UpstreamError
from app.features.timeline import (
    DEFAULT_KEY_PAPERS,
    DEFAULT_MAX_PAPERS,
    KEY_PAPERS_CAP,
    MAX_PAPERS_CAP,
    TOPIC_MAX_CHARS,
    TOPIC_MIN_CHARS,
    TimelineOut,
    TimelineRequest,
    run_timeline,
)
from app.ingest.embed import OpenAIKeyMissing

router = APIRouter()

MAX_CATEGORIES = 10
MAX_ID_CHARS = 40


class CompareRequest(BaseModel):
    arxiv_ids: list[Annotated[str, Field(min_length=1, max_length=MAX_ID_CHARS)]] = Field(
        min_length=MIN_PAPERS, max_length=MAX_PAPERS
    )
    dimensions: list[Dimension] = Field(default_factory=lambda: list(DIMENSIONS), min_length=1)


class TimelineBody(BaseModel):
    topic: str = Field(min_length=TOPIC_MIN_CHARS, max_length=TOPIC_MAX_CHARS)
    date_from: date | None = None
    date_to: date | None = None
    categories: list[str] = Field(default_factory=list, max_length=MAX_CATEGORIES)
    max_papers: int = Field(DEFAULT_MAX_PAPERS, ge=1, le=MAX_PAPERS_CAP)
    key_papers: int = Field(DEFAULT_KEY_PAPERS, ge=1, le=KEY_PAPERS_CAP)


def get_explore_deps() -> ExploreDeps:
    """The real clients, store and tracer (tests override this dependency)."""
    return ExploreDeps.default()


Deps = Annotated[ExploreDeps, Depends(get_explore_deps)]


def _require_key() -> None:
    if get_settings().openai_api_key is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set (put it in .env)"
        )


def _http_error(err: Exception) -> HTTPException | None:
    if isinstance(err, InvalidRequest):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(err))
    if isinstance(err, UnknownPapers):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(err))
    if isinstance(err, OpenAIKeyMissing):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(err))
    if isinstance(err, UpstreamError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, str(err))
    return None


@router.post("/compare", response_model=CompareOut)
async def compare_papers(body: CompareRequest, deps: Deps) -> CompareOut:
    """A table of 2-5 papers by the chosen dimensions; every shown cell carries citations."""
    _require_key()
    try:
        return await run_in_threadpool(run_compare, body.arxiv_ids, body.dimensions, deps=deps)
    except Exception as err:
        mapped = _http_error(err)
        if mapped is None:
            raise
        raise mapped from err


@router.post("/timeline", response_model=TimelineOut)
async def topic_timeline(body: TimelineBody, deps: Deps) -> TimelineOut:
    """Papers on a topic by date, key papers marked, with a one-line "what changed"."""
    _require_key()
    request = TimelineRequest(
        topic=body.topic,
        date_from=body.date_from,
        date_to=body.date_to,
        categories=tuple(body.categories),
        max_papers=body.max_papers,
        key_papers=body.key_papers,
    )
    try:
        return await run_in_threadpool(run_timeline, request, deps=deps)
    except Exception as err:
        mapped = _http_error(err)
        if mapped is None:
            raise
        raise mapped from err

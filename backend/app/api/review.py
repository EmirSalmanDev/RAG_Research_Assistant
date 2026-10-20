"""The literature review over HTTP (docs/adr/0011).

    POST /library/review  {"collection": "Thesis ch. 2", "focus": "...", "max_papers": 30}

It spends money (one chat call per group of eight papers above eight, plus one writing call)
and only when called: no page makes it on load. Everything that can be refused is checked
BEFORE the first paid call: 404 for an unknown collection, 422 for a request that breaks a
rule (fewer than two papers, a bad focus or max_papers), 503 when OPENAI_API_KEY is missing.
502 means OpenAI failed after the retries or answered with something unusable; whatever was
billed before the failure is in llm_usage. The route is synchronous underneath (blocking
clients), so it runs in FastAPI's thread pool. Nothing is stored.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.features.errors import InvalidRequest, UnknownCollection, UpstreamError
from app.features.library import NAME_MAX_CHARS
from app.features.review import FOCUS_MAX_CHARS, ReviewDeps, ReviewOut, run_review
from app.ingest.embed import OpenAIKeyMissing

router = APIRouter()


class ReviewRequest(BaseModel):
    collection: str = Field(min_length=1, max_length=NAME_MAX_CHARS)
    focus: str | None = Field(
        None, max_length=FOCUS_MAX_CHARS, description="a topic hint for the draft (optional)"
    )
    max_papers: int | None = Field(
        None, ge=1, description="papers used (default 30, at most 60); the rest are left out"
    )


def get_review_deps() -> ReviewDeps:
    """The real chat client, store and tracer (tests override this dependency)."""
    return ReviewDeps.default()


Deps = Annotated[ReviewDeps, Depends(get_review_deps)]


def _http_error(err: Exception) -> HTTPException | None:
    if isinstance(err, InvalidRequest):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(err))
    if isinstance(err, UnknownCollection):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(err))
    if isinstance(err, OpenAIKeyMissing):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(err))
    if isinstance(err, UpstreamError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, str(err))
    return None


@router.post("/library/review", response_model=ReviewOut)
async def write_review(body: ReviewRequest, deps: Deps) -> ReviewOut:
    """A cited related-work draft for the collection: every shown claim carries at least
    one citation that exists. A citation that resolves does not prove it supports the sentence."""
    try:
        return await run_in_threadpool(
            run_review, body.collection, body.focus, body.max_papers, deps=deps
        )
    except Exception as err:
        mapped = _http_error(err)
        if mapped is None:
            raise
        raise mapped from err

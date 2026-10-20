"""Upload a PDF that is not on arXiv, and delete such an upload (docs/adr/0011).

    POST   /library/upload?title=&collection=&year=&abstract=   body: the PDF (application/pdf)
    DELETE /library/uploads/{id}                                only `local-` ids

The body is the raw PDF, not a multipart form (so no multipart parser is needed). Everything
that can be refused is checked BEFORE parsing and before any paid call: the content type
(415), the title and year (422), the size (413, also from Content-Length before a byte is
read), an empty body (422), the `%PDF-` magic bytes (415), a missing OPENAI_API_KEY (503) and
another upload in progress (429). A PDF without a text layer (a scan) is 422: there is no OCR.
The upload costs the embeddings of its chunks (about $0.0002 per 10 pages) and nothing else.

The PDF is parsed in this process, in a worker thread, one at a time; it is never stored or
served. Uploading the same bytes again changes nothing and makes no embedding request.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db.session import session_scope
from app.features import library
from app.features.deps import upstream
from app.features.errors import UpstreamError
from app.ingest.embed import OpenAIKeyMissing
from app.ingest.upload import (
    ABSTRACT_MAX_CHARS,
    TITLE_MAX_CHARS,
    EmptyUpload,
    InvalidMetadata,
    NotAnUpload,
    NotAPdf,
    NoTextLayer,
    TooLarge,
    UploadBusy,
    UploadDeps,
    UploadError,
    UploadNotFound,
    check_metadata,
    delete_upload,
    ingest_pdf,
    upload_slot,
)

router = APIRouter()

ACCEPTED_TYPES = ("application/pdf", "application/octet-stream")


class UploadOut(BaseModel):
    arxiv_id: str  # local-<12 hex digits of the sha256>
    title: str
    year: int
    created: bool  # false: the same file was uploaded before, nothing changed
    collection: str | None
    abs_url: None = None  # an uploaded document has no arXiv page
    chunks: dict[str, int]
    embedded: int  # points embedded by this call (0 for a repeated upload)
    embedding_requests: int
    tokens_in: int
    cost_usd: float


class UploadDeleted(BaseModel):
    arxiv_id: str
    chunks: int
    points: int
    library_items: int


def get_upload_deps() -> UploadDeps:
    """The real embedder, Qdrant client and PDF parser (tests override this dependency)."""
    return UploadDeps.default()


Deps = Annotated[UploadDeps, Depends(get_upload_deps)]


def _http_error(err: Exception) -> HTTPException | None:
    if isinstance(err, TooLarge):
        return HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(err))
    if isinstance(err, NotAPdf):
        return HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(err))
    if isinstance(err, EmptyUpload | NoTextLayer | InvalidMetadata | library.InvalidInput):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(err))
    if isinstance(err, UploadError):
        return HTTPException(status.HTTP_409_CONFLICT, str(err))
    if isinstance(err, UploadBusy):
        return HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, str(err), headers={"Retry-After": "10"}
        )
    if isinstance(err, OpenAIKeyMissing):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(err))
    if isinstance(err, UpstreamError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, str(err))
    if isinstance(err, UploadNotFound):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(err))
    if isinstance(err, NotAnUpload):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(err))
    return None


async def read_body(request: Request, limit: int) -> bytes:
    """The raw request body, refused as soon as it is over `limit` bytes (a declared
    Content-Length over the limit is refused before a byte is read)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise TooLarge(f"the upload is {int(declared):,} bytes; the limit is {limit:,}")
    parts: list[bytes] = []
    total = 0
    async for part in request.stream():
        total += len(part)
        if total > limit:
            raise TooLarge(f"the upload is over the limit of {limit:,} bytes")
        parts.append(part)
    return b"".join(parts)


@router.post("/library/upload", response_model=UploadOut)
async def upload_pdf(
    request: Request,
    response: Response,
    deps: Deps,
    title: Annotated[str, Query(min_length=1, max_length=TITLE_MAX_CHARS)],
    collection: Annotated[str | None, Query(max_length=library.NAME_MAX_CHARS)] = None,
    year: Annotated[int | None, Query(ge=1900, le=2100)] = None,
    abstract: Annotated[str | None, Query(max_length=ABSTRACT_MAX_CHARS)] = None,
) -> UploadOut:
    """Add the PDF in the request body as an uploaded document (201 when new, 200 when the
    same file was uploaded before)."""
    try:
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in ACCEPTED_TYPES:
            raise NotAPdf("send the PDF as the request body with Content-Type: application/pdf")
        clean_title, clean_year, clean_abstract = check_metadata(title, year, abstract)
        target = library.clean_name(collection) if collection else None
        if get_settings().openai_api_key is None:
            raise OpenAIKeyMissing("OPENAI_API_KEY is not set (put it in .env)")
        with upload_slot():
            pdf = await read_body(request, get_settings().upload_max_bytes)

            def process() -> UploadOut:
                with session_scope() as session:
                    result = ingest_pdf(
                        session, pdf, title=clean_title, year=clean_year,
                        abstract=clean_abstract, deps=deps, collection=target,
                    )  # fmt: skip
                return UploadOut(
                    arxiv_id=result.arxiv_id,
                    title=result.title,
                    year=result.year,
                    created=result.created,
                    collection=target,
                    chunks=result.chunks,
                    embedded=result.embedded,
                    embedding_requests=result.spend.embedding_requests,
                    tokens_in=result.spend.tokens_in,
                    cost_usd=float(result.spend.cost_usd),
                )

            out = await run_in_threadpool(process)
    except Exception as err:
        mapped = _http_error(err) or _http_error(upstream(err))
        if mapped is None:
            raise
        raise mapped from err
    response.status_code = status.HTTP_201_CREATED if out.created else status.HTTP_200_OK
    return out


@router.delete("/library/uploads/{arxiv_id}", response_model=UploadDeleted)
async def delete_uploaded_document(arxiv_id: str, deps: Deps) -> UploadDeleted:
    """Remove an uploaded document: its Qdrant points, chunks, text and library items.
    Only `local-` ids; an arXiv paper cannot be deleted here."""

    def work() -> UploadDeleted:
        with session_scope() as session:
            result = delete_upload(
                session, arxiv_id, client=deps.qdrant(), collection=deps.qdrant_collection
            )
        return UploadDeleted(
            arxiv_id=result.arxiv_id,
            chunks=result.chunks,
            points=result.points,
            library_items=result.library_items,
        )

    try:
        return await run_in_threadpool(work)
    except Exception as err:
        mapped = _http_error(err) or _http_error(upstream(err))
        if mapped is None:
            raise
        raise mapped from err

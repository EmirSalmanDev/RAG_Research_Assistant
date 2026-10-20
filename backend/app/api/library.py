"""The library over HTTP: collections, saved papers with a reading status and notes, and the
BibTeX export (docs/adr/0011).

    GET    /library/collections                       every collection with its paper count
    POST   /library/collections                       create one (also an empty one)
    PUT    /library/collections/{name}                rename and/or change the description
    DELETE /library/collections/{name}                delete it and its items (never the papers)
    GET    /library/items?collection=                 the collection's papers
    POST   /library/items                             add a paper that is already in the database
    PATCH  /library/items/{arxiv_id}?collection=      status, notes, or move to another collection
    DELETE /library/items/{arxiv_id}?collection=      remove the paper from the collection
    GET    /library/bibtex?collection=                the .bib file (text/plain, attachment)

An item is addressed by the paper's id and its collection (the table's key: one item per
paper and collection). Nothing here calls OpenAI or costs anything; it all reads and writes
Postgres. Adding a paper that is not in the database is refused with 404 and a hint: that
is a later step (it needs the worker). Links go to the arXiv abstract page, never to a PDF;
an uploaded document has none (`abs_url` is null).

The review (POST /library/review) and the PDF upload live in their own modules.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Literal, get_args

from fastapi import APIRouter, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.db.session import session_scope
from app.features import library
from app.features.bibtex import build_entries, export_filename, render_bibtex
from app.features.library import (
    DESCRIPTION_MAX_CHARS,
    NAME_MAX_CHARS,
    NOTES_MAX_CHARS,
    CollectionExists,
    CollectionInfo,
    CollectionNotFound,
    InvalidInput,
    ItemExists,
    ItemNotFound,
    ItemView,
    LibraryError,
    PaperNotHeld,
)

router = APIRouter()

LibraryStatus = Literal["to_read", "reading", "read", "archived"]
LIBRARY_STATUS_VALUES: tuple[str, ...] = get_args(LibraryStatus)
REFERENCE_MAX_CHARS = 300

CollectionParam = Annotated[
    str, Query(min_length=1, max_length=NAME_MAX_CHARS, description="the collection's name")
]


# --- the shapes -------------------------------------------------------------------------------


class CollectionOut(BaseModel):
    name: str
    description: str | None
    paper_count: int
    created_at: datetime | None  # null: the collection exists only through its items ("Saved")


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=NAME_MAX_CHARS)
    description: str | None = Field(None, max_length=DESCRIPTION_MAX_CHARS)


class CollectionUpdate(BaseModel):
    """Send only what changes: `name` renames (the items move with it), `description` null
    clears the description."""

    name: str | None = Field(None, min_length=1, max_length=NAME_MAX_CHARS)
    description: str | None = Field(None, max_length=DESCRIPTION_MAX_CHARS)


class CollectionDeleted(BaseModel):
    name: str
    items_removed: int


class ItemOut(BaseModel):
    arxiv_id: str
    collection: str
    status: LibraryStatus
    notes: str | None
    added_at: datetime
    title: str
    authors: list[str]
    published_at: datetime
    evidence: Literal["full_text", "abstract_only", "uploaded"]
    abs_url: str | None  # null for an uploaded document


class ItemCreate(BaseModel):
    reference: str = Field(
        min_length=1,
        max_length=REFERENCE_MAX_CHARS,
        description="an arXiv id (also with a version) or an arxiv.org abs/pdf link",
    )
    collection: str = Field(min_length=1, max_length=NAME_MAX_CHARS)
    status: LibraryStatus = "to_read"


class ItemPatch(BaseModel):
    """Send only what changes. `notes` null or empty clears the note; `collection` moves
    the item to that collection (409 if the paper is in it already)."""

    status: LibraryStatus | None = None
    notes: str | None = Field(None, max_length=NOTES_MAX_CHARS)
    collection: str | None = Field(None, min_length=1, max_length=NAME_MAX_CHARS)


# --- helpers ----------------------------------------------------------------------------------


def _http_error(err: LibraryError) -> HTTPException:
    if isinstance(err, InvalidInput):
        return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(err))
    if isinstance(err, CollectionNotFound | ItemNotFound | PaperNotHeld):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(err))
    if isinstance(err, CollectionExists | ItemExists):
        return HTTPException(status.HTTP_409_CONFLICT, str(err))
    return HTTPException(status.HTTP_400_BAD_REQUEST, str(err))


async def _in_session[T](work: Callable[[Session], T]) -> T:
    """Run `work` in one transaction on a worker thread; a library error becomes its status."""

    def run() -> T:
        with session_scope() as session:
            return work(session)

    try:
        return await run_in_threadpool(run)
    except LibraryError as err:
        raise _http_error(err) from err


def collection_out(info: CollectionInfo) -> CollectionOut:
    return CollectionOut(
        name=info.name,
        description=info.description,
        paper_count=info.paper_count,
        created_at=info.created_at,
    )


def item_out(item: ItemView) -> ItemOut:
    return ItemOut(
        arxiv_id=item.arxiv_id,
        collection=item.collection,
        status=item.status,
        notes=item.notes,
        added_at=item.added_at,
        title=item.title,
        authors=item.authors,
        published_at=item.published_at,
        evidence=item.evidence,
        abs_url=item.abs_url,
    )


# --- collections ------------------------------------------------------------------------------


@router.get("/library/collections", response_model=list[CollectionOut])
async def list_collections() -> list[CollectionOut]:
    infos = await _in_session(library.list_collections)
    return [collection_out(i) for i in infos]


@router.post(
    "/library/collections", response_model=CollectionOut, status_code=status.HTTP_201_CREATED
)
async def create_collection(body: CollectionCreate) -> CollectionOut:
    info = await _in_session(lambda s: library.create_collection(s, body.name, body.description))
    return collection_out(info)


@router.put("/library/collections/{name:path}", response_model=CollectionOut)
async def update_collection(name: str, body: CollectionUpdate) -> CollectionOut:
    description = body.description if "description" in body.model_fields_set else library.UNSET
    info = await _in_session(
        lambda s: library.update_collection(s, name, new_name=body.name, description=description)
    )
    return collection_out(info)


@router.delete("/library/collections/{name:path}", response_model=CollectionDeleted)
async def delete_collection(name: str) -> CollectionDeleted:
    removed = await _in_session(lambda s: library.delete_collection(s, name))
    return CollectionDeleted(name=name, items_removed=removed)


# --- items ------------------------------------------------------------------------------------


@router.get("/library/items", response_model=list[ItemOut])
async def list_items(collection: CollectionParam) -> list[ItemOut]:
    items = await _in_session(lambda s: library.list_items(s, collection))
    return [item_out(i) for i in items]


@router.post("/library/items", response_model=ItemOut)
async def add_item(body: ItemCreate, response: Response) -> ItemOut:
    """201 when the paper was added, 200 when it was in the collection already."""
    item, created = await _in_session(
        lambda s: library.add_item(s, body.reference, body.collection, status=body.status)
    )
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return item_out(item)


@router.patch("/library/items/{arxiv_id:path}", response_model=ItemOut)
async def update_item(arxiv_id: str, collection: CollectionParam, body: ItemPatch) -> ItemOut:
    given = body.model_fields_set
    item = await _in_session(
        lambda s: library.update_item(
            s,
            arxiv_id,
            collection,
            status=body.status if "status" in given and body.status else library.UNSET,
            notes=body.notes if "notes" in given else library.UNSET,
            new_collection=(
                body.collection if "collection" in given and body.collection else library.UNSET
            ),
        )
    )
    return item_out(item)


@router.delete("/library/items/{arxiv_id:path}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_item(arxiv_id: str, collection: CollectionParam) -> Response:
    await _in_session(lambda s: library.remove_item(s, arxiv_id, collection))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- the .bib file ----------------------------------------------------------------------------


@router.get("/library/bibtex", response_class=Response)
async def export_bibtex(collection: CollectionParam) -> Response:
    """The collection's references as a .bib file, built from stored metadata (no model,
    no network). The keys are the ones the review's citations use."""

    def build(session: Session) -> str:
        return render_bibtex(build_entries(library.collection_bib_papers(session, collection)))

    text = await _in_session(build)
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{export_filename(collection)}"',
            "Cache-Control": "no-store",
        },
    )

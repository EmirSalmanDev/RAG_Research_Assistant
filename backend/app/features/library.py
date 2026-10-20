"""The library: named collections of saved papers, reading status and notes (docs/adr/0011).

Everything here reads and writes Postgres only (no Qdrant, no OpenAI, no cost). The
functions take a session and do not commit: the caller owns the transaction, so a rename
(which touches two tables) is one transaction.

A collection exists when it has a row in `library_collections` OR when some
`library_items` row names it. So the digest's "Saved" collection needs no row, an empty
collection can still be created, and deleting a collection removes its items but never the
papers. One item per (paper, collection): that is the primary key.
"""

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import LIBRARY_STATUSES, LibraryCollection, LibraryItem, Paper
from app.features.bibtex import BibPaper, author_names, order_papers
from app.features.paper_facts import has_section_chunks
from app.ids import abs_url, is_local_id
from app.sources.arxiv import validate_arxiv_id

NAME_MAX_CHARS = 100
DESCRIPTION_MAX_CHARS = 500
NOTES_MAX_CHARS = 10_000
DEFAULT_STATUS = "to_read"

Evidence = Literal["full_text", "abstract_only", "uploaded"]

_LOCAL_ID = re.compile(r"^local-[0-9a-f]{12}$")
_ARXIV_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf|html)/(?P<id>[^?#]+?)(?:\.pdf)?/?(?:[?#].*)?$",
    re.IGNORECASE,
)


class LibraryError(Exception):
    """Base class: the request cannot be done (the API turns each subclass into a status)."""


class InvalidInput(LibraryError, ValueError):
    """A name, status, note or paper reference that breaks a rule (HTTP 422)."""


class CollectionNotFound(LibraryError):
    """No collection with that name (HTTP 404)."""


class CollectionExists(LibraryError):
    """A collection with that name exists already (HTTP 409)."""


class ItemNotFound(LibraryError):
    """The paper is not in that collection (HTTP 404)."""


class ItemExists(LibraryError):
    """The paper is already in the collection it should move to (HTTP 409)."""


class PaperNotHeld(LibraryError):
    """The paper is not in the database: it has to be ingested first (HTTP 404)."""

    def __init__(self, arxiv_id: str) -> None:
        super().__init__(
            f"{arxiv_id} is not in the database yet. Ingest the paper first "
            "(adding a paper that is not in the database is a later step)."
        )
        self.arxiv_id = arxiv_id


def _rows_changed(result: Any) -> int:
    """How many rows an INSERT, UPDATE or DELETE statement changed."""
    return int(cast(CursorResult[Any], result).rowcount)


class Unset:
    """Marks an argument that was not given (None means "clear it")."""


UNSET = Unset()


# --- validation ---------------------------------------------------------------------------------


def clean_name(raw: str) -> str:
    """A collection name: surrounding space removed, 1 to 100 characters, no control characters."""
    name = " ".join(raw.split())
    if not name or len(name) > NAME_MAX_CHARS or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise InvalidInput(f"a collection name has 1 to {NAME_MAX_CHARS} characters")
    return name


def clean_description(raw: str | None) -> str | None:
    text = (raw or "").strip()
    if len(text) > DESCRIPTION_MAX_CHARS:
        raise InvalidInput(f"a description has at most {DESCRIPTION_MAX_CHARS} characters")
    return text or None


def clean_notes(raw: str | None) -> str | None:
    text = (raw or "").strip()
    if len(text) > NOTES_MAX_CHARS:
        raise InvalidInput(f"notes have at most {NOTES_MAX_CHARS} characters")
    return text or None


def clean_status(raw: str) -> str:
    if raw not in LIBRARY_STATUSES:
        raise InvalidInput("status must be one of: " + ", ".join(LIBRARY_STATUSES))
    return raw


def parse_reference(raw: str) -> str:
    """The paper id in what the user typed: an arXiv id (with or without a version), an
    arxiv.org abs/pdf link, "arXiv:<id>", or the id of an uploaded document."""
    text = raw.strip()
    if _LOCAL_ID.match(text):
        return text
    link = _ARXIV_URL.match(text)
    if link:
        text = link["id"]
    elif text.lower().startswith("arxiv:"):
        text = text[len("arxiv:") :]
    try:
        return validate_arxiv_id(text)
    except ValueError as err:
        raise InvalidInput(f"not an arXiv id or arxiv.org link: {raw.strip()[:80]!r}") from err


# --- collections --------------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionInfo:
    name: str
    description: str | None
    paper_count: int
    created_at: datetime | None  # None: the collection exists only through its items


def list_collections(session: Session) -> list[CollectionInfo]:
    counts = Counter(
        dict(
            session.execute(
                select(LibraryItem.collection, func.count()).group_by(LibraryItem.collection)
            ).all()
        )
    )
    rows = {c.name: c for c in session.scalars(select(LibraryCollection))}
    out = [
        CollectionInfo(
            name,
            rows[name].description if name in rows else None,
            counts.get(name, 0),
            rows[name].created_at if name in rows else None,
        )
        for name in set(rows) | set(counts)
    ]
    return sorted(out, key=lambda c: (c.name.casefold(), c.name))


def collection_exists(session: Session, name: str) -> bool:
    in_table = session.scalar(select(LibraryCollection.id).where(LibraryCollection.name == name))
    if in_table is not None:
        return True
    return (
        session.scalar(select(LibraryItem.arxiv_id).where(LibraryItem.collection == name).limit(1))
        is not None
    )


def require_collection(session: Session, name: str) -> None:
    if not collection_exists(session, name):
        raise CollectionNotFound(f"no collection named {name!r}")


def create_collection(
    session: Session, name: str, description: str | None = None
) -> CollectionInfo:
    name = clean_name(name)
    if collection_exists(session, name):
        raise CollectionExists(f"a collection named {name!r} exists already")
    row = LibraryCollection(name=name, description=clean_description(description))
    session.add(row)
    session.flush()
    session.refresh(row)
    return CollectionInfo(row.name, row.description, 0, row.created_at)


def update_collection(
    session: Session,
    name: str,
    *,
    new_name: str | None = None,
    description: str | None | Unset = UNSET,
) -> CollectionInfo:
    """Rename a collection and/or change its description. The rename moves the collection's
    items too, in the caller's one transaction."""
    require_collection(session, name)
    target = name if new_name is None else clean_name(new_name)
    if target != name and collection_exists(session, target):
        raise CollectionExists(f"a collection named {target!r} exists already")
    row = session.scalar(select(LibraryCollection).where(LibraryCollection.name == name))
    if row is None and not isinstance(description, Unset):
        row = LibraryCollection(name=name)  # an implicit collection gets its row
        session.add(row)
    if row is not None:
        if not isinstance(description, Unset):
            row.description = clean_description(description)
        row.name = target
    if target != name:
        session.execute(
            update(LibraryItem).where(LibraryItem.collection == name).values(collection=target)
        )
    session.flush()
    return next(c for c in list_collections(session) if c.name == target)


def delete_collection(session: Session, name: str) -> int:
    """Delete the collection and its items (not the papers). Returns how many items went."""
    require_collection(session, name)
    removed = _rows_changed(
        session.execute(delete(LibraryItem).where(LibraryItem.collection == name))
    )
    session.execute(delete(LibraryCollection).where(LibraryCollection.name == name))
    session.flush()
    return removed


# --- items --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ItemView:
    arxiv_id: str
    collection: str
    status: str
    notes: str | None
    added_at: datetime
    title: str
    authors: list[str]
    published_at: datetime
    evidence: Evidence
    abs_url: str | None  # None: an uploaded document


def _evidence(arxiv_id: str, has_fulltext: bool) -> Evidence:
    if is_local_id(arxiv_id):
        return "uploaded"
    return "full_text" if has_fulltext else "abstract_only"


def _views(session: Session, collection: str, arxiv_ids: list[str] | None = None) -> list[ItemView]:
    query = (
        select(LibraryItem, Paper, has_section_chunks().label("fulltext"))
        .join(Paper, Paper.arxiv_id == LibraryItem.arxiv_id)
        .where(LibraryItem.collection == collection)
        .order_by(LibraryItem.added_at, LibraryItem.arxiv_id)
    )
    if arxiv_ids is not None:
        query = query.where(LibraryItem.arxiv_id.in_(arxiv_ids))
    return [
        ItemView(
            arxiv_id=item.arxiv_id,
            collection=item.collection,
            status=item.status,
            notes=item.notes,
            added_at=item.added_at,
            title=paper.title,
            authors=author_names(paper.authors),
            published_at=paper.published_at,
            evidence=_evidence(item.arxiv_id, bool(fulltext)),
            abs_url=abs_url(item.arxiv_id),
        )
        for item, paper, fulltext in session.execute(query).all()
    ]


def list_items(session: Session, collection: str) -> list[ItemView]:
    require_collection(session, collection)
    return _views(session, collection)


def add_item(
    session: Session, reference: str, collection: str, *, status: str = DEFAULT_STATUS
) -> tuple[ItemView, bool]:
    """Put a paper that is already in the database into a collection (a collection that
    does not exist yet comes into being with its first item). Returns (item, created):
    adding it twice changes nothing."""
    arxiv_id = parse_reference(reference)
    name = clean_name(collection)
    status = clean_status(status)
    if session.get(Paper, arxiv_id) is None:
        raise PaperNotHeld(arxiv_id)
    inserted = session.execute(
        pg_insert(LibraryItem)
        .values(arxiv_id=arxiv_id, collection=name, status=status)
        .on_conflict_do_nothing()
        .returning(LibraryItem.arxiv_id)  # a row comes back only when one was inserted
    ).first()
    session.flush()
    return _views(session, name, [arxiv_id])[0], inserted is not None


def get_item(session: Session, arxiv_id: str, collection: str) -> ItemView:
    views = _views(session, collection, [arxiv_id])
    if not views:
        raise ItemNotFound(f"{arxiv_id} is not in {collection!r}")
    return views[0]


def update_item(
    session: Session,
    arxiv_id: str,
    collection: str,
    *,
    status: str | Unset = UNSET,
    notes: str | None | Unset = UNSET,
    new_collection: str | Unset = UNSET,
) -> ItemView:
    """Change an item's status or notes, or move it to another collection (which must not
    hold the paper yet)."""
    item = session.get(LibraryItem, (arxiv_id, collection))
    if item is None:
        raise ItemNotFound(f"{arxiv_id} is not in {collection!r}")
    if not isinstance(status, Unset):
        item.status = clean_status(status)
    if not isinstance(notes, Unset):
        item.notes = clean_notes(notes)
    target = collection
    if not isinstance(new_collection, Unset):
        target = clean_name(new_collection)
        if target != collection:
            if session.get(LibraryItem, (arxiv_id, target)) is not None:
                raise ItemExists(f"{arxiv_id} is in {target!r} already")
            session.flush()
            session.execute(
                update(LibraryItem)
                .where(LibraryItem.arxiv_id == arxiv_id, LibraryItem.collection == collection)
                .values(collection=target)
            )
            session.expunge(item)  # its key changed under it; get_item reads the new row
    session.flush()
    return get_item(session, arxiv_id, target)


def remove_item(session: Session, arxiv_id: str, collection: str) -> None:
    removed = _rows_changed(
        session.execute(
            delete(LibraryItem).where(
                LibraryItem.arxiv_id == arxiv_id, LibraryItem.collection == collection
            )
        )
    )
    if not removed:
        raise ItemNotFound(f"{arxiv_id} is not in {collection!r}")


# --- the papers of a collection (for the .bib file and the review) -----------------------------


def collection_papers(session: Session, collection: str) -> list[Paper]:
    """The collection's papers in the one stable order (oldest first, ties by id)."""
    require_collection(session, collection)
    papers = session.scalars(
        select(Paper)
        .join(LibraryItem, LibraryItem.arxiv_id == Paper.arxiv_id)
        .where(LibraryItem.collection == collection)
    ).all()
    return sorted(papers, key=lambda p: (p.published_at, p.arxiv_id))


def bib_paper(paper: Paper) -> BibPaper:
    return BibPaper(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=author_names(paper.authors),
        published_at=paper.published_at,
        primary_category=paper.primary_category,
    )


def collection_bib_papers(session: Session, collection: str) -> list[BibPaper]:
    return order_papers(bib_paper(p) for p in collection_papers(session, collection))

"""The daily digest over HTTP.

    GET  /digest?date=YYYY-MM-DD             the day's ranked papers (the latest available
                                             digest if there is none for that date)
    POST /digest/{arxiv_id}/save?date=...    Save: nudges the best interest up, adds the
                                             paper to the library collection "Saved"
    POST /digest/{arxiv_id}/dismiss?date=... Dismiss: nudges it down; never shown again
    POST /digest/{arxiv_id}/undo?date=...    back to new, reversing exactly the nudge

`GET /digest` only reads: it never builds a digest, scores anything or calls an API (the
08:00 Beat job and `make digest` build it). The three POSTs change the entry and one
interest weight in one transaction and are safe to repeat. Links go to the arXiv abstract
page, never to a PDF.
"""

from datetime import date, datetime
from typing import Annotated, Literal, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.api.papers import ABS_URL, _author_names
from app.db.session import session_scope
from app.features import digest

router = APIRouter()

AUTHORS_SHOWN = 3


class BestInterestOut(BaseModel):
    id: int
    name: str


class DigestEntryOut(BaseModel):
    arxiv_id: str
    rank: int
    score: float
    status: Literal["new", "saved", "dismissed"]
    title: str
    authors: list[str]  # the first three
    author_count: int
    categories: list[str]
    published_at: datetime
    hf_upvotes: int | None
    trending: bool
    best_interest: BestInterestOut | None
    summary: str  # the stored TL;DR, else the abstract's first sentence
    summary_source: Literal["analysis", "abstract"]
    abs_url: str


class DigestOut(BaseModel):
    requested_date: date
    digest_date: date | None  # None: no digest has been built yet
    is_fallback: bool  # the digest shown is not the one requested
    has_interests: bool  # False: the page asks for interests (scores are upvotes only)
    recent_dates: list[date]
    entries: list[DigestEntryOut]


def _entry_out(card: digest.DigestCard) -> DigestEntryOut:
    entry, paper = card.entry, card.paper
    names = _author_names(paper.authors)
    best = (
        BestInterestOut(id=entry.best_interest_id, name=card.best_interest_name)
        if entry.best_interest_id is not None and card.best_interest_name is not None
        else None
    )
    return DigestEntryOut(
        arxiv_id=paper.arxiv_id, rank=entry.rank, score=entry.score,
        status=entry.status,
        title=paper.title, authors=names[:AUTHORS_SHOWN], author_count=len(names),
        categories=list(paper.categories), published_at=paper.published_at,
        hf_upvotes=paper.hf_upvotes, trending=card.trending, best_interest=best,
        summary=card.summary, summary_source=card.summary_source,
        abs_url=ABS_URL.format(arxiv_id=paper.arxiv_id),
    )  # fmt: skip


class DigestStore(Protocol):
    def get(self, requested: date | None) -> DigestOut: ...

    def act(self, digest_date: date, arxiv_id: str, action: digest.Action) -> DigestEntryOut | None:
        """The updated entry; None when the digest has no such paper."""
        ...


class SqlDigestStore:
    def get(self, requested: date | None) -> DigestOut:
        with session_scope() as session:
            view = digest.load_digest(session, requested)
            shown = view.digest_date
            return DigestOut(
                requested_date=view.requested_date, digest_date=shown,
                is_fallback=shown is not None and shown != view.requested_date,
                has_interests=view.has_interests, recent_dates=view.recent_dates,
                entries=[_entry_out(c) for c in view.cards],
            )  # fmt: skip

    def act(self, digest_date: date, arxiv_id: str, action: digest.Action) -> DigestEntryOut | None:
        with session_scope() as session:
            try:
                digest.apply_action(session, digest_date, arxiv_id, action)
            except digest.EntryNotFoundError:
                return None
            card = digest.load_card(session, digest_date, arxiv_id)
            return _entry_out(card) if card is not None else None


def get_digest_store() -> DigestStore:
    """The real store (tests override this dependency)."""
    return SqlDigestStore()


Store = Annotated[DigestStore, Depends(get_digest_store)]


@router.get("/digest", response_model=DigestOut)
async def read_digest(
    store: Store, on: Annotated[date | None, Query(alias="date")] = None
) -> DigestOut:
    return await run_in_threadpool(store.get, on)


async def _act(
    store: DigestStore, arxiv_id: str, digest_date: date, action: digest.Action
) -> DigestEntryOut:
    result = await run_in_threadpool(store.act, digest_date, arxiv_id, action)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "that paper is not in this digest")
    return result


# `:path` because old arXiv ids contain a slash (hep-th/9901001).
DigestDate = Annotated[
    date, Query(alias="date", description="the digest's date (digest_date of GET /digest)")
]


@router.post("/digest/{arxiv_id:path}/save", response_model=DigestEntryOut)
async def save_entry(arxiv_id: str, store: Store, on: DigestDate) -> DigestEntryOut:
    return await _act(store, arxiv_id, on, "save")


@router.post("/digest/{arxiv_id:path}/dismiss", response_model=DigestEntryOut)
async def dismiss_entry(arxiv_id: str, store: Store, on: DigestDate) -> DigestEntryOut:
    return await _act(store, arxiv_id, on, "dismiss")


@router.post("/digest/{arxiv_id:path}/undo", response_model=DigestEntryOut)
async def undo_entry(arxiv_id: str, store: Store, on: DigestDate) -> DigestEntryOut:
    return await _act(store, arxiv_id, on, "undo")

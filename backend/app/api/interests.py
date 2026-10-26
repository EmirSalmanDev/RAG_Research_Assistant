"""Interests and settings: what the digest ranks against, and the page that edits it.

    GET    /interests          every interest (active or not)
    POST   /interests          create one            -> 201
    PUT    /interests/{id}     replace its editable fields
    DELETE /interests/{id}     remove it             -> 204
    GET    /settings           digest size, categories, full-text top N, budget cap and
                               the month-to-date spend
    PUT    /settings           save them (all four fields)

Nothing here calls OpenAI or any other paid API. An interest's embedding is made later
by the nightly scoring task (or `make digest`); `embedded` says whether the stored
vector matches the interest's current text. The budget cap is stored here; when the month's
spend reaches it, the background jobs that spend money pause (app/features/budget.py).
Interest text is the owner's own, but it ends up in an embedding request, so it is
length-limited like any other input.
"""

from datetime import datetime
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.db.models import Interest
from app.db.session import session_scope
from app.features import app_settings
from app.features import interests as repo

router = APIRouter()

NAME_MAX, DESCRIPTION_MAX, KEYWORD_MAX, KEYWORDS_MAX, CATEGORIES_MAX = 100, 1000, 60, 30, 30
CATEGORY_MAX = 32  # the longest real arXiv category id is about 16 characters
BUDGET_MAX = 100_000.0


def _check_categories(values: list[str]) -> list[str]:
    cleaned = [v.strip() for v in values]
    bad = [v for v in cleaned if len(v) > CATEGORY_MAX or not app_settings.is_valid_category(v)]
    if bad:
        raise ValueError(f"not an arXiv category (like cs.CL or hep-th): {bad[0]!r}")
    return list(dict.fromkeys(cleaned))


class InterestIn(BaseModel):
    name: str = Field(min_length=1, max_length=NAME_MAX)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    keywords: list[Annotated[str, Field(max_length=KEYWORD_MAX)]] = Field(
        default_factory=list, max_length=KEYWORDS_MAX
    )
    categories: list[str] = Field(default_factory=list, max_length=CATEGORIES_MAX)
    is_active: bool = True

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def _description(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @field_validator("categories")
    @classmethod
    def _categories(cls, value: list[str]) -> list[str]:
        return _check_categories(value)


class InterestOut(BaseModel):
    id: int
    name: str
    description: str | None
    keywords: list[str]
    categories: list[str]
    is_active: bool
    weight: float  # 1.0 at first; Save and Dismiss on the digest move it
    embedded: bool  # a vector for the current name, description and keywords exists
    created_at: datetime
    updated_at: datetime


class SettingsIn(BaseModel):
    digest_size: int = Field(ge=1, le=100)
    # An empty list means "use the default list"; a category added here is fetched from
    # then on, older papers are not backfilled.
    categories: list[str] = Field(default_factory=list, max_length=CATEGORIES_MAX)
    fulltext_top_n: int = Field(ge=0, le=100)  # 0 = only library and requested papers
    budget_cap_usd: float | None = Field(default=None, ge=0, le=BUDGET_MAX)

    @field_validator("categories")
    @classmethod
    def _categories(cls, value: list[str]) -> list[str]:
        return _check_categories(value)


class SettingsOut(BaseModel):
    digest_size: int
    categories: list[str]  # the list in use
    categories_default: list[str]  # what an empty list falls back to
    categories_overridden: bool
    fulltext_top_n: int
    budget_cap_usd: float | None  # background jobs that spend pause at the cap
    month_to_date_usd: float  # from llm_usage, month of the digest time zone


def _interest_out(row: Interest) -> InterestOut:
    return InterestOut(
        id=row.id, name=row.name, description=row.description, keywords=list(row.keywords),
        categories=list(row.categories), is_active=row.is_active, weight=row.weight,
        embedded=repo.has_current_embedding(row), created_at=row.created_at,
        updated_at=row.updated_at,
    )  # fmt: skip


class InterestStore(Protocol):
    def list(self) -> list[InterestOut]: ...
    def create(self, body: InterestIn) -> InterestOut:
        """Raises `DuplicateInterestError` when the name exists."""
        ...

    def update(self, interest_id: int, body: InterestIn) -> InterestOut | None: ...
    def delete(self, interest_id: int) -> bool: ...
    def get_settings(self) -> SettingsOut: ...
    def put_settings(self, body: SettingsIn) -> SettingsOut: ...


class SqlInterestStore:
    def list(self) -> list[InterestOut]:
        with session_scope() as session:
            return [_interest_out(r) for r in repo.list_interests(session)]

    def create(self, body: InterestIn) -> InterestOut:
        with session_scope() as session:
            row = repo.create_interest(
                session, name=body.name, description=body.description, keywords=body.keywords,
                categories=body.categories, is_active=body.is_active,
            )  # fmt: skip
            return _interest_out(row)

    def update(self, interest_id: int, body: InterestIn) -> InterestOut | None:
        with session_scope() as session:
            row = repo.update_interest(
                session, interest_id, name=body.name, description=body.description,
                keywords=body.keywords, categories=body.categories, is_active=body.is_active,
            )  # fmt: skip
            return _interest_out(row) if row is not None else None

    def delete(self, interest_id: int) -> bool:
        with session_scope() as session:
            return repo.delete_interest(session, interest_id)

    def get_settings(self) -> SettingsOut:
        with session_scope() as session:
            return _settings_out(session)

    def put_settings(self, body: SettingsIn) -> SettingsOut:
        with session_scope() as session:
            app_settings.save(
                session,
                {
                    app_settings.KEY_DIGEST_SIZE: body.digest_size,
                    app_settings.KEY_CATEGORIES: body.categories or None,  # empty: default
                    app_settings.KEY_TOP_N: body.fulltext_top_n,
                    app_settings.KEY_BUDGET_CAP: body.budget_cap_usd,
                },
            )
            session.flush()
            return _settings_out(session)


def _settings_out(session: Session) -> SettingsOut:
    effective = app_settings.load(session)
    return SettingsOut(
        digest_size=effective.digest_size,
        categories=effective.categories,
        categories_default=list(get_settings().arxiv_categories),
        categories_overridden=effective.categories_overridden,
        fulltext_top_n=effective.fulltext_top_n,
        budget_cap_usd=effective.budget_cap_usd,
        month_to_date_usd=float(app_settings.month_to_date_spend(session)),
    )


def get_interest_store() -> InterestStore:
    """The real store (tests override this dependency)."""
    return SqlInterestStore()


Store = Annotated[InterestStore, Depends(get_interest_store)]


@router.get("/interests", response_model=list[InterestOut])
async def list_interests(store: Store) -> list[InterestOut]:
    return await run_in_threadpool(store.list)


@router.post("/interests", response_model=InterestOut, status_code=status.HTTP_201_CREATED)
async def create_interest(body: InterestIn, store: Store) -> InterestOut:
    try:
        return await run_in_threadpool(store.create, body)
    except repo.DuplicateInterestError as err:
        raise HTTPException(status.HTTP_409_CONFLICT, "an interest with this name exists") from err


@router.put("/interests/{interest_id}", response_model=InterestOut)
async def update_interest(interest_id: int, body: InterestIn, store: Store) -> InterestOut:
    try:
        result = await run_in_threadpool(store.update, interest_id, body)
    except repo.DuplicateInterestError as err:
        raise HTTPException(status.HTTP_409_CONFLICT, "an interest with this name exists") from err
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown interest")
    return result


@router.delete("/interests/{interest_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_interest(interest_id: int, store: Store) -> None:
    if not await run_in_threadpool(store.delete, interest_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown interest")


@router.get("/settings", response_model=SettingsOut)
async def read_settings(store: Store) -> SettingsOut:
    return await run_in_threadpool(store.get_settings)


@router.put("/settings", response_model=SettingsOut)
async def save_settings(body: SettingsIn, store: Store) -> SettingsOut:
    return await run_in_threadpool(store.put_settings, body)

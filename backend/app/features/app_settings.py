"""The settings page's values (table `app_settings`: key -> JSON) over the config defaults.

A key that is not stored means "use the default from app/config.py", so a fresh database
needs no rows and removing a row restores the default. Keys:

    digest_size      papers in one day's digest
    categories       arXiv categories the daily fetch uses (replaces `arxiv_categories`;
                     adding one does not backfill older papers)
    fulltext_top_n   top papers by relevance that get full text each night
    budget_cap_usd   monthly spending cap; when this month's spend reaches it, the
                     background jobs that spend money pause (app/features/budget.py,
                     docs/adr/0013)

Nothing here calls an API.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import AppSetting, LLMUsage

# arXiv category ids: an archive ("hep-th", "cs", "physics") and, for most, a subject
# class after a dot ("cs.CL", "stat.ML", "astro-ph.GA", "physics.comp-ph").
CATEGORY_RE = re.compile(r"^[a-z]+(-[a-z]+)*(\.[A-Za-z]+(-[A-Za-z]+)*)?$")

KEY_DIGEST_SIZE = "digest_size"
KEY_CATEGORIES = "categories"
KEY_TOP_N = "fulltext_top_n"
KEY_BUDGET_CAP = "budget_cap_usd"


@dataclass(frozen=True)
class EffectiveSettings:
    digest_size: int
    categories: list[str]
    categories_overridden: bool
    fulltext_top_n: int
    budget_cap_usd: float | None


def is_valid_category(value: str) -> bool:
    return CATEGORY_RE.fullmatch(value) is not None


def _stored(session: Session) -> dict[str, Any]:
    return {row.key: row.value for row in session.scalars(select(AppSetting))}


def load(session: Session) -> EffectiveSettings:
    config = get_settings()
    stored = _stored(session)
    categories = stored.get(KEY_CATEGORIES)
    return EffectiveSettings(
        digest_size=int(stored.get(KEY_DIGEST_SIZE, config.digest_size)),
        categories=list(categories) if categories else list(config.arxiv_categories),
        categories_overridden=bool(categories),
        fulltext_top_n=int(stored.get(KEY_TOP_N, config.fulltext_top_n)),
        budget_cap_usd=stored.get(KEY_BUDGET_CAP),
    )


def save(session: Session, values: dict[str, Any]) -> None:
    """Store the given keys (a value of None deletes the key = back to the default).
    The caller commits."""
    for key, value in values.items():
        if value is None:
            session.execute(delete(AppSetting).where(AppSetting.key == key))
            continue
        stmt = pg_insert(AppSetting).values(key=key, value=value)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[AppSetting.key],
                set_={"value": stmt.excluded.value, "updated_at": func.now()},
            )
        )


def arxiv_categories(session: Session) -> list[str]:
    """The categories the arXiv fetch uses: the stored override, else the config list."""
    return load(session).categories


def month_start(now: datetime | None = None) -> datetime:
    """Midnight of the first day of the current month in the digest time zone, as UTC."""
    zone = ZoneInfo(get_settings().digest_timezone)
    local = (now or datetime.now(UTC)).astimezone(zone)
    return local.replace(day=1, hour=0, minute=0, second=0, microsecond=0).astimezone(UTC)


def month_to_date_spend(session: Session, now: datetime | None = None) -> Decimal:
    """USD spent since the start of the month, from `llm_usage` (every LLM and embedding
    call is logged there)."""
    total = session.scalar(
        select(func.coalesce(func.sum(LLMUsage.cost_usd), 0)).where(
            LLMUsage.created_at >= month_start(now)
        )
    )
    return Decimal(total or 0)

"""The monthly budget guard: background jobs that spend money stop when the cap is reached.

The cap is the `budget_cap_usd` setting (page /settings); the month's spend is read from
`llm_usage`, where every model call is logged (rule 9). Only the Celery tasks that spend
money in the background ask this module: the abstract and full-text embedding steps, the
embedding of edited interests, and the Batch API submit and collect steps. Everything else
keeps running (fetching papers is free) and so do your own requests (chat, search,
compare): the cap pauses the unattended spending, it does not lock you out of the app.

No cap stored means no guard. A cap of 0 pauses every guarded job. The check reads the
database at the moment the task runs, so raising the cap, or a new month, resumes the jobs
without a restart (docs/adr/0013).
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from app.features import app_settings

PAUSED_REASON = "budget cap reached"


@dataclass(frozen=True)
class BudgetState:
    cap_usd: Decimal | None
    spent_usd: Decimal
    month_start: datetime

    @property
    def paused(self) -> bool:
        return self.cap_usd is not None and self.spent_usd >= self.cap_usd


def budget_state(session: Session, now: datetime | None = None) -> BudgetState:
    now = now or datetime.now(UTC)
    cap = app_settings.load(session).budget_cap_usd
    return BudgetState(
        # The cap is stored as a JSON number; str() keeps 0.1 from becoming 0.1000000000000000055.
        cap_usd=None if cap is None else Decimal(str(cap)),
        spent_usd=app_settings.month_to_date_spend(session, now),
        month_start=app_settings.month_start(now),
    )


def is_paused(session: Session, now: datetime | None = None) -> bool:
    """True when background jobs that spend money must not start."""
    return budget_state(session, now).paused

"""The retry rule of the daily chain's two fetch steps, fetch_arxiv and fetch_hf_daily
(docs/adr/0003).

The other tasks retry after 0-1, 0-2 and 0-4 seconds (Celery's backoff, `RETRY_OPTIONS` in
tasks.py). For a source that is down, that uses every attempt within a few minutes, and the
rest of the chain is lost for the day (2026-10-28: arXiv failed from its first page on, and
all 4 attempts were used up in 4 min 46 s). These two steps wait minutes instead: with the
defaults, 2, 6 and 18 minutes, each +/-20 %.

`plan_attempt()` is called at the start of an attempt, before anything runs, so the attempt's
`ingest_runs` row can already say whether a failure will be retried and when.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings

# The start of a failed attempt's error when the step will be retried. The health page
# (app/api/status.py) and the daily catch-up (app/ingest/catchup.py) look for it, so it is
# written and read through this one constant.
RETRY_MARKER = "will retry in "


def is_retry_note(error: object) -> bool:
    """True for an `ingest_runs` error written by an attempt that will be retried."""
    return isinstance(error, str) and error.startswith(RETRY_MARKER)


def retry_delay_s(
    retries_done: int,
    settings: Settings,
    uniform: Callable[[float, float], float] = random.uniform,
) -> float:
    """Seconds to wait before the next attempt, when `retries_done` retries have already
    happened (0 for the first retry): first_delay * factor**retries_done minutes, times a
    random factor in [1 - jitter, 1 + jitter]."""
    base_s = (
        settings.daily_fetch_retry_first_delay_min
        * settings.daily_fetch_retry_factor**retries_done
        * 60
    )
    jitter = settings.daily_fetch_retry_jitter
    return base_s * uniform(1 - jitter, 1 + jitter)


@dataclass(frozen=True)
class AttemptPlan:
    attempt: int  # 1-based number of this attempt
    attempts: int  # all attempts, the first one included (retries + 1)
    delay_s: float | None  # wait before the next attempt; None = this is the last attempt

    @property
    def note(self) -> str:
        """Put in front of the error of a failed attempt (ingest_runs, the health page)."""
        if self.delay_s is None:
            return f"last attempt ({self.attempt} of {self.attempts}), no retry"
        minutes = max(1, round(self.delay_s / 60))
        return f"{RETRY_MARKER}{minutes} min (attempt {self.attempt} of {self.attempts})"

    def error_note(self, err: Exception) -> str | None:
        """The note for `err`. A ValueError is bad input that is not retried, so it gets none."""
        return None if isinstance(err, ValueError) else self.note


def plan_attempt(
    retries_done: int,
    settings: Settings,
    uniform: Callable[[float, float], float] = random.uniform,
) -> AttemptPlan:
    """What happens if the attempt after `retries_done` retries fails. The delay is drawn
    now, so the note written on the run is the delay that is actually used."""
    attempts = settings.daily_fetch_retries + 1
    attempt = retries_done + 1
    delay = retry_delay_s(retries_done, settings, uniform) if attempt < attempts else None
    return AttemptPlan(attempt=attempt, attempts=attempts, delay_s=delay)

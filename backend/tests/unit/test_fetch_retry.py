"""Offline unit tests of the daily fetch steps' retry rule (app/ingest/retry.py): the delay
schedule and its jitter bounds, the visibility-timeout limit, and the notes on failed runs."""

import random

import pytest
from pydantic import ValidationError

from app.config import BROKER_VISIBILITY_TIMEOUT_S, Settings
from app.ingest import retry
from app.worker.celery_app import celery_app

DEFAULTS = Settings(_env_file=None)


def _low(a: float, b: float) -> float:
    return a


def _high(a: float, b: float) -> float:
    return b


def test_the_defaults_are_three_retries_after_2_6_and_18_minutes() -> None:
    assert DEFAULTS.daily_fetch_retries == 3
    middle = [retry.retry_delay_s(n, DEFAULTS, lambda a, b: (a + b) / 2) for n in range(3)]
    assert middle == pytest.approx([120.0, 360.0, 1080.0])


def test_the_jitter_stays_within_plus_minus_20_percent() -> None:
    lows = [retry.retry_delay_s(n, DEFAULTS, _low) for n in range(3)]
    highs = [retry.retry_delay_s(n, DEFAULTS, _high) for n in range(3)]
    assert lows == pytest.approx([96.0, 288.0, 864.0])
    assert highs == pytest.approx([144.0, 432.0, 1296.0])
    rng = random.Random(7)
    for n in range(3):
        drawn = [retry.retry_delay_s(n, DEFAULTS, rng.uniform) for _ in range(2000)]
        assert lows[n] <= min(drawn) and max(drawn) <= highs[n]
        # the draws really spread over the range (not a constant factor)
        assert max(drawn) - min(drawn) > 0.3 * (highs[n] - lows[n])


def test_every_delay_stays_below_the_broker_visibility_timeout() -> None:
    """A retry waiting for its countdown is an unacknowledged message: past the visibility
    timeout Redis would deliver it again, and the step would run twice."""
    configured = celery_app.conf.broker_transport_options["visibility_timeout"]
    assert configured == BROKER_VISIBILITY_TIMEOUT_S == 3600
    longest = max(retry.retry_delay_s(n, DEFAULTS, _high) for n in range(3))
    assert longest == pytest.approx(1296.0)  # 18 min x 1.2 = 21.6 min
    assert longest < configured


def test_settings_whose_longest_delay_reaches_the_timeout_are_refused() -> None:
    # 2 x 3^3 x 1.2 = 64.8 min >= 60 min
    with pytest.raises(ValidationError, match="visibility timeout"):
        Settings(_env_file=None, daily_fetch_retries=4)
    # 50 x 1.2 = 60 min exactly: refused too (it must stay below)
    with pytest.raises(ValidationError, match="visibility timeout"):
        Settings(_env_file=None, daily_fetch_retries=1, daily_fetch_retry_first_delay_min=50)
    ok = Settings(_env_file=None, daily_fetch_retries=1, daily_fetch_retry_first_delay_min=49)
    assert ok.daily_fetch_retries == 1
    assert Settings(_env_file=None, daily_fetch_retries=0).daily_fetch_retries == 0


def test_plan_attempt_says_when_the_next_attempt_comes_and_which_is_the_last() -> None:
    plans = [retry.plan_attempt(n, DEFAULTS, lambda a, b: 1.0) for n in range(4)]
    assert [(p.attempt, p.attempts) for p in plans] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert [p.delay_s for p in plans] == [120.0, 360.0, 1080.0, None]
    assert [p.note for p in plans] == [
        "will retry in 2 min (attempt 1 of 4)",
        "will retry in 6 min (attempt 2 of 4)",
        "will retry in 18 min (attempt 3 of 4)",
        "last attempt (4 of 4), no retry",
    ]


def test_the_note_shows_the_drawn_delay_in_whole_minutes() -> None:
    assert retry.plan_attempt(0, DEFAULTS, _low).note == "will retry in 2 min (attempt 1 of 4)"
    assert retry.plan_attempt(2, DEFAULTS, _high).note == "will retry in 22 min (attempt 3 of 4)"
    tiny = Settings(_env_file=None, daily_fetch_retry_first_delay_min=0.1)
    assert retry.plan_attempt(0, tiny, _low).note.startswith("will retry in 1 min")


def test_a_value_error_gets_no_note_because_it_is_not_retried() -> None:
    plan = retry.plan_attempt(0, DEFAULTS, _low)
    assert plan.error_note(ValueError("bad id")) is None
    assert plan.error_note(RuntimeError("down")) == plan.note


def test_without_retries_the_first_attempt_is_the_last() -> None:
    none = Settings(_env_file=None, daily_fetch_retries=0)
    plan = retry.plan_attempt(0, none)
    assert plan.delay_s is None and plan.note == "last attempt (1 of 1), no retry"


def test_the_retry_note_starts_with_the_marker_the_health_page_reads() -> None:
    plans = [retry.plan_attempt(n, DEFAULTS, _low) for n in range(4)]
    assert [retry.is_retry_note(p.note + ": RetryableError: x") for p in plans] == [
        True, True, True, False,
    ]  # fmt: skip
    assert all(p.note.startswith(retry.RETRY_MARKER) for p in plans[:3])
    assert not retry.is_retry_note(None) and not retry.is_retry_note("RuntimeError: will retry in")

"""The catch-up's settings, its Beat entries and the "today's 07:00" it looks back to
(app/config.py, app/worker/schedule.py, app/ingest/catchup.py). The decision itself is
tested on a database in tests/integration/test_daily_catchup.py."""

from collections.abc import Iterator
from datetime import UTC, datetime, time

import pytest
from celery.schedules import crontab
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.ingest import catchup
from app.ingest import tasks as ingest_tasks
from app.worker.celery_app import celery_app
from app.worker.schedule import beat_schedule


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_times_are_13_and_19_istanbul() -> None:
    assert Settings(_env_file=None).daily_catchup_at == [time(13, 0), time(19, 0)]


@pytest.mark.parametrize(
    ("value", "times"),
    [
        ("", []),
        ("   ", []),
        ("10:30", [time(10, 30)]),
        (" 19:00 , 13:05 ", [time(19, 0), time(13, 5)]),
        ("07:01,23:59", [time(7, 1), time(23, 59)]),
    ],
)
def test_valid_times(value: str, times: list[time]) -> None:
    assert Settings(_env_file=None, daily_catchup_times=value).daily_catchup_at == times


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("25:00", "is not HH:MM"),
        ("13:60", "is not HH:MM"),
        ("1300", "is not HH:MM"),
        ("1:00", "is not HH:MM"),
        ("13:00:00", "is not HH:MM"),
        ("13:00,", "is not HH:MM"),
        ("13:00;19:00", "is not HH:MM"),
        ("noon", "is not HH:MM"),
        ("06:00", "later than the daily run at 07:00"),
        ("07:00", "later than the daily run at 07:00"),
        ("13:00,13:00", "repeat"),
    ],
)
def test_bad_times_are_refused(value: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None, daily_catchup_times=value)


def test_a_bad_time_in_the_environment_stops_the_start(
    monkeypatch: pytest.MonkeyPatch, fresh_settings: None
) -> None:
    # The API, the worker and Beat all read get_settings() when they start.
    monkeypatch.setenv("DAILY_CATCHUP_TIMES", "13:00,25:00")
    with pytest.raises(ValidationError, match="'25:00' is not HH:MM"):
        get_settings()


def test_beat_has_one_catchup_entry_per_time_in_istanbul_time() -> None:
    schedule = celery_app.conf.beat_schedule  # built from the default settings
    assert celery_app.conf.timezone == "Europe/Istanbul"
    catchups = {k: v for k, v in schedule.items() if v["task"] == "ingest.daily_catchup"}
    assert catchups == {
        f"daily-catchup-{hour}00": {
            "task": "ingest.daily_catchup",
            "schedule": crontab(hour=hour, minute=0),
        }
        for hour in (13, 19)
    }
    assert "ingest.daily_catchup" in celery_app.tasks
    assert ingest_tasks.daily_catchup.name == "ingest.daily_catchup"
    # 07:00 and the 08:00 digest are unchanged
    assert schedule["daily-ingest"]["schedule"] == crontab(hour=7, minute=0)
    assert schedule["digest-build"]["schedule"] == crontab(hour=8, minute=0)


def test_no_catchup_entry_when_the_setting_is_empty() -> None:
    schedule = beat_schedule(Settings(_env_file=None, daily_catchup_times=""))
    assert set(schedule) == {"daily-ingest", "digest-build", "analysis-submit", "analysis-collect"}
    assert set(beat_schedule(Settings(_env_file=None, daily_catchup_times="10:30"))) - set(
        schedule
    ) == {"daily-catchup-1030"}


@pytest.mark.parametrize(
    ("now", "since"),
    [
        # 13:00 and 19:00 Istanbul (UTC+3): today's 07:00 = 04:00 UTC
        (datetime(2026, 10, 29, 10, 0, tzinfo=UTC), datetime(2026, 10, 29, 4, 0, tzinfo=UTC)),
        (datetime(2026, 10, 29, 16, 0, tzinfo=UTC), datetime(2026, 10, 29, 4, 0, tzinfo=UTC)),
        # exactly 07:00 is its own start
        (datetime(2026, 10, 29, 4, 0, tzinfo=UTC), datetime(2026, 10, 29, 4, 0, tzinfo=UTC)),
        # 02:00 Istanbul (a late send after midnight) and 06:59: yesterday's 07:00
        (datetime(2026, 10, 28, 23, 0, tzinfo=UTC), datetime(2026, 10, 28, 4, 0, tzinfo=UTC)),
        (datetime(2026, 10, 29, 3, 59, tzinfo=UTC), datetime(2026, 10, 28, 4, 0, tzinfo=UTC)),
        # 21:30 UTC is already the next day in Istanbul (00:30)
        (datetime(2026, 10, 29, 21, 30, tzinfo=UTC), datetime(2026, 10, 29, 4, 0, tzinfo=UTC)),
    ],
)
def test_latest_daily_start(now: datetime, since: datetime) -> None:
    assert catchup.latest_daily_start(now) == since

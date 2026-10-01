"""Celery Beat schedule: the daily ingest chain at 07:00 Istanbul time.

The time zone is set on the Celery app (`timezone="Europe/Istanbul"`), so the crontab
below is read in Istanbul time. Beat does not catch up a missed run (e.g. the Mac was
asleep at 07:00); the next run's 4-day window overlap picks up what was missed.
"""

from celery.schedules import crontab

BEAT_SCHEDULE = {
    "daily-ingest": {
        "task": "ingest.daily_pipeline",
        "schedule": crontab(hour=7, minute=0),
    },
}

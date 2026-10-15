"""Celery Beat schedule: the daily ingest chain at 07:00 Istanbul time, the daily
paper-analysis submission at 09:00 (it only submits when `analysis_auto_submit` is on)
and the analysis collector every 30 minutes (no API call while no batch is open).

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
    "analysis-submit": {
        "task": "analysis.submit",
        "schedule": crontab(hour=9, minute=0),  # after the full-text chains of 07:00
    },
    "analysis-collect": {
        "task": "analysis.collect",
        "schedule": crontab(minute="*/30"),
    },
}

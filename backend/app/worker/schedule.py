"""Celery Beat schedule: the daily ingest chain at 07:00 Istanbul time (its steps include
the relevance scoring and the choice of papers for full text), the digest build at 08:00,
the daily paper-analysis submission at 09:00 (it only submits when `analysis_auto_submit` is on)
and the analysis collector every 30 minutes (no API call while no batch is open).

The time zone is set on the Celery app (`timezone="Europe/Istanbul"`), so the crontab
below is read in Istanbul time. Beat keeps each entry's last run in a file inside its
container and sends a run that came due while it was stopped as soon as it starts (seen on
2026-10-02 and 2026-10-04). So a Beat start can begin a daily run at once, which is why
Beat is in the compose profile "scheduler" and not started by `make up` (docs/adr/0013).
"""

from celery.schedules import crontab

BEAT_SCHEDULE = {
    "daily-ingest": {
        "task": "ingest.daily_pipeline",
        "schedule": crontab(hour=7, minute=0),
    },
    "digest-build": {
        "task": "digest.build",
        "schedule": crontab(hour=8, minute=0),  # ready when you open the app
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

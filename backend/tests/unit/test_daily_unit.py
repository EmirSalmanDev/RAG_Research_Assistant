"""Offline unit tests: the arXiv window query and its paging stop, the window arithmetic,
the Celery chains (order, immutable signatures), the Beat schedule and Celery settings,
the failure hook, and the backfill CLI guards."""

from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from celery.schedules import crontab

from app.features import tasks as digest_tasks
from app.ingest import cli, daily, tasks
from app.sources.arxiv import ArxivClient
from app.worker.celery_app import celery_app
from tests.helpers import CountingLimiter
from tests.unit.test_arxiv import FakeResponse, FakeSession

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _feed(updated: list[datetime], total: int) -> bytes:
    entries = "".join(
        f"""<entry><id>http://arxiv.org/abs/2609.{n:05d}v1</id>
  <updated>{u:%Y-%m-%dT%H:%M:%SZ}</updated><published>{u:%Y-%m-%dT%H:%M:%SZ}</published>
  <title>Invented {n}</title><summary>Invented abstract {n}.</summary>
  <author><name>Ada Example</name></author>
  <arxiv:primary_category term="cs.CL"/><category term="cs.CL"/></entry>"""
        for n, u in enumerate(updated, 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>""".encode()


def _hours_ago(*hours: int) -> list[datetime]:
    return [NOW - timedelta(hours=h) for h in hours]


# --- updated_papers: the window applied client-side -------------------------------------------


def test_updated_papers_stops_paging_at_the_window_start() -> None:
    since = NOW - timedelta(hours=30)
    session = FakeSession(
        [
            FakeResponse(200, _feed(_hours_ago(1, 2), 1000)),
            FakeResponse(200, _feed(_hours_ago(20, 40), 1000)),  # 40 h: before the window
        ]
    )
    client = ArxivClient(limiter=CountingLimiter(), session=session, sleep=lambda s: None)
    papers = client.updated_papers(["cs.CL", "cs.LG"], since, NOW, page_size=2)
    assert [p.updated_at for p in papers] == _hours_ago(1, 2, 20)
    assert len(session.urls) == 2  # no third page: the window ended inside page two
    query = parse_qs(urlparse(session.urls[0]).query)
    assert query["search_query"] == ["(cat:cs.CL OR cat:cs.LG)"]
    assert query["sortBy"] == ["lastUpdatedDate"] and query["sortOrder"] == ["descending"]


def test_updated_papers_skips_entries_after_until_and_respects_the_cap() -> None:
    session = FakeSession([FakeResponse(200, _feed(_hours_ago(-5, 1, 2, 3), 1000))])
    client = ArxivClient(limiter=CountingLimiter(), session=session, sleep=lambda s: None)
    papers = client.updated_papers(
        ["cs.CL"], NOW - timedelta(days=1), NOW, page_size=4, max_papers=2
    )
    assert [p.updated_at for p in papers] == _hours_ago(1, 2)  # future entry skipped, cap 2


# --- window arithmetic ------------------------------------------------------------------------


def test_daily_window_has_a_four_day_overlap() -> None:
    cursor = NOW - timedelta(days=1)
    assert daily.daily_window(cursor, NOW) == (cursor - timedelta(days=4), NOW)


def test_first_run_without_cursor_looks_back_one_day_plus_overlap() -> None:
    assert daily.daily_window(None, NOW) == (NOW - timedelta(days=5), NOW)


def test_daily_window_never_starts_more_than_max_days_back() -> None:
    old_cursor = NOW - timedelta(days=24)
    assert daily.daily_window(old_cursor, NOW, max_days=7) == (NOW - timedelta(days=7), NOW)
    recent = NOW - timedelta(days=2)  # cursor - 4 days = 6 days back: inside the cap
    assert daily.daily_window(recent, NOW, max_days=7) == (recent - timedelta(days=4), NOW)


def test_backfill_plan() -> None:
    plan = daily.plan_backfill(7, NOW)
    assert (plan.since, plan.until) == (NOW - timedelta(days=7), NOW)


# --- chains, schedule, settings ---------------------------------------------------------------


def _steps(signature: Any) -> list[tuple[str, tuple[Any, ...], bool]]:
    return [(s.task, tuple(s.args), s.immutable) for s in signature.tasks]


def test_daily_chain_order_with_immutable_signatures() -> None:
    assert _steps(tasks.daily_chain()) == [
        ("ingest.fetch_arxiv", (), True),
        ("ingest.fetch_hf_daily", (), True),
        ("ingest.index_abstracts", (), True),
        ("ingest.enrich_new", (), True),
        ("digest.score_relevance", (), True),  # score before the top-N selection reads scores
        ("ingest.select_fulltext", (), True),
    ]


def test_paper_chain_order_passes_only_the_id() -> None:
    assert _steps(tasks.paper_chain("2601.00001")) == [
        ("ingest.fetch_fulltext", ("2601.00001",), True),
        ("ingest.clean_paper", ("2601.00001",), True),
        ("ingest.chunk_paper", ("2601.00001",), True),
        ("ingest.embed_paper", ("2601.00001",), True),
    ]


def test_beat_runs_the_daily_pipeline_at_7_istanbul_time() -> None:
    entry = celery_app.conf.beat_schedule["daily-ingest"]
    assert entry["task"] == "ingest.daily_pipeline"
    assert entry["schedule"] == crontab(hour=7, minute=0)
    assert celery_app.conf.timezone == "Europe/Istanbul"


def test_beat_builds_the_digest_at_8_istanbul_time_after_the_7_o_clock_chain() -> None:
    schedule = celery_app.conf.beat_schedule
    assert schedule["digest-build"]["task"] == "digest.build"
    assert schedule["digest-build"]["schedule"] == crontab(hour=8, minute=0)
    assert digest_tasks.build_digest.name == "digest.build"  # importing registers the tasks
    assert {"digest.build", "digest.score_relevance"} <= set(celery_app.tasks)
    # the entry names a registered task, and its time is read in Istanbul time
    assert celery_app.conf.timezone == "Europe/Istanbul"
    assert schedule["daily-ingest"]["schedule"] == crontab(hour=7, minute=0)


def test_beat_submits_analyses_daily_and_collects_every_30_minutes() -> None:
    schedule = celery_app.conf.beat_schedule
    assert schedule["analysis-submit"]["task"] == "analysis.submit"
    assert schedule["analysis-submit"]["schedule"] == crontab(hour=9, minute=0)
    assert schedule["analysis-collect"]["task"] == "analysis.collect"
    assert schedule["analysis-collect"]["schedule"] == crontab(minute="*/30")
    assert {"analysis.submit", "analysis.collect"} <= set(celery_app.tasks)


def test_celery_settings_for_one_small_worker() -> None:
    conf = celery_app.conf
    assert conf.worker_concurrency == 1 and conf.worker_prefetch_multiplier == 1
    assert conf.task_acks_late is True
    names = set(celery_app.tasks)
    for name in ("fetch_arxiv", "fetch_hf_daily", "index_abstracts", "enrich_new",
                 "select_fulltext", "fetch_fulltext", "clean_paper", "chunk_paper",
                 "embed_paper", "paper", "daily_pipeline"):  # fmt: skip
        assert f"ingest.{name}" in names


@pytest.mark.parametrize("task", [tasks.fetch_arxiv, tasks.fetch_fulltext, tasks.embed_paper])
def test_tasks_retry_three_times_with_backoff(task: Any) -> None:
    assert task.max_retries == 3 and task.retry_backoff is True
    assert task.autoretry_for == (Exception,) and task.dont_autoretry_for == (ValueError,)


def test_last_failure_marks_the_paper_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    marked: list[str] = []
    monkeypatch.setattr(tasks, "mark_failed", marked.append)
    tasks.fetch_fulltext.on_failure(RuntimeError("down"), "task-id", ("2601.00001",), {}, None)
    tasks.embed_paper.on_failure(
        RuntimeError("down"), "task-id", (), {"arxiv_id": "2601.00002"}, None
    )
    assert marked == ["2601.00001", "2601.00002"]


# --- backfill CLI guards ----------------------------------------------------------------------


def test_backfill_refuses_more_than_7_days_without_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_backfill(days: int, max_papers: int) -> dict[str, int]:
        calls.append((days, max_papers))
        return {}

    monkeypatch.setattr(cli, "_backfill", fake_backfill)
    monkeypatch.setattr(cli, "close_qdrant", lambda: None)
    with pytest.raises(SystemExit):
        cli.main(["backfill", "--days", "8"])
    with pytest.raises(SystemExit):
        cli.main(["backfill", "--days", "0"])
    assert cli.main(["backfill", "--days", "7"]) == 0
    assert cli.main(["backfill", "--days", "30", "--confirm", "--max-papers", "50"]) == 0
    assert calls == [(7, 20_000), (30, 50)]


def test_embedding_request_estimate() -> None:
    assert cli.embedding_requests(0, 100) == 0
    assert cli.embedding_requests(256, 100) == 3  # 100 + 100 + 56
    assert cli.embedding_requests(300, 100) == 3
    assert cli.embedding_requests(5674, 100) == 57

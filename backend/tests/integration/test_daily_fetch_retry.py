"""The daily chain's fetch steps under the minutes-scale retry rule (app/ingest/retry.py),
run eagerly (in this process, no broker) on a scratch Postgres.

Faked: arXiv and Hugging Face (they fail on purpose, then answer), the abstract indexing,
the enrichment, the interest embedding and the Qdrant reads. Real: the Celery chain and its
retries, the arXiv window and cursor, the `ingest_runs` rows and the /status reading of them.
In eager mode a retry runs at once; the countdown it asked for is recorded by a spy, so the
tests see the delay the worker would wait."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api import status
from app.config import get_settings
from app.db.models import IngestRun, Paper
from app.features import tasks as digest_tasks
from app.ingest import daily
from app.ingest import tasks as ingest_tasks
from app.sources.arxiv import ArxivPaper
from app.sources.http import RetryableError
from tests.digest_fakes import FakeDense, FakeVectors

pytestmark = pytest.mark.integration

TIMEOUT = "network error: HTTPSConnectionPool(host='export.arxiv.org', port=443): Read timed out."
HTTP_429 = "Page request resulted in HTTP 429 (https://export.arxiv.org/api/query?start=0)"
OLD_CURSOR = datetime(2026, 10, 27, 18, 25, 42, tzinfo=UTC)


class FakeArxiv:
    """Raises the queued errors on `updated_papers`, one per call, then answers."""

    def __init__(self, errors: Sequence[Exception], paper: ArxivPaper) -> None:
        self.errors = list(errors)
        self.paper = paper
        self.calls = 0

    def updated_papers(self, categories: Any, since: datetime, *a: Any, **k: Any) -> list[Any]:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return [self.paper]

    def get_papers(self, arxiv_ids: Any) -> list[ArxivPaper]:
        return []


class FakeHF:
    def __init__(self, errors: Sequence[Exception] = ()) -> None:
        self.errors = list(errors)
        self.calls = 0

    def daily_papers(self, day: Any) -> list[Any]:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return []


class NoChain:
    """Stands in for a paper's full-text chain: select_fulltext starts it, nothing runs."""

    def apply_async(self) -> None:
        return None


class Night:
    """Everything one test needs: the fakes, the countdowns asked for, the later steps."""

    def __init__(self) -> None:
        self.countdowns: dict[str, list[float]] = {"arxiv": [], "hf": []}
        self.later_steps: list[str] = []
        self.arxiv: FakeArxiv | None = None
        self.hf = FakeHF()


def _paper() -> ArxivPaper:
    now = datetime.now(UTC)
    return ArxivPaper(
        arxiv_id="2610.90001", version=1, title="An invented paper", abstract="Invented.",
        authors=["Ada Example"], categories=["cs.CL"], primary_category="cs.CL",
        published_at=now - timedelta(hours=2), updated_at=now - timedelta(hours=1),
    )  # fmt: skip


@pytest.fixture
def night(migrated_session: Session, monkeypatch: pytest.MonkeyPatch) -> Night:
    import app.ingest.enrich as enrich
    import app.ingest.index as index
    import app.sources.hf as hf

    n = Night()
    # No jitter here, so the notes and countdowns are exact (the jitter has unit tests).
    monkeypatch.setenv("DAILY_FETCH_RETRY_JITTER", "0")
    get_settings.cache_clear()
    monkeypatch.setattr(ingest_tasks, "_arxiv", lambda: n.arxiv)
    monkeypatch.setattr(hf, "HFClient", lambda: n.hf)
    for name, task in (("arxiv", ingest_tasks.fetch_arxiv), ("hf", ingest_tasks.fetch_hf_daily)):
        original = task.retry

        def spy(*args: Any, _name: str = name, _orig: Any = original, **kwargs: Any) -> Any:
            n.countdowns[_name].append(kwargs["countdown"])
            return _orig(*args, **kwargs)

        monkeypatch.setattr(task, "retry", spy)

    def later(name: str) -> Any:
        def fake(*args: Any, **kwargs: Any) -> Any:
            n.later_steps.append(name)
            return 0 if name == "index" else {}

        return fake

    monkeypatch.setattr(ingest_tasks, "_embedders", lambda: {})
    monkeypatch.setattr(index, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(index, "index_abstracts", later("index"))
    monkeypatch.setattr(enrich, "default_enrich_clients", lambda: None)
    monkeypatch.setattr(enrich, "enrich_daily", later("enrich"))
    monkeypatch.setattr(ingest_tasks, "paper_chain", lambda arxiv_id: NoChain())
    monkeypatch.setattr(digest_tasks, "_dense", lambda: FakeDense())
    monkeypatch.setattr(digest_tasks, "_vectors", lambda: FakeVectors())

    # The last successful arXiv run, as on the real store before 2026-10-28.
    s = migrated_session
    s.add(IngestRun(source="arxiv-daily", counts={"cursor": OLD_CURSOR.isoformat()}, errors={},
                    finished_at=OLD_CURSOR))  # fmt: skip
    s.commit()
    return n


def _runs(s: Session, source: str) -> list[IngestRun]:
    s.expire_all()
    query = select(IngestRun).where(IngestRun.source == source).order_by(IngestRun.id)
    return list(s.scalars(query))


def _error(run: IngestRun) -> str:
    return str(run.errors.get("error", ""))


def test_arxiv_fails_twice_then_succeeds_and_the_chain_goes_on(
    migrated_session: Session, night: Night
) -> None:
    s = migrated_session
    night.arxiv = arxiv = FakeArxiv(
        [RetryableError(TIMEOUT), RetryableError(HTTP_429, 429)], _paper()
    )

    ingest_tasks.daily_chain().apply().get()  # raises if any step failed

    # 1. three attempts; the worker would have waited 2 and then 6 minutes
    assert arxiv.calls == 3
    assert night.countdowns["arxiv"] == [120.0, 360.0]
    # 2. one row per attempt: two say they will be retried, the third worked
    _, *runs = _runs(s, "arxiv-daily")
    assert len(runs) == 3
    assert _error(runs[0]) == f"will retry in 2 min (attempt 1 of 4): RetryableError: {TIMEOUT}"
    assert _error(runs[1]) == f"will retry in 6 min (attempt 2 of 4): RetryableError: {HTTP_429}"
    assert runs[2].errors == {} and runs[2].counts["fetched"] == 1
    # 3. the cursor moved once: only the successful attempt set one
    with_cursor = [r for r in runs if r.counts.get("cursor")]
    assert with_cursor == [runs[2]]
    cursor = daily.last_cursor(s)
    assert cursor == datetime.fromisoformat(runs[2].counts["cursor"]) and cursor > OLD_CURSOR
    assert s.get(Paper, "2610.90001") is not None
    # 4. the rest of the chain ran
    assert [r.errors for r in _runs(s, "hf-daily")] == [{}]
    assert night.later_steps == ["index", "enrich"]
    assert len(_runs(s, "score-relevance")) == 1 and len(_runs(s, "select-fulltext")) == 1
    # 5. /status shows a pending retry as one, the note first (it cuts errors at 300 chars)
    shown = status._run_out(runs[1], datetime.now(UTC))
    assert shown.error is not None
    assert shown.error.startswith("will retry in 6 min (attempt 2 of 4)")
    assert shown.retrying is True  # the health summary says "waiting to retry", not "failed"


def test_arxiv_failing_every_attempt_stops_the_chain_and_keeps_the_cursor(
    migrated_session: Session, night: Night
) -> None:
    s = migrated_session
    night.arxiv = arxiv = FakeArxiv(
        [RetryableError(TIMEOUT)] * 4 + [RetryableError(HTTP_429, 429)], _paper()
    )

    with pytest.raises(RetryableError, match="Read timed out"):
        ingest_tasks.daily_chain().apply().get()

    assert arxiv.calls == 4  # 4 attempts, the fifth error is never reached
    assert night.countdowns["arxiv"] == [120.0, 360.0, 1080.0]
    _, *runs = _runs(s, "arxiv-daily")
    assert [_error(r).split(": RetryableError")[0] for r in runs] == [
        "will retry in 2 min (attempt 1 of 4)",
        "will retry in 6 min (attempt 2 of 4)",
        "will retry in 18 min (attempt 3 of 4)",
        "last attempt (4 of 4), no retry",
    ]
    assert all(r.finished_at is not None and not r.counts.get("cursor") for r in runs)
    assert daily.last_cursor(s) == OLD_CURSOR  # unchanged: the next run loses nothing
    # the chain stopped at its first step, as before
    assert _runs(s, "hf-daily") == [] and night.hf.calls == 0
    assert night.later_steps == []


def test_a_value_error_is_not_retried_and_says_nothing_about_a_retry(
    migrated_session: Session, night: Night
) -> None:
    s = migrated_session
    night.arxiv = arxiv = FakeArxiv([ValueError("not a valid arXiv category")], _paper())

    with pytest.raises(ValueError):
        ingest_tasks.daily_chain().apply().get()

    assert arxiv.calls == 1 and night.countdowns["arxiv"] == []
    _, *runs = _runs(s, "arxiv-daily")
    assert [_error(r) for r in runs] == ["ValueError: not a valid arXiv category"]
    assert night.later_steps == []


def test_hf_daily_fails_once_then_succeeds_and_the_chain_goes_on(
    migrated_session: Session, night: Night
) -> None:
    s = migrated_session
    night.arxiv = FakeArxiv([], _paper())
    night.hf = FakeHF([RetryableError("GET /api/daily_papers -> HTTP 503", 503)])

    ingest_tasks.daily_chain().apply().get()

    assert night.countdowns == {"arxiv": [], "hf": [120.0]}
    runs = _runs(s, "hf-daily")
    assert [_error(r) for r in runs] == [
        "will retry in 2 min (attempt 1 of 4): RetryableError: GET /api/daily_papers -> HTTP 503",
        "",
    ]
    assert night.later_steps == ["index", "enrich"]
    assert len(_runs(s, "arxiv-daily")) == 2  # the seed + one successful run: not re-run


def test_hf_daily_failing_every_attempt_stops_the_chain(
    migrated_session: Session, night: Night
) -> None:
    s = migrated_session
    night.arxiv = FakeArxiv([], _paper())
    night.hf = FakeHF([RetryableError("GET /api/daily_papers -> HTTP 503", 503)] * 4)

    with pytest.raises(RetryableError):
        ingest_tasks.daily_chain().apply().get()

    assert night.countdowns["hf"] == [120.0, 360.0, 1080.0] and night.hf.calls == 4
    runs = _runs(s, "hf-daily")
    assert [_error(r).split(": RetryableError")[0] for r in runs] == [
        "will retry in 2 min (attempt 1 of 4)",
        "will retry in 6 min (attempt 2 of 4)",
        "will retry in 18 min (attempt 3 of 4)",
        "last attempt (4 of 4), no retry",
    ]
    assert night.later_steps == []
    # arXiv had already worked: its cursor moved once and stays
    cursor = daily.last_cursor(s)
    assert cursor is not None and cursor > OLD_CURSOR

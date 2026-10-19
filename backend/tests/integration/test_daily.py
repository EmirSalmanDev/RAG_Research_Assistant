"""daily.py and the Celery tasks on a throwaway database and a `chunks_test_<random>`
collection, with fake arXiv / Hugging Face / full-text sources and fake embedders (no
network, no OpenAI). Tasks are run directly with `.run()` (no broker needed)."""

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk as ChunkRow
from app.db.models import IngestRun, LibraryItem, Paper
from app.ingest import daily, tasks
from app.ingest import index as index_module
from app.ingest.index import ensure_collection, index_abstracts
from app.sources.arxiv import ArxivPaper
from app.sources.hf import HFPaper
from tests.integration.test_fulltext import HF_MD, FakeSources
from tests.integration.test_indexing import DIM, FakeDense, FakeSparse

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _arxiv(n: int, version: int = 1, hours_ago: int = 1) -> ArxivPaper:
    when = NOW - timedelta(hours=hours_ago)
    return ArxivPaper(
        arxiv_id=f"2609.{n:05d}", version=version, title=f"Invented {n}",
        abstract=f"An invented abstract number {n}.", authors=["Ada Example"],
        categories=["cs.CL", "cs.LG"], primary_category="cs.CL",
        published_at=when, updated_at=when,
    )  # fmt: skip


class FakeArxiv:
    def __init__(self, window: list[ArxivPaper], by_id: dict[str, ArxivPaper] | None = None):
        self.window, self.by_id = window, by_id or {}
        self.calls: list[tuple[datetime, datetime | None]] = []
        self.categories: list[list[str]] = []
        self.id_requests: list[list[str]] = []

    def updated_papers(self, categories: Sequence[str], since: datetime,
                       until: datetime | None = None, page_size: int = 100,
                       max_papers: int = 20_000) -> list[ArxivPaper]:  # fmt: skip
        self.calls.append((since, until))
        self.categories.append(list(categories))
        return self.window[:max_papers]

    def get_papers(self, arxiv_ids: Sequence[str]) -> list[ArxivPaper]:
        self.id_requests.append(list(arxiv_ids))
        return [self.by_id[i] for i in arxiv_ids if i in self.by_id]


def _count(session: Session, model: Any, *where: Any) -> int:
    return session.scalar(select(func.count()).select_from(model).where(*where)) or 0


@pytest.fixture
def collection(
    qdrant: QdrantClient, scratch_collection: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[str]:
    """The scratch collection, also set as the configured one (tasks use the default)."""
    ensure_collection(qdrant, scratch_collection, DIM)
    monkeypatch.setenv("QDRANT_COLLECTION", scratch_collection)
    get_settings.cache_clear()
    index_module.get_qdrant.cache_clear()
    yield scratch_collection
    index_module.close_qdrant()
    get_settings.cache_clear()


# --- runs, cursor, window ---------------------------------------------------------------------


def test_recorded_run_writes_counts_or_the_error(migrated_session: Session) -> None:
    with daily.recorded_run(migrated_session, "invented-step") as counts:
        counts["done"] = 3
    with pytest.raises(RuntimeError), daily.recorded_run(migrated_session, "invented-step"):
        raise RuntimeError("boom")
    ok, failed = migrated_session.scalars(select(IngestRun).order_by(IngestRun.id)).all()
    assert (ok.counts, ok.errors, ok.finished_at is not None) == ({"done": 3}, {}, True)
    assert failed.errors == {"error": "RuntimeError: boom"} and failed.finished_at is not None


def test_cursor_comes_from_the_last_successful_arxiv_run(migrated_session: Session) -> None:
    s = migrated_session
    assert daily.last_cursor(s) is None
    for source, cursor, errors in [
        ("arxiv-backfill", "2026-09-20T00:00:00+00:00", {}),
        ("arxiv-daily", "2026-09-29T07:00:00+00:00", {}),
        ("arxiv-daily", "2026-09-30T07:00:00+00:00", {"error": "x"}),  # failed: ignored
        ("hf-daily", "2026-09-30T08:00:00+00:00", {}),  # not an arXiv run: ignored
    ]:
        s.add(IngestRun(source=source, counts={"cursor": cursor}, errors=errors,
                        finished_at=datetime.fromisoformat(cursor)))  # fmt: skip
    s.commit()
    assert daily.last_cursor(s) == datetime(2026, 9, 29, 7, tzinfo=UTC)


def test_daily_fetch_uses_the_cursor_and_is_idempotent(migrated_session: Session) -> None:
    s = migrated_session
    arxiv = FakeArxiv([_arxiv(1), _arxiv(2)])
    first = daily.fetch_arxiv_daily(s, arxiv, now=NOW)
    assert arxiv.calls[0] == (NOW - timedelta(days=5), NOW)  # no cursor: 1 day + 4 days
    assert (first["new"], first["cursor"]) == (2, NOW.isoformat())

    later = NOW + timedelta(days=1)
    arxiv.window = [_arxiv(1, version=2), _arxiv(2), _arxiv(3)]
    second = daily.fetch_arxiv_daily(s, arxiv, now=later)
    assert arxiv.calls[1] == (NOW - timedelta(days=4), later)  # cursor minus the overlap
    assert (second["new"], second["new_version"], second["unchanged"]) == (1, 1, 1)
    assert _count(s, Paper) == 3
    assert [r.source for r in s.scalars(select(IngestRun))] == ["arxiv-daily", "arxiv-daily"]


def test_hf_daily_merges_and_fetches_missing_papers(migrated_session: Session) -> None:
    s = migrated_session
    daily.store_papers(s, [_arxiv(1)])
    s.commit()

    class FakeHF:
        def daily_papers(self, day: date) -> list[HFPaper]:
            return [
                HFPaper(id="2609.00001", upvotes=5, projectPage="https://example.org/p1"),
                HFPaper(id="2609.00007", upvotes=9, githubRepo="https://github.com/a/b"),
            ]

    arxiv = FakeArxiv([], by_id={"2609.00007": _arxiv(7)})
    counts = daily.fetch_hf_daily(s, FakeHF(), arxiv, today=NOW.date())
    assert arxiv.id_requests == [["2609.00007"]]  # one batched request for the missing one
    assert (counts["listed"], counts["missing"], counts["new"], counts["merged"]) == (2, 1, 1, 2)
    p1, p7 = s.get(Paper, "2609.00001"), s.get(Paper, "2609.00007")
    assert p1 is not None and (p1.hf_upvotes, p1.project_url) == (5, "https://example.org/p1")
    assert p7 is not None and (p7.hf_upvotes, p7.github_url) == (9, "https://github.com/a/b")


# --- backfill ---------------------------------------------------------------------------------


def test_backfill_twice_gives_the_same_counts_and_no_new_embeddings(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    s = migrated_session
    arxiv = FakeArxiv([_arxiv(n) for n in range(1, 6)])
    plan = daily.plan_backfill(7, NOW)
    results = []
    for _ in range(2):
        dense = FakeDense()

        def index(sess: Session, d: FakeDense = dense) -> int:
            return index_abstracts(sess, dense=d, sparse=FakeSparse())

        result = daily.backfill(s, arxiv, plan, index)
        results.append((result["fetch"]["fetched"], result["index"]["indexed"], len(dense.calls)))
        assert _count(s, ChunkRow) == qdrant.count(collection, exact=True).count == 5
    assert results == [(5, 5, 1), (5, 0, 0)]  # the second run embeds nothing
    assert daily.last_cursor(s) == NOW  # the backfill also sets the cursor


def test_a_crash_between_postgres_and_qdrant_repairs_itself(
    migrated_session: Session, qdrant: QdrantClient, collection: str
) -> None:
    s = migrated_session
    daily.store_papers(s, [_arxiv(n) for n in range(1, 4)])
    s.commit()

    class CrashingDense(FakeDense):
        def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
            raise ConnectionError("worker killed")

    with pytest.raises(ConnectionError):
        index_abstracts(s, dense=CrashingDense(), sparse=FakeSparse())
    assert _count(s, ChunkRow) == 3 and qdrant.count(collection, exact=True).count == 0
    assert index_abstracts(s, dense=FakeDense(), sparse=FakeSparse()) == 3  # re-run repairs
    assert qdrant.count(collection, exact=True).count == 3


# --- the Celery tasks, run directly -----------------------------------------------------------


@pytest.fixture
def fake_task_deps(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    deps: dict[str, Any] = {
        "arxiv": FakeArxiv([_arxiv(1), _arxiv(2)]),
        "sources": FakeSources(hf=HF_MD),
    }
    monkeypatch.setattr(tasks, "_arxiv", lambda: deps["arxiv"])
    monkeypatch.setattr(tasks, "_embedders", lambda: {"dense": FakeDense(), "sparse": FakeSparse()})
    monkeypatch.setattr(tasks, "default_sources", lambda: deps["sources"])
    return deps


@pytest.mark.usefixtures("fake_task_deps")
def test_tasks_are_idempotent(migrated_session: Session, collection: str) -> None:
    s = migrated_session
    for _ in range(2):
        tasks.fetch_arxiv.run()
        tasks.index_abstracts.run()
        assert _count(s, Paper) == 2 and _count(s, ChunkRow) == 2
    sources = {r.source for r in s.scalars(select(IngestRun))}
    assert {"arxiv-daily", "index-abstracts"} <= sources


def test_paper_chain_steps_index_the_full_text(
    migrated_session: Session, qdrant: QdrantClient, collection: str, fake_task_deps: dict[str, Any]
) -> None:
    s = migrated_session
    tasks.fetch_arxiv.run()
    for _ in range(2):  # the second pass changes nothing
        for step in (tasks.fetch_fulltext, tasks.clean_paper, tasks.chunk_paper, tasks.embed_paper):
            step.run("2609.00001")
        s.expire_all()
        paper = s.get(Paper, "2609.00001")
        assert paper is not None and paper.fulltext_status == "indexed"
        rows = _count(s, ChunkRow, ChunkRow.arxiv_id == "2609.00001")
        assert rows > 1 and rows == qdrant.count(collection, exact=True).count


@pytest.mark.usefixtures("fake_task_deps")
def test_a_failed_step_after_its_last_retry_marks_the_paper_failed(
    migrated_session: Session, collection: str
) -> None:
    tasks.fetch_arxiv.run()
    tasks.fetch_fulltext.on_failure(RuntimeError("down"), "id", ("2609.00002",), {}, None)
    migrated_session.expire_all()
    paper = migrated_session.get(Paper, "2609.00002")
    assert paper is not None and paper.fulltext_status == "failed"


@pytest.mark.usefixtures("fake_task_deps")
def test_select_fulltext_starts_one_chain_per_selected_paper(
    migrated_session: Session, collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks.fetch_arxiv.run()
    migrated_session.add(LibraryItem(arxiv_id="2609.00002", collection="invented"))
    migrated_session.commit()
    started: list[str] = []

    class Started:
        def __init__(self, arxiv_id: str) -> None:
            self.arxiv_id = arxiv_id

        def apply_async(self) -> None:
            started.append(self.arxiv_id)

    monkeypatch.setattr(tasks, "paper_chain", Started)
    assert tasks.select_fulltext.run()["selected"] == 1
    assert started == ["2609.00002"]


def test_the_fetch_uses_the_categories_saved_on_the_settings_page(
    migrated_session: Session,
) -> None:
    from app.config import get_settings
    from app.features import app_settings

    s = migrated_session
    arxiv = FakeArxiv([])
    daily.fetch_arxiv_daily(s, arxiv, NOW)
    assert arxiv.categories[-1] == get_settings().arxiv_categories  # nothing saved: config

    app_settings.save(s, {"categories": ["cs.CV", "stat.ML"]})
    s.commit()
    daily.fetch_arxiv_daily(s, arxiv, NOW)
    assert arxiv.categories[-1] == ["cs.CV", "stat.ML"]  # the override, no backfill is done
    assert len(arxiv.calls) == 2 and arxiv.calls[1][0] == arxiv.calls[1][1] - timedelta(days=4)  # type: ignore[operator]

    app_settings.save(s, {"categories": None})  # cleared: back to the config list
    s.commit()
    daily.fetch_arxiv_daily(s, arxiv, NOW)
    assert arxiv.categories[-1] == get_settings().arxiv_categories

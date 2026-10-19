"""The whole nightly chain, run eagerly (in this process, no broker) with the network
steps replaced by fakes: after it and the 08:00 digest task, a ranked digest exists.

Faked: the arXiv / Hugging Face / enrichment / abstract-indexing steps (covered by their
own tests), the interest embedding (OpenAI) and the Qdrant vector reads. Real: the
Celery chain wiring, the scoring task, the top-N selection, the digest task and a
scratch Postgres. The per-paper full-text chains are recorded, not started."""

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DigestEntry, IngestRun, Interest, Paper
from app.features import app_settings, digest
from app.features import tasks as digest_tasks
from app.ingest import daily
from app.ingest import tasks as ingest_tasks
from tests.digest_fakes import FakeDense, FakeVectors, make_interest, make_paper

pytestmark = pytest.mark.integration


class Recorder:
    def __init__(self) -> None:
        self.steps: list[str] = []
        self.chains: list[str] = []

    def step(self, name: str) -> Any:
        def fake(*args: Any, **kwargs: Any) -> dict[str, int]:
            self.steps.append(name)
            return {}

        return fake


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> tuple[Recorder, FakeDense, FakeVectors]:
    import app.ingest.enrich as enrich
    import app.ingest.index as index
    import app.sources.hf as hf

    rec = Recorder()
    monkeypatch.setattr(daily, "fetch_arxiv_daily", rec.step("arxiv"))
    monkeypatch.setattr(daily, "fetch_hf_daily", rec.step("hf"))
    monkeypatch.setattr(ingest_tasks, "_arxiv", lambda: None)
    monkeypatch.setattr(ingest_tasks, "_embedders", lambda: {})
    monkeypatch.setattr(hf, "HFClient", lambda: None)
    monkeypatch.setattr(index, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(index, "index_abstracts", lambda *a, **k: 0)
    monkeypatch.setattr(enrich, "default_enrich_clients", lambda: None)
    monkeypatch.setattr(enrich, "enrich_daily", rec.step("enrich"))

    class FakeChain:
        def __init__(self, arxiv_id: str) -> None:
            self.arxiv_id = arxiv_id

        def apply_async(self) -> None:
            rec.chains.append(self.arxiv_id)

    monkeypatch.setattr(ingest_tasks, "paper_chain", FakeChain)
    dense, vectors = FakeDense(), FakeVectors()
    monkeypatch.setattr(digest_tasks, "_dense", lambda: dense)
    monkeypatch.setattr(digest_tasks, "_vectors", lambda: vectors)
    return rec, dense, vectors


def test_the_nightly_chain_scores_selects_and_the_digest_task_builds(
    migrated_session: Session, fakes: tuple[Recorder, FakeDense, FakeVectors]
) -> None:
    rec, dense, vectors = fakes
    s = migrated_session
    now = datetime.now(UTC)
    alpha = make_interest(s, "Alpha topic")  # not embedded yet: the chain does it
    for arxiv_id, vec, age in (
        ("2610.00001", [1.0, 0.0], 1),  # best
        ("2610.00002", [0.7, 0.7], 1),
        ("2610.00003", [0.1, 1.0], 2),
        ("2610.00004", [0.0, 1.0], 2),  # worst
    ):
        make_paper(s, arxiv_id, age_days=age, now=now)
        vectors.vectors[arxiv_id] = vec
    app_settings.save(s, {"fulltext_top_n": 2, "digest_size": 3})
    s.commit()

    result = ingest_tasks.daily_chain().apply()
    result.get()  # raises if any step failed

    # 1. the steps ran in order, and the score step came before the selection
    assert rec.steps == ["arxiv", "hf", "enrich"]
    runs = [r.source for r in s.scalars(select(IngestRun).order_by(IngestRun.id))]
    assert runs.index("score-relevance") < runs.index("select-fulltext")
    # 2. the interest was embedded in one request (the only paid call), then scored
    assert [feature for _, feature in dense.calls] == ["embed-interest"] and len(dense.calls) == 1
    s.expire_all()
    interest = s.get(Interest, alpha.id)
    assert interest is not None and interest.embedding == [1.0, 0.0]
    scores = dict(s.execute(select(Paper.arxiv_id, Paper.relevance_score)).all())
    assert scores["2610.00001"] == pytest.approx(0.8)
    assert all(v is not None for v in scores.values())
    # 3. the top 2 by relevance were handed to the full-text chain, best first
    assert rec.chains == ["2610.00001", "2610.00002"]

    # 4. before 08:00 there is no digest; the digest task then writes it
    assert s.scalar(select(DigestEntry.id)) is None
    built = digest_tasks.build_digest.apply().get()
    assert built["status"] == "built" and built["entries"] == 3
    s.expire_all()
    rows = s.scalars(select(DigestEntry).order_by(DigestEntry.rank)).all()
    assert [r.arxiv_id for r in rows] == ["2610.00001", "2610.00002", "2610.00003"]
    assert {r.digest_date for r in rows} == {digest.today_in_zone()}
    assert all(r.best_interest_id == alpha.id and r.status == "new" for r in rows)

    # 5. running the night again changes nothing: same digest, no second embedding
    ingest_tasks.daily_chain().apply().get()
    assert digest_tasks.build_digest.apply().get()["status"] == "exists"
    assert len(dense.calls) == 1
    s.expire_all()
    assert [r.arxiv_id for r in s.scalars(select(DigestEntry).order_by(DigestEntry.rank))] == [
        r.arxiv_id for r in rows
    ]


def test_a_failed_interest_embedding_does_not_stop_the_chain(
    migrated_session: Session,
    fakes: tuple[Recorder, FakeDense, FakeVectors],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, vectors = fakes
    s = migrated_session
    make_interest(s, "Keyword topic", keywords=["invented"])
    make_paper(s, "2610.00001", abstract="An invented abstract.", now=datetime.now(UTC))
    vectors.vectors["2610.00001"] = [1.0, 0.0]

    class Broken:
        def embed(self, texts: Any, feature: str) -> Any:
            raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(digest_tasks, "_dense", lambda: Broken())
    counts = digest_tasks.score_relevance.apply().get()
    assert "RuntimeError" in counts["embed_error"]
    assert counts["papers"] == 1  # scoring went on (by keywords: the interest has no vector)
    s.expire_all()
    assert s.scalar(select(Paper.relevance_score)) == pytest.approx(0.2)


def test_the_digest_task_builds_a_given_date(
    migrated_session: Session, fakes: tuple[Recorder, FakeDense, FakeVectors]
) -> None:
    s = migrated_session
    make_paper(s, "2610.00001", age_days=0.2, now=datetime.now(UTC))
    counts = digest_tasks.build_digest.apply(args=["2026-01-05"]).get()
    assert counts["digest_date"] == "2026-01-05"
    assert counts["entries"] == 0  # a day long before the paper existed

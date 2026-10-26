"""read_database() on a throwaway database: the real queries behind GET /status."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sqlalchemy.orm import Session

from app.api import status
from app.config import get_settings
from app.db.models import IngestRun, LLMUsage
from app.features import app_settings
from tests.digest_fakes import make_paper

pytestmark = pytest.mark.integration


def test_counts_latest_run_per_source_and_the_budget(migrated_session: Session) -> None:
    s = migrated_session
    make_paper(s, "2610.00001").fulltext_status = "indexed"
    make_paper(s, "2610.00002")
    make_paper(s, "2610.00003")
    old, new = datetime(2026, 10, 1, tzinfo=UTC), datetime(2026, 10, 2, tzinfo=UTC)
    s.add_all([
        IngestRun(source="embed-paper", started_at=old, finished_at=old, counts={}, errors={}),
        IngestRun(source="embed-paper", started_at=new, finished_at=new,
                  counts={"skipped": "budget cap reached"}, errors={}),
        IngestRun(source="index-abstracts", started_at=new, finished_at=new, counts={},
                  errors={"error": "RuntimeError: boom"}),
        IngestRun(source="select-fulltext", started_at=new, counts={}, errors={}),  # still running
    ])  # fmt: skip
    s.add(LLMUsage(feature="t", model="m", tokens_in=1, tokens_out=0, cost_usd=Decimal("1.25"),
                   created_at=datetime.now(UTC)))  # fmt: skip
    app_settings.save(s, {app_settings.KEY_BUDGET_CAP: 1.0})
    s.commit()

    out = status.read_database()

    assert out.papers_total == 3
    assert out.papers_by_fulltext_status == {"indexed": 1, "none": 2}
    by_source = {r.source: r for r in out.last_runs}
    assert set(by_source) == {"embed-paper", "index-abstracts", "select-fulltext"}  # one each
    assert by_source["embed-paper"].skipped == "budget cap reached" and by_source["embed-paper"].ok
    assert by_source["index-abstracts"].ok is False
    assert by_source["index-abstracts"].error == "RuntimeError: boom"
    assert (
        by_source["select-fulltext"].ok is False
        and by_source["select-fulltext"].finished_at is None
    )
    assert out.budget.cap_usd == 1.0 and out.budget.spent_usd == 1.25 and out.budget.paused


def test_an_empty_database(migrated_session: Session) -> None:
    out = status.read_database()
    assert out.papers_total == 0 and out.last_runs == []
    assert out.budget.cap_usd is None and not out.budget.paused


async def test_qdrant_points_before_the_first_ingest_is_zero_not_an_error(
    qdrant: QdrantClient, scratch_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", scratch_collection)
    get_settings.cache_clear()
    assert not qdrant.collection_exists(scratch_collection)
    assert await status.read_qdrant_points() == 0  # a fresh install has no collection yet

    qdrant.create_collection(
        scratch_collection, vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    qdrant.upsert(scratch_collection, [PointStruct(id=1, vector=[0.1, 0.2]),
                                       PointStruct(id=2, vector=[0.3, 0.4])])  # fmt: skip
    assert await status.read_qdrant_points() == 2

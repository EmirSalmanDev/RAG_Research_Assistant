"""GET /status with every reader replaced (no store, no network): the parts fail one by one."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from app.api import health, status
from app.main import app

NOW = datetime(2026, 10, 20, 7, 0, tzinfo=UTC)


def _database() -> status.DatabaseStatus:
    return status.DatabaseStatus(
        papers_total=3,
        papers_by_fulltext_status={"none": 2, "indexed": 1},
        last_runs=[
            status.RunOut(
                source="embed-paper",
                started_at=NOW,
                finished_at=NOW,
                ok=True,
                error=None,
                skipped="budget cap reached",
            )  # fmt: skip
        ],
        budget=status.BudgetOut(cap_usd=5.0, spent_usd=5.2, paused=True, month_start=NOW),
    )


async def _ok() -> None:
    return None


async def _queue() -> int:
    return 4


async def _points() -> int:
    return 8912


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def everything_readable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(health, "check_postgres", _ok)
    monkeypatch.setattr(health, "check_qdrant", _ok)
    monkeypatch.setattr(health, "check_redis", _ok)
    monkeypatch.setattr(status, "read_database", _database)
    monkeypatch.setattr(status, "read_queue_length", _queue)
    monkeypatch.setattr(status, "read_qdrant_points", _points)


async def test_everything_readable(client: AsyncClient) -> None:
    r = await client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["stores"] == {"postgres": "up", "qdrant": "up", "redis": "up"}
    assert body["queue_length"] == 4 and body["qdrant_points"] == 8912 and body["errors"] == {}
    assert body["database"]["budget"]["paused"] is True
    assert body["database"]["last_runs"][0]["skipped"] == "budget cap reached"


async def test_a_dead_redis_does_not_hide_the_database(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    async def down() -> None:
        raise ConnectionError("redis is gone")

    async def queue_down() -> int:
        raise ConnectionError("redis is gone")

    monkeypatch.setattr(health, "check_redis", down)
    monkeypatch.setattr(status, "read_queue_length", queue_down)
    r = await client.get("/status")
    assert r.status_code == 200  # the page must be able to show what is wrong
    body = r.json()
    assert body["stores"]["redis"] == "down" and body["queue_length"] is None
    assert body["errors"] == {"queue_length": "ConnectionError: redis is gone"}
    assert body["database"]["papers_total"] == 3 and body["qdrant_points"] == 8912


async def test_a_missing_collection_is_a_note_not_an_error_page(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    async def missing() -> int:
        raise ValueError("Collection `chunks` not found")

    monkeypatch.setattr(status, "read_qdrant_points", missing)
    body = (await client.get("/status")).json()
    assert body["qdrant_points"] is None and "not found" in body["errors"]["qdrant_points"]


async def test_a_database_that_hangs_times_out(
    client: AsyncClient, monkeypatch: MonkeyPatch
) -> None:
    import time

    monkeypatch.setattr(status, "TIMEOUT_S", 0.05)
    monkeypatch.setattr(status, "read_database", lambda: time.sleep(0.3))
    r = await client.get("/status")
    body = r.json()
    assert body["database"] is None and "no answer" in body["errors"]["database"]
    await asyncio.sleep(0.4)  # let the abandoned thread finish before the test ends

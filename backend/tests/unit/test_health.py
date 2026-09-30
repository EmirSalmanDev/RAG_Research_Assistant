"""/health with the three stores mocked (no network)."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from pytest import MonkeyPatch

from app.api import health
from app.main import app


async def _ok() -> None:
    return None


async def _fail() -> None:
    raise ConnectionError("store unreachable")


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def stores_up(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(health, "check_postgres", _ok)
    monkeypatch.setattr(health, "check_qdrant", _ok)
    monkeypatch.setattr(health, "check_redis", _ok)


async def test_all_up(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"postgres": "up", "qdrant": "up", "redis": "up"}


@pytest.mark.parametrize("store", ["postgres", "qdrant", "redis"])
async def test_one_down_returns_503(
    client: AsyncClient, monkeypatch: MonkeyPatch, store: str
) -> None:
    monkeypatch.setattr(health, f"check_{store}", _fail)
    r = await client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body[store] == "down"
    assert [v for k, v in body.items() if k != store] == ["up", "up"]


async def test_hung_store_times_out(client: AsyncClient, monkeypatch: MonkeyPatch) -> None:
    import asyncio

    async def hang() -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(health, "TIMEOUT_S", 0.05)
    monkeypatch.setattr(health, "check_redis", hang)
    r = await client.get("/health")
    assert r.status_code == 503
    assert r.json()["redis"] == "down"

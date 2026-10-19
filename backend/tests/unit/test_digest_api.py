"""/digest with a fake store: query handling and status codes, no database."""

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, date, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.digest import BestInterestOut, DigestEntryOut, DigestOut, get_digest_store
from app.main import app

DAY = date(2026, 10, 19)
ENTRY = DigestEntryOut(
    arxiv_id="2610.00001", rank=1, score=0.8, status="new", title="An Invented Paper",
    authors=["Ada Example"], author_count=1, categories=["cs.CL"],
    published_at=datetime(2026, 10, 18, tzinfo=UTC), hf_upvotes=12, trending=True,
    best_interest=BestInterestOut(id=1, name="Alpha"), summary="Invented.",
    summary_source="abstract", abs_url="https://arxiv.org/abs/2610.00001",
)  # fmt: skip


class FakeStore:
    def __init__(self) -> None:
        self.got: list[date | None] = []
        self.acted: list[tuple[date, str, str]] = []

    def get(self, requested: date | None) -> DigestOut:
        self.got.append(requested)
        return DigestOut(
            requested_date=requested or DAY, digest_date=DAY, is_fallback=requested != DAY,
            has_interests=True, recent_dates=[DAY], entries=[ENTRY],
        )  # fmt: skip

    def act(self, digest_date: date, arxiv_id: str, action: str) -> DigestEntryOut | None:
        self.acted.append((digest_date, arxiv_id, action))
        if arxiv_id == "2610.99999":
            return None
        return ENTRY.model_copy(update={"status": {"save": "saved", "dismiss": "dismissed",
                                                   "undo": "new"}[action]})  # fmt: skip


@pytest.fixture
def store() -> Iterator[FakeStore]:
    fake = FakeStore()
    app.dependency_overrides[get_digest_store] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
async def client(store: FakeStore) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_get_digest_passes_the_date(client: AsyncClient, store: FakeStore) -> None:
    r = await client.get("/digest")
    assert r.status_code == 200 and store.got == [None]
    body = r.json()
    assert body["entries"][0]["best_interest"] == {"id": 1, "name": "Alpha"}
    assert body["entries"][0]["trending"] is True
    r = await client.get("/digest", params={"date": "2026-10-17"})
    assert store.got[-1] == date(2026, 10, 17) and r.json()["is_fallback"] is True


async def test_a_bad_date_is_422(client: AsyncClient) -> None:
    assert (await client.get("/digest?date=yesterday")).status_code == 422
    assert (await client.get("/digest?date=2026-13-45")).status_code == 422


@pytest.mark.parametrize(("action", "state"), [("save", "saved"), ("dismiss", "dismissed"),
                                               ("undo", "new")])  # fmt: skip
async def test_actions_return_the_updated_entry(
    client: AsyncClient, store: FakeStore, action: str, state: str
) -> None:
    r = await client.post(f"/digest/2610.00001/{action}", params={"date": "2026-10-19"})
    assert r.status_code == 200 and r.json()["status"] == state
    assert store.acted == [(DAY, "2610.00001", action)]


async def test_an_action_needs_a_date(client: AsyncClient) -> None:
    assert (await client.post("/digest/2610.00001/save")).status_code == 422


async def test_an_unknown_entry_is_404_and_a_slash_id_is_routed(
    client: AsyncClient, store: FakeStore
) -> None:
    r = await client.post("/digest/2610.99999/save", params={"date": "2026-10-19"})
    assert r.status_code == 404
    r = await client.post("/digest/hep-th/9901001/dismiss", params={"date": "2026-10-19"})
    assert r.status_code == 200 and store.acted[-1][1] == "hep-th/9901001"


async def test_get_is_the_only_read_and_actions_are_post_only(client: AsyncClient) -> None:
    assert (await client.get("/digest/2610.00001/save?date=2026-10-19")).status_code in (404, 405)
    assert (await client.delete("/digest")).status_code == 405

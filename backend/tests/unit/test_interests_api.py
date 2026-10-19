"""/interests and /settings with a fake store: validation and status codes, no database.
All text is invented."""

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from itertools import count

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.interests import InterestIn, InterestOut, SettingsIn, SettingsOut, get_interest_store
from app.features.interests import DuplicateInterestError
from app.main import app

STAMP = datetime(2026, 10, 19, 5, 0, tzinfo=UTC)


class FakeStore:
    def __init__(self) -> None:
        self.rows: dict[int, InterestOut] = {}
        self.ids = count(1)
        self.settings = SettingsOut(
            digest_size=20, categories=["cs.CL"], categories_default=["cs.CL"],
            categories_overridden=False, fulltext_top_n=20, budget_cap_usd=None,
            month_to_date_usd=0.5,
        )  # fmt: skip

    def _out(self, i: int, body: InterestIn) -> InterestOut:
        return InterestOut(
            id=i, name=body.name, description=body.description, keywords=body.keywords,
            categories=body.categories, is_active=body.is_active, weight=1.0, embedded=False,
            created_at=STAMP, updated_at=STAMP,
        )  # fmt: skip

    def list(self) -> list[InterestOut]:
        return list(self.rows.values())

    def create(self, body: InterestIn) -> InterestOut:
        if any(r.name == body.name for r in self.rows.values()):
            raise DuplicateInterestError(body.name)
        row = self._out(next(self.ids), body)
        self.rows[row.id] = row
        return row

    def update(self, interest_id: int, body: InterestIn) -> InterestOut | None:
        if interest_id not in self.rows:
            return None
        if any(r.name == body.name and r.id != interest_id for r in self.rows.values()):
            raise DuplicateInterestError(body.name)
        self.rows[interest_id] = self._out(interest_id, body)
        return self.rows[interest_id]

    def delete(self, interest_id: int) -> bool:
        return self.rows.pop(interest_id, None) is not None

    def get_settings(self) -> SettingsOut:
        return self.settings

    def put_settings(self, body: SettingsIn) -> SettingsOut:
        self.settings = self.settings.model_copy(
            update={
                "digest_size": body.digest_size,
                "categories": body.categories or self.settings.categories_default,
                "categories_overridden": bool(body.categories),
                "fulltext_top_n": body.fulltext_top_n,
                "budget_cap_usd": body.budget_cap_usd,
            }
        )
        return self.settings


@pytest.fixture
def store() -> Iterator[FakeStore]:
    fake = FakeStore()
    app.dependency_overrides[get_interest_store] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
async def client(store: FakeStore) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_create_list_update_delete(client: AsyncClient) -> None:
    r = await client.post(
        "/interests",
        json={"name": "  Agents   and tools ", "description": " About agents. ",
              "keywords": ["agent", "tool use"], "categories": ["cs.AI", "cs.AI", " cs.CL "]},
    )  # fmt: skip
    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "Agents and tools"  # whitespace tidied
    assert body["description"] == "About agents."
    assert body["categories"] == ["cs.AI", "cs.CL"]  # trimmed, no repeats
    assert body["is_active"] is True and body["weight"] == 1.0 and body["embedded"] is False

    assert [i["name"] for i in (await client.get("/interests")).json()] == ["Agents and tools"]

    r = await client.put(f"/interests/{body['id']}", json={"name": "Agents", "is_active": False})
    assert r.status_code == 200 and r.json()["is_active"] is False
    assert r.json()["keywords"] == [] and r.json()["description"] is None  # PUT replaces

    assert (await client.delete(f"/interests/{body['id']}")).status_code == 204
    assert (await client.get("/interests")).json() == []


async def test_unknown_ids_are_404(client: AsyncClient) -> None:
    assert (await client.put("/interests/99", json={"name": "x"})).status_code == 404
    assert (await client.delete("/interests/99")).status_code == 404


async def test_duplicate_names_are_409(client: AsyncClient) -> None:
    await client.post("/interests", json={"name": "One"})
    other = (await client.post("/interests", json={"name": "Two"})).json()["id"]
    assert (await client.post("/interests", json={"name": "One"})).status_code == 409
    assert (await client.put(f"/interests/{other}", json={"name": "One"})).status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"name": ""},
        {"name": "   "},
        {"name": "x" * 101},
        {"name": "x", "description": "d" * 1001},
        {"name": "x", "keywords": ["k"] * 31},
        {"name": "x", "keywords": ["k" * 61]},
        {"name": "x", "categories": ["not a category"]},
        {"name": "x", "categories": ["CS.CL"]},
        {"name": "x", "categories": ["cs.CL", "bad!"]},
        {"name": "x", "is_active": "maybe"},
        {},
    ],
)
async def test_invalid_interests_are_422(client: AsyncClient, body: dict[str, object]) -> None:
    assert (await client.post("/interests", json=body)).status_code == 422


async def test_settings_roundtrip(client: AsyncClient) -> None:
    r = await client.get("/settings")
    assert r.status_code == 200
    assert r.json()["month_to_date_usd"] == 0.5 and r.json()["budget_cap_usd"] is None

    r = await client.put(
        "/settings",
        json={"digest_size": 10, "categories": ["cs.CV", "stat.ML"], "fulltext_top_n": 5,
              "budget_cap_usd": 12.5},
    )  # fmt: skip
    assert r.status_code == 200
    assert r.json()["categories"] == ["cs.CV", "stat.ML"] and r.json()["categories_overridden"]
    assert (await client.get("/settings")).json()["digest_size"] == 10

    r = await client.put(  # an empty list = the default list; null cap = no cap
        "/settings", json={"digest_size": 10, "categories": [], "fulltext_top_n": 0}
    )
    assert r.json()["categories"] == ["cs.CL"] and r.json()["categories_overridden"] is False
    assert r.json()["fulltext_top_n"] == 0 and r.json()["budget_cap_usd"] is None


@pytest.mark.parametrize(
    "body",
    [
        {"digest_size": 0, "fulltext_top_n": 5},
        {"digest_size": 101, "fulltext_top_n": 5},
        {"digest_size": 5, "fulltext_top_n": -1},
        {"digest_size": 5, "fulltext_top_n": 101},
        {"digest_size": 5, "fulltext_top_n": 5, "budget_cap_usd": -1},
        {"digest_size": 5, "fulltext_top_n": 5, "categories": ["cs.CL", "nonsense here"]},
        {"digest_size": 5, "fulltext_top_n": 5, "categories": ["cs." + "a" * 5000]},
        {"digest_size": "many", "fulltext_top_n": 5},
        {"fulltext_top_n": 5},
    ],
)
async def test_invalid_settings_are_422(client: AsyncClient, body: dict[str, object]) -> None:
    assert (await client.put("/settings", json=body)).status_code == 422

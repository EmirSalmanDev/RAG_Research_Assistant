"""The real stores behind /interests, /settings and /digest, on a scratch database:
create interests over HTTP, build a digest with fake vectors, then save, dismiss and undo
over HTTP and watch the interest weights. No OpenAI call is possible or made."""

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Interest, LLMUsage
from app.features import digest
from app.main import app
from tests.digest_fakes import FakeVectors, make_paper

pytestmark = pytest.mark.integration


@pytest.fixture
async def client(migrated_session: Session) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _weights(client: AsyncClient) -> dict[str, float]:
    return {i["name"]: i["weight"] for i in (await client.get("/interests")).json()}


async def test_the_whole_digest_flow_over_http(
    client: AsyncClient, migrated_session: Session
) -> None:
    s = migrated_session
    # 1. interests over HTTP: no embedding, no cost
    for name in ("Alpha", "Beta"):
        r = await client.post("/interests", json={"name": name, "keywords": [name.lower()]})
        assert r.status_code == 201 and r.json()["embedded"] is False
    assert (await client.post("/interests", json={"name": "Alpha"})).status_code == 409
    assert s.scalar(select(LLMUsage.id)) is None

    # 2. before any digest exists: empty, and reading did not build one
    r = await client.get("/digest")
    assert r.status_code == 200
    assert (r.json()["digest_date"], r.json()["entries"], r.json()["has_interests"]) == (
        None, [], True,
    )  # fmt: skip

    # 3. the nightly job (here: called directly, with fake vectors and stored vectors)
    alpha, beta = s.scalars(select(Interest).order_by(Interest.name)).all()
    alpha.embedding, beta.embedding = [1.0, 0.0], [0.0, 1.0]
    s.commit()
    # The papers must be young relative to the clock the digest reads (7-day window); the
    # default fixed date of make_paper ages out of the window as the real date moves on.
    now = datetime.now(UTC)
    for i, upvotes in enumerate((25, None), start=1):
        make_paper(
            s, f"2610.0000{i}", age_days=1, upvotes=upvotes,
            abstract=f"First sentence {i}. Second sentence.", now=now,
        )  # fmt: skip
    vectors = FakeVectors({"2610.00001": [1.0, 0.0], "2610.00002": [0.0, 1.0]})
    digest.build_digest(s, vectors, now=now)
    s.commit()

    # 4. GET /digest
    body = (await client.get("/digest")).json()
    day = body["digest_date"]
    assert body["is_fallback"] is False and body["recent_dates"] == [day]
    first, second = body["entries"]
    assert (first["arxiv_id"], first["rank"], first["status"]) == ("2610.00001", 1, "new")
    assert first["best_interest"]["name"] == "Alpha" and first["trending"] is True
    assert first["summary"] == "First sentence 1." and first["summary_source"] == "abstract"
    assert first["abs_url"] == "https://arxiv.org/abs/2610.00001"
    assert second["best_interest"]["name"] == "Beta" and second["trending"] is False

    # 5. a past date falls back to the latest digest
    past = (datetime.fromisoformat(day) - timedelta(days=3)).date().isoformat()
    fallback = (await client.get("/digest", params={"date": past})).json()
    assert fallback["digest_date"] == day and fallback["is_fallback"] is True

    # 6. Save, Dismiss, Undo over HTTP
    q = {"date": day}
    r = await client.post("/digest/2610.00001/save", params=q)
    assert r.status_code == 200 and r.json()["status"] == "saved"
    assert (await client.post("/digest/2610.00001/save", params=q)).json()["status"] == "saved"
    assert (await client.post("/digest/2610.00002/dismiss", params=q)).json()["status"] == (
        "dismissed"
    )
    assert await _weights(client) == {"Alpha": pytest.approx(1.10), "Beta": pytest.approx(0.95)}
    assert (await client.post("/digest/2610.00001/undo", params=q)).json()["status"] == "new"
    assert (await client.post("/digest/2610.00002/undo", params=q)).json()["status"] == "new"
    assert await _weights(client) == {"Alpha": 1.0, "Beta": 1.0}  # exactly back

    # 7. unknown paper or date
    assert (await client.post("/digest/2610.99999/save", params=q)).status_code == 404
    assert (await client.post("/digest/2610.00001/save", params={"date": past})).status_code == 404


async def test_settings_over_http_and_the_month_to_date_spend(
    client: AsyncClient, migrated_session: Session
) -> None:
    s = migrated_session
    s.add(LLMUsage(feature="t", model="m", tokens_in=1, tokens_out=1, cost_usd=Decimal("0.25")))
    s.commit()
    r = await client.get("/settings")
    assert r.status_code == 200
    body = r.json()
    assert (body["digest_size"], body["fulltext_top_n"], body["budget_cap_usd"]) == (20, 20, None)
    assert body["categories"] == body["categories_default"] and not body["categories_overridden"]
    assert body["month_to_date_usd"] == pytest.approx(0.25)

    r = await client.put("/settings", json={"digest_size": 7, "categories": ["cs.CV"],
                                            "fulltext_top_n": 3, "budget_cap_usd": 5})  # fmt: skip
    assert r.status_code == 200
    saved = (await client.get("/settings")).json()
    assert (saved["digest_size"], saved["categories"], saved["budget_cap_usd"]) == (7, ["cs.CV"], 5)
    assert saved["categories_overridden"] is True

    r = await client.put(
        "/settings", json={"digest_size": 7, "categories": [], "fulltext_top_n": 3}
    )
    cleared = r.json()
    assert (
        cleared["categories"] == cleared["categories_default"] and cleared["budget_cap_usd"] is None
    )


async def test_editing_an_interest_marks_it_for_a_new_embedding(
    client: AsyncClient, migrated_session: Session
) -> None:
    s = migrated_session
    made = (await client.post("/interests", json={"name": "Alpha", "keywords": ["a"]})).json()
    row = s.get(Interest, made["id"])
    assert row is not None
    from app.features import interests as repo

    row.embedding, row.embedding_model = [0.1, 0.2], "m"
    row.embedding_hash = repo.current_hash(row)
    s.commit()
    assert (await client.get("/interests")).json()[0]["embedded"] is True

    only_flags = {"name": "Alpha", "keywords": ["a"], "categories": ["cs.CL"], "is_active": False}
    assert (await client.put(f"/interests/{made['id']}", json=only_flags)).json()["embedded"]
    changed = {"name": "Alpha", "keywords": ["a", "b"]}
    assert (await client.put(f"/interests/{made['id']}", json=changed)).json()["embedded"] is False
    assert (await client.delete(f"/interests/{made['id']}")).status_code == 204

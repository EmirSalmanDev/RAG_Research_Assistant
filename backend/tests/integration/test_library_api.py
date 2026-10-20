"""The library routes over HTTP against a scratch database: status codes, shapes, the .bib
download. All papers are invented."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.orm import Session

from app.api.library import LIBRARY_STATUS_VALUES
from app.db.models import LIBRARY_STATUSES
from app.main import app
from tests.bib_validate import keys, validate
from tests.digest_fakes import make_paper

pytestmark = pytest.mark.integration


@pytest.fixture
async def api(migrated_session: Session) -> AsyncIterator[AsyncClient]:
    make_paper(migrated_session, "2610.00001", age_days=3, title="Widgets and Gadgets")
    make_paper(migrated_session, "2610.00002", age_days=2, title="More Widgets")
    make_paper(migrated_session, "hep-th/9901001", age_days=9, title="An Old Widget")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def _add(api: AsyncClient, ref: str, collection: str = "Thesis") -> dict[str, Any]:
    r = await api.post("/library/items", json={"reference": ref, "collection": collection})
    assert r.status_code in (200, 201), r.text
    body: dict[str, Any] = r.json()
    return body


def test_the_status_values_match_the_model() -> None:
    assert LIBRARY_STATUS_VALUES == LIBRARY_STATUSES


async def test_collections_are_created_listed_renamed_and_deleted(api: AsyncClient) -> None:
    r = await api.post("/library/collections", json={"name": "Thesis", "description": "Ch. 2"})
    assert r.status_code == 201
    assert r.json()["name"] == "Thesis" and r.json()["paper_count"] == 0
    assert (await api.post("/library/collections", json={"name": "Thesis"})).status_code == 409
    assert (await api.post("/library/collections", json={"name": "  "})).status_code == 422
    assert (await api.post("/library/collections", json={"name": ""})).status_code == 422
    await _add(api, "2610.00001")
    listing = (await api.get("/library/collections")).json()
    assert [(c["name"], c["paper_count"]) for c in listing] == [("Thesis", 1)]

    r = await api.put("/library/collections/Thesis", json={"name": "Thesis v2"})
    assert r.status_code == 200 and r.json()["name"] == "Thesis v2"
    assert r.json()["description"] == "Ch. 2" and r.json()["paper_count"] == 1
    r = await api.put("/library/collections/Thesis v2", json={"description": None})
    assert r.json()["description"] is None
    assert (await api.put("/library/collections/Nope", json={"name": "X"})).status_code == 404

    r = await api.delete("/library/collections/Thesis v2")
    assert r.json() == {"name": "Thesis v2", "items_removed": 1}
    assert (await api.delete("/library/collections/Thesis v2")).status_code == 404
    assert (await api.get("/library/collections")).json() == []


async def test_a_collection_name_with_a_slash_works_in_the_path(api: AsyncClient) -> None:
    await api.post("/library/collections", json={"name": "RL / agents"})
    r = await api.put("/library/collections/RL / agents", json={"name": "RL agents"})
    assert r.status_code == 200 and r.json()["name"] == "RL agents"


async def test_adding_papers_by_id_and_link(api: AsyncClient) -> None:
    r = await api.post(
        "/library/items", json={"reference": "arxiv.org/abs/2610.00001v2", "collection": "T"}
    )
    assert r.status_code == 201
    item = r.json()
    assert item["arxiv_id"] == "2610.00001" and item["status"] == "to_read"
    assert item["evidence"] == "abstract_only" and item["notes"] is None
    assert item["abs_url"] == "https://arxiv.org/abs/2610.00001"
    again = await api.post("/library/items", json={"reference": "2610.00001", "collection": "T"})
    assert again.status_code == 200 and again.json() == item
    old = await _add(api, "hep-th/9901001", "T")
    assert old["arxiv_id"] == "hep-th/9901001"
    listing = (await api.get("/library/items", params={"collection": "T"})).json()
    assert [i["arxiv_id"] for i in listing] == ["2610.00001", "hep-th/9901001"]


async def test_an_unknown_paper_is_404_with_the_ingest_hint(api: AsyncClient) -> None:
    r = await api.post("/library/items", json={"reference": "2610.99999", "collection": "T"})
    assert r.status_code == 404 and "Ingest the paper first" in r.json()["detail"]
    r = await api.post("/library/items", json={"reference": "not an id", "collection": "T"})
    assert r.status_code == 422
    r = await api.post(
        "/library/items", json={"reference": "2610.00001", "collection": "T", "status": "done"}
    )
    assert r.status_code == 422
    assert (await api.get("/library/collections")).json() == []


async def test_status_notes_and_move(api: AsyncClient) -> None:
    await _add(api, "2610.00001")
    await _add(api, "2610.00001", "Other")
    url = "/library/items/2610.00001"
    r = await api.patch(url, params={"collection": "Thesis"}, json={"status": "reading"})
    assert r.status_code == 200 and r.json()["status"] == "reading"
    r = await api.patch(url, params={"collection": "Thesis"}, json={"notes": "see section 4"})
    assert r.json()["notes"] == "see section 4" and r.json()["status"] == "reading"
    r = await api.patch(url, params={"collection": "Thesis"}, json={"notes": None})
    assert r.json()["notes"] is None and r.json()["status"] == "reading"
    for bad in ({"status": "finished"}, {"notes": "n" * 10_001}):
        assert (await api.patch(url, params={"collection": "Thesis"}, json=bad)).status_code == 422
    r = await api.patch(url, params={"collection": "Thesis"}, json={"collection": "Other"})
    assert r.status_code == 409
    r = await api.patch(url, params={"collection": "Thesis"}, json={"collection": "Third"})
    assert r.status_code == 200 and r.json()["collection"] == "Third"
    r = await api.patch(url, params={"collection": "Thesis"}, json={"status": "read"})
    assert r.status_code == 404
    r = await api.patch(url, json={"status": "read"})  # the collection is required
    assert r.status_code == 422


async def test_an_old_style_id_with_a_slash_is_addressable(api: AsyncClient) -> None:
    await _add(api, "hep-th/9901001")
    r = await api.patch(
        "/library/items/hep-th/9901001", params={"collection": "Thesis"}, json={"status": "read"}
    )
    assert r.status_code == 200 and r.json()["status"] == "read"
    r = await api.delete("/library/items/hep-th/9901001", params={"collection": "Thesis"})
    assert r.status_code == 204


async def test_removing_an_item(api: AsyncClient) -> None:
    await _add(api, "2610.00001")
    r = await api.delete("/library/items/2610.00001", params={"collection": "Thesis"})
    assert r.status_code == 204
    r = await api.delete("/library/items/2610.00001", params={"collection": "Thesis"})
    assert r.status_code == 404
    assert (await api.get("/papers/2610.00001")).status_code == 200  # the paper stays


async def test_listing_an_unknown_collection_is_404(api: AsyncClient) -> None:
    assert (await api.get("/library/items", params={"collection": "Nope"})).status_code == 404
    assert (await api.get("/library/bibtex", params={"collection": "Nope"})).status_code == 404
    assert (await api.get("/library/items")).status_code == 422


async def test_the_bib_download(api: AsyncClient) -> None:
    for ref in ("2610.00002", "hep-th/9901001", "2610.00001"):
        await _add(api, ref, "My Thesis")
    r = await api.get("/library/bibtex", params={"collection": "My Thesis"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/plain; charset=utf-8"
    assert r.headers["content-disposition"] == 'attachment; filename="my-thesis.bib"'
    assert validate(r.text) == [] and len(set(keys(r.text))) == 3
    assert [e for e in r.text.splitlines() if e.startswith("  eprint")] == [
        "  eprint = {hep-th/9901001},",  # oldest first
        "  eprint = {2610.00001},",
        "  eprint = {2610.00002},",
    ]
    again = await api.get("/library/bibtex", params={"collection": "My Thesis"})
    assert again.text == r.text  # deterministic

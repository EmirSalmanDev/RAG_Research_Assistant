"""POST /library/upload and DELETE /library/uploads/{id} over HTTP against a scratch database
and a scratch Qdrant collection, with fake embedders. PDFs are generated from invented text."""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from qdrant_client import QdrantClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.upload import get_upload_deps
from app.config import get_settings
from app.db.models import Paper
from app.ingest.upload import local_id_for, upload_slot
from app.main import app
from tests.integration.test_upload import World, collection  # noqa: F401  (the fixture)
from tests.pdf_fixtures import make_blank_pdf, make_pdf

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

PDF = {"content-type": "application/pdf"}


@pytest.fixture
def world(qdrant: QdrantClient, collection: str) -> World:  # noqa: F811
    return World(qdrant, collection)


@pytest.fixture
async def api(
    migrated_session: Session, world: World, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")  # the fake embedders never use it
    get_settings.cache_clear()
    app.dependency_overrides[get_upload_deps] = lambda: world.deps
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def test_upload_then_again_then_delete(
    api: AsyncClient, world: World, migrated_session: Session
) -> None:
    pdf = make_pdf(3, "api")
    r = await api.post(
        "/library/upload", params={"title": "An Invented Paper", "year": 2018, "collection": "T"},
        content=pdf, headers=PDF,
    )  # fmt: skip
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["arxiv_id"] == local_id_for(pdf) and body["created"] is True
    assert body["abs_url"] is None and body["year"] == 2018 and body["collection"] == "T"
    assert body["chunks"]["abstract"] == 1 and body["embedded"] == sum(body["chunks"].values())
    assert body["embedding_requests"] >= 1 and body["tokens_in"] > 0

    items = (await api.get("/library/items", params={"collection": "T"})).json()
    assert [(i["arxiv_id"], i["evidence"], i["abs_url"]) for i in items] == [
        (body["arxiv_id"], "uploaded", None)
    ]
    bib = (await api.get("/library/bibtex", params={"collection": "T"})).text
    assert "note = {Uploaded document}" in bib and "url" not in bib and "eprint" not in bib

    usage, parsed = len(world.usage), world.parsed
    again = await api.post(
        "/library/upload", params={"title": "Other title"}, content=pdf, headers=PDF
    )
    assert again.status_code == 200 and again.json()["created"] is False
    assert again.json()["embedded"] == 0 and again.json()["embedding_requests"] == 0
    assert len(world.usage) == usage and world.parsed == parsed  # no request, no parse

    gone = await api.delete(f"/library/uploads/{body['arxiv_id']}")
    assert gone.status_code == 200
    assert gone.json()["points"] == body["embedded"] and gone.json()["library_items"] == 1
    assert (await api.get(f"/papers/{body['arxiv_id']}")).status_code == 404
    assert (await api.delete(f"/library/uploads/{body['arxiv_id']}")).status_code == 404


async def test_requests_that_break_a_rule_are_refused_before_parsing(
    api: AsyncClient, world: World, migrated_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = make_pdf(2, "rules")
    # wrong content type, no title, bad year
    r = await api.post("/library/upload", params={"title": "T"}, content=pdf,
                       headers={"content-type": "text/plain"})  # fmt: skip
    assert r.status_code == 415
    assert (await api.post("/library/upload", content=pdf, headers=PDF)).status_code == 422
    r = await api.post(
        "/library/upload", params={"title": "T", "year": 1700}, content=pdf, headers=PDF
    )
    assert r.status_code == 422
    r = await api.post("/library/upload", params={"title": "   "}, content=pdf, headers=PDF)
    assert r.status_code == 422
    # empty, not a PDF
    r = await api.post("/library/upload", params={"title": "T"}, content=b"", headers=PDF)
    assert r.status_code == 422
    r = await api.post(
        "/library/upload", params={"title": "T"}, content=b"PK\x03\x04zip", headers=PDF
    )
    assert r.status_code == 415
    # a bad collection name is refused too
    r = await api.post(
        "/library/upload", params={"title": "T", "collection": "  "}, content=pdf, headers=PDF
    )
    assert r.status_code == 422
    assert world.parsed == 0 and world.usage == []
    assert migrated_session.scalar(select(func.count()).select_from(Paper)) == 0


async def test_an_oversized_body_is_refused_from_the_header_and_while_streaming(
    api: AsyncClient, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "2000")
    get_settings.cache_clear()
    big = make_pdf(3, "big")
    assert len(big) > 2000
    r = await api.post("/library/upload", params={"title": "T"}, content=big, headers=PDF)
    assert r.status_code == 413 and "limit" in r.json()["detail"]

    async def chunked() -> AsyncIterator[bytes]:  # no Content-Length: the stream is cut instead
        for _ in range(10):
            yield b"%PDF-" + b"x" * 500

    r = await api.post("/library/upload", params={"title": "T"}, content=chunked(), headers=PDF)
    assert r.status_code == 413
    assert world.parsed == 0


async def test_a_pdf_without_text_is_422(api: AsyncClient, world: World) -> None:
    r = await api.post(
        "/library/upload", params={"title": "Scan"}, content=make_blank_pdf(), headers=PDF
    )
    assert r.status_code == 422 and "no text layer" in r.json()["detail"]
    assert world.usage == []


async def test_a_second_upload_at_the_same_time_is_429(api: AsyncClient, world: World) -> None:
    with upload_slot():
        r = await api.post(
            "/library/upload", params={"title": "T"}, content=make_pdf(2, "busy"), headers=PDF
        )
    assert r.status_code == 429 and r.headers["retry-after"] == "10"
    assert world.parsed == 0


async def test_without_a_key_nothing_is_stored(
    api: AsyncClient, world: World, migrated_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    r = await api.post(
        "/library/upload", params={"title": "T"}, content=make_pdf(2, "k"), headers=PDF
    )
    assert r.status_code == 503
    assert world.parsed == 0
    assert migrated_session.scalar(select(func.count()).select_from(Paper)) == 0


async def test_an_embedding_failure_is_502_and_a_retry_finishes_it(
    api: AsyncClient, world: World, migrated_session: Session
) -> None:
    from app.ingest.embed import EmbeddingError

    healthy = world.deps.dense

    class Down:
        def embed(self, texts, feature):  # type: ignore[no-untyped-def]
            raise EmbeddingError("the embeddings API did not answer")

    world.deps.dense = lambda record: Down()
    pdf = make_pdf(2, "retry")
    r = await api.post("/library/upload", params={"title": "T"}, content=pdf, headers=PDF)
    assert r.status_code == 502
    world.deps.dense = healthy
    r = await api.post("/library/upload", params={"title": "T"}, content=pdf, headers=PDF)
    assert r.status_code == 200 and r.json()["created"] is False and r.json()["embedded"] > 0


async def test_an_arxiv_paper_cannot_be_deleted_through_the_upload_route(
    api: AsyncClient, migrated_session: Session
) -> None:
    from tests.digest_fakes import make_paper

    make_paper(migrated_session, "2610.00001")
    r = await api.delete("/library/uploads/2610.00001")
    assert r.status_code == 422 and "not an uploaded document" in r.json()["detail"]
    assert (await api.get("/papers/2610.00001")).status_code == 200

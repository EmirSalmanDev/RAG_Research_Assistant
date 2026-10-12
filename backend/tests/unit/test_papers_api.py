"""GET /papers/{id}, /papers/{id}/chunks and /chunks/{id} with a fake store.
The ASGI app runs in-process: no server, no database. All text is invented."""

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.papers import (
    ChunkOut,
    PaperChunksOut,
    PaperOut,
    _author_names,
    get_paper_store,
)
from app.main import app

CHUNK_ID = uuid.UUID("00000000-0000-4000-8000-0000000000c1")
PUBLISHED = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)

PAPER = PaperOut(
    arxiv_id="2609.00001",
    version=2,
    title="An Invented Paper",
    abstract="An invented abstract.",
    authors=["Ada Example", "Bo Sample"],
    categories=["cs.CL", "cs.LG"],
    primary_category="cs.CL",
    published_at=PUBLISHED,
    hf_upvotes=None,
    citation_count=None,
    github_url=None,
    project_url=None,
    fulltext_status="none",
    abs_url="https://arxiv.org/abs/2609.00001",
)
CHUNK = ChunkOut(
    chunk_id=CHUNK_ID,
    arxiv_id="2609.00001",
    title="An Invented Paper",
    chunk_type="section",
    section_path="Method > Training",
    ord=3,
    text="Invented chunk text.",
    published_at=PUBLISHED,
    abs_url="https://arxiv.org/abs/2609.00001",
)


class FakeStore:
    def __init__(self) -> None:
        self.limits: list[int] = []

    def get_paper(self, arxiv_id: str) -> PaperOut | None:
        return PAPER if arxiv_id == PAPER.arxiv_id else None

    def list_chunks(self, arxiv_id: str, limit: int) -> PaperChunksOut | None:
        self.limits.append(limit)
        if arxiv_id != PAPER.arxiv_id:
            return None
        return PaperChunksOut(arxiv_id=arxiv_id, chunks=[CHUNK])

    def get_chunk(self, chunk_id: uuid.UUID) -> ChunkOut | None:
        return CHUNK if chunk_id == CHUNK_ID else None


@pytest.fixture
def store() -> Iterator[FakeStore]:
    fake = FakeStore()
    app.dependency_overrides[get_paper_store] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
async def client(store: FakeStore) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_paper_returns_metadata_and_abs_link(client: AsyncClient) -> None:
    r = await client.get("/papers/2609.00001")
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "An Invented Paper"
    assert body["authors"] == ["Ada Example", "Bo Sample"]
    assert body["abs_url"] == "https://arxiv.org/abs/2609.00001"


async def test_unknown_paper_is_404(client: AsyncClient) -> None:
    assert (await client.get("/papers/0000.00000")).status_code == 404
    assert (await client.get("/papers/0000.00000/chunks")).status_code == 404


async def test_paper_chunks_route_is_not_swallowed_by_the_paper_route(
    client: AsyncClient, store: FakeStore
) -> None:
    r = await client.get("/papers/2609.00001/chunks", params={"limit": 5})
    assert r.status_code == 200
    assert [c["section_path"] for c in r.json()["chunks"]] == ["Method > Training"]
    assert store.limits == [5]


async def test_chunk_limit_is_bounded(client: AsyncClient) -> None:
    assert (await client.get("/papers/2609.00001/chunks?limit=0")).status_code == 422
    assert (await client.get("/papers/2609.00001/chunks?limit=501")).status_code == 422


async def test_old_style_ids_with_a_slash_reach_the_store(client: AsyncClient) -> None:
    assert (await client.get("/papers/hep-th/9901001")).status_code == 404  # not 405/422


async def test_chunk_returns_text_and_abs_link(client: AsyncClient) -> None:
    r = await client.get(f"/chunks/{CHUNK_ID}")
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Invented chunk text."
    assert body["abs_url"] == "https://arxiv.org/abs/2609.00001"


async def test_unknown_or_malformed_chunk_id(client: AsyncClient) -> None:
    assert (await client.get(f"/chunks/{uuid.uuid4()}")).status_code == 404
    assert (await client.get("/chunks/not-a-uuid")).status_code == 422


def test_author_names_accepts_objects_and_plain_strings() -> None:
    assert _author_names([{"name": "A"}, "B", {"x": 1}, 3]) == ["A", "B"]

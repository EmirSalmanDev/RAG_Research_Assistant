"""GET /search with the retrieval pipeline replaced by a fake (no stores, no models)."""

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import search as search_api
from app.config import get_settings
from app.ingest.embed import OpenAIKeyMissing
from app.main import app
from app.rag.rerank import RerankerUnavailable
from app.rag.search import SearchFilters, SearchHit

HIT = SearchHit(
    chunk_id=uuid.UUID("0214d8ce-b010-59e2-b4d6-06b4e659c8d5"),
    arxiv_id="2601.00001",
    title="An Invented Paper",
    section_path="3 Method",
    chunk_type="section",
    published_at="2026-01-02T09:30:00+00:00",
    text="Invented chunk text.",
    score=4.25,
)


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replaces the pipeline: records its arguments and returns one invented hit."""
    seen: list[dict[str, Any]] = []

    def fake_retrieve(query: str, filters: SearchFilters, **kw: Any) -> list[SearchHit]:
        seen.append({"query": query, "filters": filters, **kw})
        return [HIT]

    monkeypatch.setattr(search_api, "retrieve", fake_retrieve)
    return seen


async def test_returns_hits_with_an_abstract_link(
    client: AsyncClient, calls: list[dict[str, Any]]
) -> None:
    r = await client.get("/search", params={"q": "  how does it work?  "})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "how does it work?"
    assert (body["mode"], body["reranked"]) == ("hybrid", False)  # the configured defaults
    assert body["hits"] == [
        {
            "chunk_id": "0214d8ce-b010-59e2-b4d6-06b4e659c8d5",
            "arxiv_id": "2601.00001",
            "title": "An Invented Paper",
            "section_path": "3 Method",
            "chunk_type": "section",
            "published_at": "2026-01-02T09:30:00Z",
            "score": 4.25,
            "text": "Invented chunk text.",
            "abs_url": "https://arxiv.org/abs/2601.00001",
        }
    ]
    (call,) = calls
    assert call["query"] == "how does it work?" and call["limit"] == 10
    assert call["filters"] == SearchFilters()


async def test_filters_and_limit_reach_the_pipeline(
    client: AsyncClient, calls: list[dict[str, Any]]
) -> None:
    r = await client.get(
        "/search",
        params={
            "q": "lora",
            "limit": 5,
            "category": ["cs.CL", "cs.IR"],  # sent as category=cs.CL&category=cs.IR
            "arxiv_id": "2601.00001",
            "published_from": "2026-09-01",
            "published_to": "2026-09-30",
        },
    )
    assert r.status_code == 200
    (call,) = calls
    assert call["limit"] == 5
    assert call["filters"] == SearchFilters(
        arxiv_ids=("2601.00001",),
        categories=("cs.CL", "cs.IR"),
        published_from=datetime(2026, 9, 1, tzinfo=UTC),
        published_to=datetime(2026, 9, 30, 23, 59, 59, 999999, tzinfo=UTC),  # whole last day
    )


async def test_mode_and_rerank_follow_the_settings(
    client: AsyncClient, calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SEARCH_MODE", "bm25")
    monkeypatch.setenv("RERANK_ENABLED", "true")  # the config switch can turn it on
    get_settings.cache_clear()
    r = await client.get("/search", params={"q": "lora"})
    assert (r.json()["mode"], r.json()["reranked"]) == ("bm25", True)
    assert (calls[0]["config"].mode, calls[0]["config"].rerank) == ("bm25", True)
    get_settings.cache_clear()


async def test_rerank_is_off_unless_the_request_asks(
    client: AsyncClient, calls: list[dict[str, Any]]
) -> None:
    await client.get("/search", params={"q": "lora"})
    await client.get("/search", params={"q": "lora", "rerank": "true"})
    await client.get("/search", params={"q": "lora", "rerank": "false"})
    assert [c["config"].rerank for c in calls] == [False, True, False]


async def test_the_request_overrides_the_configured_default(
    client: AsyncClient, calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RERANK_ENABLED", "true")
    get_settings.cache_clear()
    r = await client.get("/search", params={"q": "lora", "rerank": "false"})
    assert r.json()["reranked"] is False and calls[0]["config"].rerank is False
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "   "},
        {"q": "x" * 1001},
        {"q": "lora", "limit": 0},
        {"q": "lora", "limit": 51},
        {"q": "lora", "published_from": "2026-09-30", "published_to": "2026-09-01"},
        {"q": "lora", "published_from": "not-a-date"},
        {"q": "lora", "rerank": "maybe"},
    ],
)
async def test_bad_requests_are_rejected_before_searching(
    client: AsyncClient, calls: list[dict[str, Any]], params: dict[str, Any]
) -> None:
    r = await client.get("/search", params=params)
    assert r.status_code == 422
    assert calls == []


@pytest.mark.parametrize(
    "error",
    [
        RerankerUnavailable("reranker is not in the cache; run make fetch-reranker"),
        OpenAIKeyMissing("OPENAI_API_KEY is not set (put it in .env); cannot embed"),
    ],
)
async def test_missing_setup_is_a_503_that_says_what_to_do(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def broken(*args: Any, **kwargs: Any) -> list[SearchHit]:
        raise error

    monkeypatch.setattr(search_api, "retrieve", broken)
    r = await client.get("/search", params={"q": "lora"})
    assert r.status_code == 503
    assert r.json()["detail"] == str(error)

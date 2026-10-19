"""POST /compare and POST /timeline over HTTP, with the collaborators faked.
The ASGI app runs in-process: no server, no network, no stores. All text is invented."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

import openai
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.explore import get_explore_deps
from app.config import get_settings
from app.features.deps import ExploreDeps
from app.features.timeline import LineFill, TimelineLines
from app.main import app
from tests.chat_fakes import parsed_reply, status_error
from tests.unit.test_compare import A, B, C, _deps, _fill
from tests.unit.test_timeline import PAPERS
from tests.unit.test_timeline import _deps as timeline_deps


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invented key: the routes only check that one is set (OpenAI itself is faked)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invented-for-tests")
    get_settings.cache_clear()


@pytest.fixture
def use() -> Iterator[Any]:
    def install(deps: ExploreDeps) -> ExploreDeps:
        app.dependency_overrides[get_explore_deps] = lambda: deps
        return deps

    yield install
    app.dependency_overrides.clear()


# --- /compare ---------------------------------------------------------------------------------


async def test_compare_returns_the_table(client: AsyncClient, key: None, use: Any) -> None:
    parsed = _fill({A: {"task": [1], "method": [2]}, B: {"task": [4], "method": []}})
    deps, *_ = _deps([parsed_reply(parsed)])
    use(deps)
    r = await client.post("/compare", json={"arxiv_ids": [A, B], "dimensions": ["method", "task"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dimensions"] == ["task", "method"]  # canonical order
    cells = {row["arxiv_id"]: row["cells"] for row in body["rows"]}
    assert [c["status"] for c in cells[A]] == ["cited", "cited"]
    assert [c["status"] for c in cells[B]] == ["cited", "not_stated"]
    assert cells[B][1]["text"] == "Not stated in the sources."
    first = cells[A][0]["citations"][0]
    source = next(s for s in body["sources"] if s["n"] == first["n"])
    assert source["chunk_id"] == first["chunk_id"] and source["arxiv_id"] == A
    assert source["abs_url"] == f"https://arxiv.org/abs/{A}"
    assert body["prompt_version"] == "compare_v1" and body["cost_usd"] > 0


async def test_compare_validates_before_any_paid_call(
    client: AsyncClient, key: None, use: Any
) -> None:
    deps, dense, search, _, rows = _deps([])
    use(deps)
    bad = [
        {"arxiv_ids": [A]},  # too few
        {"arxiv_ids": [A, B, C, "1", "2", "3"]},  # too many
        {"arxiv_ids": [A, A]},  # duplicate (rule in code)
        {"arxiv_ids": [A, B], "dimensions": ["price"]},  # unknown dimension
        {"arxiv_ids": [A, B], "dimensions": []},
    ]
    for body in bad:
        assert (await client.post("/compare", json=body)).status_code == 422, body
    r = await client.post("/compare", json={"arxiv_ids": [A, "9999.99999"]})
    assert r.status_code == 404 and "9999.99999" in r.json()["detail"]
    assert dense.requests == [] and search.calls == [] and rows == []


async def test_compare_without_a_key_is_503(client: AsyncClient, use: Any) -> None:
    deps, dense, *_ = _deps([])
    use(deps)
    r = await client.post("/compare", json={"arxiv_ids": [A, B]})
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
    assert dense.requests == []


async def test_compare_model_failure_is_502(client: AsyncClient, key: None, use: Any) -> None:
    deps, *_ = _deps([status_error(openai.InternalServerError, 500)] * 5)
    use(deps)
    r = await client.post("/compare", json={"arxiv_ids": [A, B]})
    assert r.status_code == 502


# --- /timeline --------------------------------------------------------------------------------


async def test_timeline_returns_points(client: AsyncClient, key: None, use: Any) -> None:
    lines = TimelineLines(lines=[LineFill(arxiv_id="p2", line="p2 changed things.")])
    deps, *_ = timeline_deps([parsed_reply(lines)], PAPERS)
    use(deps)
    r = await client.post(
        "/timeline",
        json={"topic": "adapters", "date_from": "2026-09-01", "key_papers": 1, "max_papers": 4},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [p["arxiv_id"] for p in body["points"]][0] == "p2"
    keys = [p for p in body["points"] if p["is_key"]]
    assert (
        len(keys) == 1
        and keys[0]["line"]
        and keys[0]["abs_url"].startswith("https://arxiv.org/abs/")
    )
    assert body["notes"]["narrow_range"] is True and body["range"]["first"] < body["range"]["last"]


async def test_timeline_validates_before_any_paid_call(
    client: AsyncClient, key: None, use: Any
) -> None:
    deps, dense, search, _, rows = timeline_deps([], PAPERS)
    use(deps)
    bad = [
        {"topic": "ab"},
        {"topic": "x" * 201},
        {"topic": "adapters", "max_papers": 51},
        {"topic": "adapters", "key_papers": 16},
        {"topic": "adapters", "date_from": "2026-09-02", "date_to": "2026-09-01"},
        {"topic": "   a   "},  # blank after trimming
        {"topic": "adapters", "date_from": "not a date"},
    ]
    for body in bad:
        assert (await client.post("/timeline", json=body)).status_code == 422, body
    assert dense.requests == [] and search.calls == [] and rows == []


async def test_timeline_without_a_key_is_503(client: AsyncClient, use: Any) -> None:
    deps, dense, *_ = timeline_deps([], PAPERS)
    use(deps)
    r = await client.post("/timeline", json={"topic": "adapters"})
    assert r.status_code == 503 and dense.requests == []


async def test_timeline_lines_failure_is_not_an_error(
    client: AsyncClient, key: None, use: Any
) -> None:
    deps, *_ = timeline_deps([status_error(openai.InternalServerError, 500)] * 5, PAPERS)
    use(deps)
    r = await client.post("/timeline", json={"topic": "adapters", "key_papers": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["notes"]["lines_call_failed"] is True
    assert all(p["line_source"] == "abstract_fallback" for p in body["points"] if p["is_key"])

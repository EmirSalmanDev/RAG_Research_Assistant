"""POST /chat over HTTP, with the pipeline's outside parts faked (see test_pipeline.py).
The ASGI app runs in-process: no server, no network, no stores."""

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.chat import get_chat_deps
from app.config import get_settings
from app.main import app
from app.rag.condense import HistoryMessage
from app.rag.conversations import ConversationNotFound
from tests.unit.test_pipeline import HISTORY, Harness, full_turn, harness

KNOWN = uuid.UUID("00000000-0000-4000-8000-0000000000bb")


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invented key: the route only checks that one is set (OpenAI itself is faked)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invented-for-tests")
    get_settings.cache_clear()


@pytest.fixture
def use() -> Iterator[Any]:
    """use(h) makes the route run on harness h; the override is removed afterwards."""

    def install(h: Harness) -> Harness:
        def load_history(conversation_id: uuid.UUID, limit: int) -> list[HistoryMessage]:
            if conversation_id != KNOWN:
                raise ConversationNotFound(str(conversation_id))
            return list(HISTORY)[-limit:]

        h.store.load_history = load_history  # type: ignore[method-assign]
        app.dependency_overrides[get_chat_deps] = lambda: h.deps
        return h

    yield install
    app.dependency_overrides.clear()


def parse_sse(body: str) -> list[tuple[str, Any]]:
    """`event: x` / `data: {...}` blocks, separated by blank lines -> (name, data)."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_a_new_conversation_streams_events_in_order(
    client: AsyncClient, key: None, use: Any
) -> None:
    h = use(harness(full_turn()[1:]))
    r = await client.post("/chat", json={"message": "  What is LoRA?  "})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = parse_sse(r.text)
    assert [e[0] for e in events] == ["meta", "sources", "token", "token", "verification", "done"]
    meta = events[0][1]
    assert meta["new_conversation"] is True and uuid.UUID(meta["conversation_id"])
    assert events[-1][1]["conversation_id"] == meta["conversation_id"]
    assert h.store.saved[0].question == "What is LoRA?"  # stripped
    assert h.store.saved[0].new_conversation is True


async def test_a_follow_up_uses_the_stored_history(
    client: AsyncClient, key: None, use: Any
) -> None:
    h = use(harness(full_turn()))
    r = await client.post(
        "/chat", json={"message": "and its limits?", "conversation_id": str(KNOWN)}
    )
    events = parse_sse(r.text)
    assert events[0][1]["standalone_question"] == "What are the limits of LoRA?"
    assert events[0][1]["new_conversation"] is False
    assert h.completions.calls[0]["kind"] == "create"  # condense ran on the history


async def test_an_unknown_conversation_is_404_before_any_call(
    client: AsyncClient, key: None, use: Any
) -> None:
    h = use(harness([]))
    r = await client.post("/chat", json={"message": "q", "conversation_id": str(uuid.uuid4())})
    assert r.status_code == 404
    assert h.completions.calls == [] and h.tracer.traces == []


async def test_a_missing_openai_key_is_503(client: AsyncClient, use: Any) -> None:
    h = use(harness([]))
    r = await client.post("/chat", json={"message": "q"})  # tests blank every key
    assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
    assert h.tracer.traces == []


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
async def test_bad_messages_are_422(client: AsyncClient, key: None, use: Any, message: str) -> None:
    use(harness([]))
    r = await client.post("/chat", json={"message": message})
    assert r.status_code == 422


async def test_a_failure_after_the_stream_started_is_an_error_event(
    client: AsyncClient, key: None, use: Any
) -> None:
    h = use(harness(full_turn()[1:2]))
    h.search.fail_with = ConnectionError("qdrant down")
    r = await client.post("/chat", json={"message": "q"})
    assert r.status_code == 200
    events = parse_sse(r.text)
    assert events[-1] == ("error", {"stage": "search", "message": "search failed: ConnectionError"})
    assert h.store.saved == []


async def test_closing_the_http_stream_closes_the_pipeline() -> None:
    """What happens on a disconnect: the SSE wrapper closes the pipeline generator, so
    every span is ended and the answer's usage is still logged; nothing is saved."""
    from app.api.chat import _sse
    from app.rag.pipeline import run_chat_turn

    h = harness(full_turn()[1:3])  # plan, then the streamed answer
    events = run_chat_turn(
        "q",
        conversation_id=KNOWN,
        new_conversation=True,
        history=[],
        deps=h.deps,
        settings=get_settings(),
    )
    stream = _sse(events)
    async for chunk in stream:
        if chunk.startswith(b"event: token"):
            break
    await stream.aclose()
    trace = h.trace
    assert all(s.ended for s in trace.spans) and len(trace.spans) == 6
    assert trace.span_named("generate").error == "turn ended early"
    assert [r[0] for r in h.chat_rows] == ["chat-plan", "chat-generate"]
    assert h.store.saved == []


def test_the_openapi_schema_documents_the_event_stream() -> None:
    operation = app.openapi()["paths"]["/chat"]["post"]
    assert "text/event-stream" in operation["responses"]["200"]["content"]

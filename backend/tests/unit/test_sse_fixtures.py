"""The /chat event stream, written out as files for the frontend's tests.

The browser side reads these streams with a hand-written parser and hand-written event
types (OpenAPI cannot describe server-sent events), so they are the weakest link between
the two halves. Instead of copying event shapes by hand, the streams here come out of the
real POST /chat route (in-process, with the pipeline's outside parts faked as in
test_pipeline.py), and frontend/lib/chat/recorded.test.ts replays them through the real
parser and reducer. The content is the invented text of this test suite, never paper text.

If the stream changes (a field, an event, an order), this test fails until the files are
rewritten, which makes the change visible in the diff the frontend tests then check:

    UPDATE_SSE_FIXTURES=1 uv run pytest tests/unit/test_sse_fixtures.py
"""

import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.chat import get_chat_deps
from app.config import get_settings
from app.main import app
from app.rag.condense import HistoryMessage
from app.rag.conversations import ConversationNotFound
from tests.chat_fakes import parsed_reply, stream_reply, text_reply
from tests.unit.test_pipeline import (
    ABSTRACT,
    HISTORY,
    Harness,
    _plan,
    _verdicts,
    full_turn,
    harness,
)

FIXTURE_DIR = Path(__file__).parents[3] / "frontend" / "lib" / "chat" / "testdata"
KNOWN = uuid.UUID("00000000-0000-4000-8000-0000000000bb")  # a stored conversation (a follow-up)


def _condense() -> Any:
    return text_reply("What are the limits of LoRA?", 200, 12)


def _no_sources() -> Harness:
    return harness([_condense(), parsed_reply(_plan(categories=["cs.CL"]), 300, 40)], hits=[])


def _insufficient() -> Harness:
    return harness(
        [
            _condense(),
            parsed_reply(_plan(), 300, 40),
            stream_reply(["INSUFFICIENT", "_SOURCES"], 1500, 5),
        ]
    )


def _search_fails() -> Harness:
    h = harness(full_turn()[:2])
    h.search.fail_with = ConnectionError("qdrant down")
    return h


def _abstract_only() -> Harness:
    return harness(
        [
            _condense(),
            parsed_reply(_plan(), 300, 40),
            stream_reply(["Abstracts only [1]."], 2000, 60),
            parsed_reply(_verdicts(True), 900, 30),
        ],
        hits=[ABSTRACT],
    )


# file name -> the pipeline setup whose stream it records
SCENARIOS: dict[str, Callable[[], Harness]] = {
    "turn-normal": lambda: harness(full_turn()),
    "turn-abstract-only": _abstract_only,
    "turn-refusal-no-sources": _no_sources,
    "turn-refusal-insufficient": _insufficient,
    "turn-error-search": _search_fails,
}


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invented-for-tests")
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


async def _stream_of(h: Harness) -> str:
    def load_history(conversation_id: uuid.UUID, limit: int) -> list[HistoryMessage]:
        if conversation_id != KNOWN:
            raise ConversationNotFound(str(conversation_id))
        return list(HISTORY)[-limit:]

    h.store.load_history = load_history  # type: ignore[method-assign]
    app.dependency_overrides[get_chat_deps] = lambda: h.deps
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/chat", json={"message": "and its limits?", "conversation_id": str(KNOWN)}
        )
    assert r.status_code == 200
    return r.text


@pytest.mark.parametrize("name", sorted(SCENARIOS))
async def test_the_recorded_stream_matches_what_the_route_sends(name: str, key: None) -> None:
    text = await _stream_of(SCENARIOS[name]())
    path = FIXTURE_DIR / f"{name}.sse"
    if os.environ.get("UPDATE_SSE_FIXTURES") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert path.exists(), f"{path.name} is missing: run with UPDATE_SSE_FIXTURES=1"
    assert path.read_text(encoding="utf-8") == text, (
        f"the /chat stream for {name} changed: run with UPDATE_SSE_FIXTURES=1 and review the diff"
    )


async def test_the_streams_are_deterministic(key: None) -> None:
    # A recorded stream that differed between two runs could never be compared.
    first = await _stream_of(SCENARIOS["turn-normal"]())
    second = await _stream_of(SCENARIOS["turn-normal"]())
    assert first == second

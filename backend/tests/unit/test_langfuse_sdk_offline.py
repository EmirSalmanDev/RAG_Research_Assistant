"""The adapter on the REAL Langfuse SDK, offline: spans go to an in-memory exporter
instead of Langfuse Cloud (the session-wide network guard would stop any lookup).
This checks what the fakes in test_langfuse_tracer.py can only assume: what the SDK
actually exports for one chat turn."""

import uuid
from collections.abc import Iterator
from typing import Any

import pytest

from app.config import get_settings
from app.observability import langfuse_tracer as lt
from app.rag.pipeline import run_chat_turn
from tests.unit.test_pipeline import CONVERSATION, HISTORY, full_turn, harness

STAGES = ["condense", "plan", "search", "rerank", "generate", "verify"]


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, Any]]:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    # The SDK keeps one client per public key: a fresh invented key per test.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"pk-lf-test-{uuid.uuid4().hex[:8]}")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test-invented")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.invalid")
    get_settings.cache_clear()
    exporter = InMemorySpanExporter()
    client = lt.make_langfuse_client(get_settings(), span_exporter=exporter)
    yield client, exporter
    client.shutdown()


def test_one_turn_exports_exactly_six_root_spans_in_one_trace(sdk: tuple[Any, Any]) -> None:
    from langfuse import propagate_attributes

    client, exporter = sdk
    h = harness(full_turn())
    h.deps.tracer = lt.LangfuseTracer(client, propagate_attributes)
    events = list(
        run_chat_turn(
            "and its limits?",
            conversation_id=CONVERSATION,
            new_conversation=False,
            history=HISTORY,
            deps=h.deps,
            settings=get_settings(),
        )
    )
    spans = sorted(exporter.get_finished_spans(), key=lambda s: s.start_time or 0)
    assert [s.name for s in spans] == STAGES  # nothing else was exported
    trace_id = events[-1].data["trace_id"]
    assert {format(s.context.trace_id, "032x") for s in spans} == {trace_id}
    attrs = [dict(s.attributes or {}) for s in spans]
    assert all(a["langfuse.internal.as_root"] is True for a in attrs)
    assert all(a["langfuse.trace.name"] == "chat-turn" for a in attrs)
    assert all(a["session.id"] == str(CONVERSATION) for a in attrs)
    rerank = attrs[STAGES.index("rerank")]
    assert rerank["langfuse.observation.level"] == "DEBUG"
    assert rerank["langfuse.observation.status_message"] == "skipped: disabled"
    # Neither key appears anywhere in what would be exported.
    text = repr(attrs)
    assert "sk-lf-test-invented" not in text and "pk-lf-test" not in text

"""The Langfuse adapter against a fake SDK client (no network), redaction, and when
get_tracer picks Langfuse. The real SDK client is never constructed here."""

import contextlib
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from app.config import get_settings
from app.observability import langfuse_tracer as lt
from app.observability import tracing
from app.observability.redact import redact_secrets, sanitize
from app.observability.tracing import NoopTracer, get_tracer
from app.rag.pipeline import describe, run_chat_turn
from tests.unit.test_pipeline import CONVERSATION, HISTORY, full_turn, harness

STAGES = ["condense", "plan", "search", "rerank", "generate", "verify"]


@dataclass
class FakeObservation:
    name: str
    start_kwargs: dict[str, Any]
    updates: list[dict[str, Any]] = field(default_factory=list)
    ends: list[int | None] = field(default_factory=list)

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self, *, end_time: int | None = None) -> None:
        self.ends.append(end_time)


@dataclass
class FakeLangfuse:
    observations: list[FakeObservation] = field(default_factory=list)
    flushes: int = 0
    fail_start: bool = False
    fail_flush: bool = False

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if self.fail_start:
            raise RuntimeError("sdk broke")
        obs = FakeObservation(kwargs["name"], kwargs)
        self.observations.append(obs)
        return obs

    def flush(self) -> None:
        self.flushes += 1
        if self.fail_flush:
            raise RuntimeError("export failed")


@dataclass
class FakePropagate:
    calls: list[dict[str, Any]] = field(default_factory=list)

    @contextlib.contextmanager
    def __call__(self, **kwargs: Any) -> Iterator[None]:
        self.calls.append(kwargs)
        yield


def _tracer() -> tuple[lt.LangfuseTracer, FakeLangfuse, FakePropagate]:
    client, propagate = FakeLangfuse(), FakePropagate()
    return lt.LangfuseTracer(client, propagate), client, propagate


def test_a_chat_turn_sends_exactly_six_spans_in_one_trace_and_flushes_once() -> None:
    tracer, client, propagate = _tracer()
    h = harness(full_turn())
    h.deps.tracer = tracer
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
    assert events[-1].event == "done"
    assert [o.name for o in client.observations] == STAGES  # no root span, nothing else
    trace_ids = {o.start_kwargs["trace_context"]["trace_id"] for o in client.observations}
    (trace_id,) = trace_ids
    assert len(trace_id) == 32 and int(trace_id, 16) >= 0
    assert events[-1].data["trace_id"] == trace_id == h.store.saved[0].trace_id
    assert all(o.start_kwargs["as_type"] == "span" for o in client.observations)
    assert client.flushes == 1
    # Every span carries the trace name and the session (the conversation).
    assert len(propagate.calls) == 6
    assert propagate.calls[0]["trace_name"] == "chat-turn"
    assert propagate.calls[0]["session_id"] == str(CONVERSATION)
    # Every span is ended exactly once, at the trace's end, with a recorded end time.
    assert all(len(o.ends) == 1 and isinstance(o.ends[0], int) for o in client.observations)
    rerank = next(o for o in client.observations if o.name == "rerank")
    assert rerank.updates[0]["level"] == "DEBUG"
    assert rerank.updates[0]["status_message"] == "skipped: disabled"
    verify = next(o for o in client.observations if o.name == "verify")
    summary = verify.updates[-1]["metadata"]["turn"]
    assert summary["outcome"] == {"status": "answered"}
    assert summary["prompts"]["generate"] == "generate_v2"


def test_span_end_times_are_the_moments_the_stages_finished() -> None:
    tracer, client, _ = _tracer()
    trace = tracer.start_trace("chat-turn", input={}, metadata={})
    first = trace.span("condense")
    first.end(output="q")
    recorded = first.ended_ns  # type: ignore[attr-defined]
    trace.span("plan").end(output={})
    assert client.observations[0].ends == []  # not closed until the trace ends
    trace.end(output={"status": "answered"})
    assert client.observations[0].ends == [recorded]


def test_errors_and_usage_are_recorded_on_the_span() -> None:
    tracer, client, _ = _tracer()
    trace = tracer.start_trace("chat-turn", input={}, metadata={"conversation_id": "c"})
    trace.span("plan").end(error="BadRequestError: nope", usage={"tokens_in": 3})
    update = client.observations[0].updates[0]
    assert update["level"] == "ERROR" and update["status_message"] == "BadRequestError: nope"
    assert update["metadata"]["usage"] == {"tokens_in": 3}


def test_span_data_is_sanitised_before_it_reaches_the_sdk() -> None:
    tracer, client, propagate = _tracer()
    trace = tracer.start_trace("chat-turn", input={}, metadata={"note": "Bearer abc.def"})
    span = trace.span(
        "search",
        input={"q": "uses sk-proj-ABCDEFGH1234 here", "img": "data:image/png;base64,iVBORw0KGgo="},
    )
    span.end(error="AuthenticationError: Incorrect API key provided: sk-proj-****abcd.")
    sent = client.observations[0]
    assert sent.start_kwargs["input"] == {"q": "uses [redacted] here", "img": "[data URI omitted]"}
    assert "sk-proj" not in sent.updates[0]["status_message"]
    assert propagate.calls[0]["metadata"] == {"note": "[redacted]"}


def test_sdk_failures_never_break_the_turn() -> None:
    tracer, client, _ = _tracer()
    client.fail_start, client.fail_flush = True, True
    h = harness(full_turn())
    h.deps.tracer = tracer
    events = list(
        run_chat_turn(
            "q",
            conversation_id=CONVERSATION,
            new_conversation=True,
            history=HISTORY,
            deps=h.deps,
            settings=get_settings(),
        )
    )
    assert events[-1].event == "done" and client.flushes == 1


# --- configuration -------------------------------------------------------------------------------


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-invented")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-invented")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.example")
    get_settings.cache_clear()
    tracing._langfuse_tracer.cache_clear()
    yield
    tracing._langfuse_tracer.cache_clear()


def test_without_keys_tracing_is_off() -> None:
    assert isinstance(get_tracer(), NoopTracer)  # tests blank both keys


def test_with_keys_get_tracer_builds_the_adapter_once(
    keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []

    def fake_make(settings: Any) -> lt.LangfuseTracer:
        built.append(settings.langfuse_host)
        return _tracer()[0]

    monkeypatch.setattr(lt, "make_langfuse_tracer", fake_make)
    first, second = get_tracer(), get_tracer()
    assert isinstance(first, lt.LangfuseTracer) and first is second
    assert built == ["https://langfuse.example"]


def test_a_failing_sdk_start_falls_back_to_no_tracing(
    keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(settings: Any) -> lt.LangfuseTracer:
        raise RuntimeError("cannot start")

    monkeypatch.setattr(lt, "make_langfuse_tracer", broken)
    assert isinstance(get_tracer(), NoopTracer)


def test_the_sdk_client_gets_every_setting_explicitly(
    keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langfuse
    from langfuse.span_filter import is_langfuse_span
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider

    seen: dict[str, Any] = {}
    monkeypatch.setattr(langfuse, "Langfuse", lambda **kw: seen.update(kw) or object())
    global_before = otel_trace.get_tracer_provider()
    lt.make_langfuse_client(get_settings())
    assert seen["public_key"] == "pk-lf-invented" and seen["secret_key"] == "sk-lf-invented"
    assert seen["base_url"] == "https://langfuse.example"  # beats LANGFUSE_BASE_URL
    assert seen["debug"] is False and seen["sample_rate"] == 1.0
    assert isinstance(seen["tracer_provider"], TracerProvider)
    assert seen["tracer_provider"] is not otel_trace.get_tracer_provider()
    assert otel_trace.get_tracer_provider() is global_before  # the global one is untouched
    assert seen["should_export_span"] is is_langfuse_span


# --- redaction -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("key sk-proj-abcdEFGH1234_xyz end", "key [redacted] end"),
        (
            "Incorrect API key provided: sk-proj-****************abcd.",
            "Incorrect API key provided: [redacted].",
        ),
        ("pk-lf-1234-abcd and sk-lf-9876-dcba", "[redacted] and [redacted]"),
        ("Authorization: Bearer eyJhbGciOi.payload.sig", "Authorization: [redacted]"),
        ("url?api_key=SECRET123&x=1", "url?[redacted]&x=1"),
        ("scikit-learn and sk-learn stay", "scikit-learn and sk-learn stay"),
    ],
)
def test_redact_secrets(text: str, expected: str) -> None:
    assert redact_secrets(text) == expected


def test_sanitize_cuts_long_strings_and_stringifies_objects() -> None:
    out = sanitize({"long": "x" * 20_010, "id": uuid.UUID(int=1), "n": [1, 2.5, None, True]})
    assert out["long"].endswith("[... 10 chars cut]") and len(out["long"]) < 20_100
    assert out["id"] == "00000000-0000-0000-0000-000000000001"
    assert out["n"] == [1, 2.5, None, True]


def test_describe_redacts_keys_in_stored_error_texts() -> None:
    err = RuntimeError("Incorrect API key provided: sk-proj-****abcd1234.")
    assert describe(err) == "RuntimeError: Incorrect API key provided: [redacted]."

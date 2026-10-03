"""The tracing interface: the no-op tracer, and the recording fake satisfies the protocol."""

from app.observability.tracing import NoopTracer, Tracer, get_tracer
from tests.tracing_fakes import RecordingTracer


def test_the_noop_tracer_accepts_everything_and_has_no_trace_id() -> None:
    trace = NoopTracer().start_trace("chat-turn", input="q", metadata={"prompt": "x"})
    span = trace.span("search", input={"q": "q"})
    span.end(output=[1], usage={"tokens_in": 1}, skipped="disabled")
    trace.end(output="answer")
    assert trace.id is None


def test_get_tracer_is_a_noop_until_the_langfuse_adapter_exists() -> None:
    assert isinstance(get_tracer(), NoopTracer)


def test_the_recording_fake_has_the_tracer_shape() -> None:
    tracer: Tracer = RecordingTracer()  # mypy checks the structural match
    trace = tracer.start_trace("chat-turn", input="q", metadata={})
    trace.span("plan").end(error="ChatResponseError")
    assert trace.id == "trace-0001"

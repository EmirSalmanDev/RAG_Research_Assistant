"""Tracing: one trace per chat turn, one span per pipeline stage (docs/adr/0006).

The pipeline talks to this small interface, not to a tracing SDK, so that:
- tests use a recording fake and check the spans without any network;
- tracing is a no-op whenever it is not configured (no Langfuse keys, or no SDK
  installed), and the chat works the same either way.

The Langfuse adapter (Langfuse Cloud, host from LANGFUSE_HOST) is added together with
the Langfuse SDK dependency; until then `get_tracer()` always returns the no-op tracer.

Spans are started and ended explicitly, not with `with` blocks: the answer is streamed
from a generator that pauses between pieces, and a stage's span must stay open across
those pauses.

Every turn records the same six spans: condense, plan, search, rerank, generate, verify.
A stage that did not run still gets its span, ended with `skipped` and the reason, so
every trace has the same shape and a missing stage is visible rather than absent.
"""

from collections.abc import Mapping
from typing import Any, Protocol


class Span(Protocol):
    def end(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        skipped: str | None = None,
    ) -> None:
        """Close the span. `error` marks a failed stage, `skipped` one that did not run."""
        ...


class Trace(Protocol):
    @property
    def id(self) -> str | None:
        """The trace id stored on the message (None when tracing is off)."""
        ...

    def span(self, name: str, *, input: Any = None) -> Span: ...

    def end(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None: ...


class Tracer(Protocol):
    def start_trace(self, name: str, *, input: Any, metadata: Mapping[str, Any]) -> Trace: ...


class _NoopSpan:
    def end(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        skipped: str | None = None,
    ) -> None:
        return None


class _NoopTrace:
    @property
    def id(self) -> str | None:
        return None

    def span(self, name: str, *, input: Any = None) -> Span:
        return _NoopSpan()

    def end(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None:
        return None


class NoopTracer:
    """Records nothing; used when tracing is not configured."""

    def start_trace(self, name: str, *, input: Any, metadata: Mapping[str, Any]) -> Trace:
        return _NoopTrace()


def get_tracer() -> Tracer:
    """The process's tracer. No-op until the Langfuse adapter is added (see above)."""
    return NoopTracer()

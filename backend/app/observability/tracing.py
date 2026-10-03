"""Tracing: one trace per chat turn, one span per pipeline stage (docs/adr/0006).

The pipeline talks to this small interface, not to a tracing SDK, so that:
- tests use a recording fake and check the spans without any network;
- tracing is a no-op whenever it is not configured (no Langfuse keys, or no SDK
  installed), and the chat works the same either way.

`get_tracer()` returns the Langfuse adapter (app/observability/langfuse_tracer.py, host
from LANGFUSE_HOST) only when both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set;
otherwise, and if the SDK cannot start, the no-op tracer. Tests blank both keys.

Spans are started and ended explicitly, not with `with` blocks: the answer is streamed
from a generator that pauses between pieces, and a stage's span must stay open across
those pauses.

Every turn records the same six spans: condense, plan, search, rerank, generate, verify.
A stage that did not run still gets its span, ended with `skipped` and the reason, so
every trace has the same shape and a missing stage is visible rather than absent.
"""

import logging
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Protocol

from app.config import get_settings

logger = logging.getLogger(__name__)


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
    """Langfuse when both keys are set, else the no-op tracer (see the module doc)."""
    s = get_settings()
    if s.langfuse_public_key is None or s.langfuse_secret_key is None:
        return NoopTracer()
    return _langfuse_tracer(s.langfuse_host, s.langfuse_public_key.get_secret_value())


@lru_cache(maxsize=1)
def _langfuse_tracer(host: str, public_key: str) -> Tracer:
    """One SDK client per process (its background threads start once). The arguments
    only key the cache; the client reads them from the settings."""
    try:
        from app.observability.langfuse_tracer import make_langfuse_tracer

        return make_langfuse_tracer(get_settings())
    except Exception:
        logger.warning("langfuse could not start; tracing is off", exc_info=True)
        return NoopTracer()

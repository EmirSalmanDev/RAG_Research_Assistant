"""The Tracer interface on Langfuse (SDK 4.15, OpenTelemetry-based), for Langfuse Cloud.

What one chat turn sends: exactly the six stage spans, in one trace, and nothing else.
- The trace id is made here (32 hex characters). Each stage span is started with
  `trace_context={"trace_id": ...}`, so Langfuse shows it as a top-level observation of
  that trace: no extra root span. The trace's name ("chat-turn") and session (the
  conversation id) are attached to every span through `propagate_attributes`.
- The SDK gets a PRIVATE OpenTelemetry TracerProvider. Without one it would install its
  own as the process-wide default, and spans that other libraries emit through the
  OpenTelemetry API would then reach it. `should_export_span=is_langfuse_span` exports
  only spans made by this SDK.
- Every value is passed through `sanitize` first: no keys, tokens or headers, no base64
  data URIs (the SDK would upload those as media files), no huge strings. The keys go
  to the SDK as parameters and are never part of span data.
- One flush at the end of each turn (one export request in the usual case).

Settings are passed explicitly, so environment variables cannot redirect the export:
`base_url` beats LANGFUSE_BASE_URL / LANGFUSE_HOST, and keys, timeout, debug, sample rate
and environment are fixed here. The SDK still reads two switches only from the
environment: LANGFUSE_TRACING_ENABLED=false (turns tracing off) and
LANGFUSE_MEDIA_UPLOAD_ENABLED (irrelevant: no media is ever sent).

A span's end is recorded when the stage finishes, but the OpenTelemetry span is closed
only when the trace ends (with that recorded end time). That lets the turn's summary
(outcome, prompt versions, skipped stages, errors) be added to the last stage's span,
since a trace without a root span has no other place for it.

Tracing must never break a chat turn: every call into the SDK is guarded, and a failure
is logged and dropped.
"""

import logging
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings
from app.observability.redact import sanitize
from app.observability.tracing import Span, Trace

logger = logging.getLogger(__name__)

TRACE_NAME = "chat-turn"
EXPORT_TIMEOUT_S = 10
SUMMARY_SPAN = "verify"  # the last stage: the turn's summary goes on its metadata

PropagateFn = Callable[..., AbstractContextManager[Any]]


def _guard(action: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        logger.warning("langfuse: %s failed; the chat turn goes on", action, exc_info=True)
        return None


@dataclass
class _LangfuseSpan:
    observation: Any  # a LangfuseSpan from the SDK (or a test fake)
    name: str
    ended_ns: int | None = None

    def end(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        skipped: str | None = None,
    ) -> None:
        self.ended_ns = time.time_ns()
        meta: dict[str, Any] = dict(metadata or {})
        if usage is not None:
            meta["usage"] = dict(usage)
        if skipped is not None:
            meta["skipped"] = skipped
        update: dict[str, Any] = {"metadata": sanitize(meta)}
        if output is not None:
            update["output"] = sanitize(output)
        if error is not None:
            update.update(level="ERROR", status_message=sanitize(error))
        elif skipped is not None:
            update.update(level="DEBUG", status_message=sanitize(f"skipped: {skipped}"))
        _guard(f"update span {self.name}", lambda: self.observation.update(**update))


@dataclass
class _LangfuseTrace:
    client: Any
    propagate: PropagateFn
    trace_attributes: dict[str, Any]
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[_LangfuseSpan] = field(default_factory=list)

    @property
    def id(self) -> str | None:
        return self.trace_id

    def span(self, name: str, *, input: Any = None) -> Span:
        def start() -> Any:
            with self.propagate(**self.trace_attributes):
                return self.client.start_observation(
                    trace_context={"trace_id": self.trace_id},
                    name=name,
                    as_type="span",
                    input=sanitize(input),
                )

        span = _LangfuseSpan(_guard(f"start span {name}", start) or _NullObservation(), name)
        self.spans.append(span)
        return span

    def end(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None:
        summary = sanitize({"turn": {"outcome": output, **dict(metadata or {})}})
        for span in self.spans:
            if span.name == SUMMARY_SPAN:
                _guard("add the turn summary", _updater(span.observation, summary))
        for span in self.spans:
            end_ns = span.ended_ns or time.time_ns()
            _guard(f"end span {span.name}", _ender(span.observation, end_ns))
        _guard("flush", self.client.flush)


def _updater(observation: Any, metadata: Any) -> Callable[[], Any]:
    return lambda: observation.update(metadata=metadata)


def _ender(observation: Any, end_ns: int) -> Callable[[], Any]:
    return lambda: observation.end(end_time=end_ns)


class _NullObservation:
    """Stands in when the SDK failed to start a span, so the turn can go on."""

    def update(self, **_: Any) -> None:
        return None

    def end(self, **_: Any) -> None:
        return None


class LangfuseTracer:
    def __init__(self, client: Any, propagate: PropagateFn) -> None:
        self._client = client
        self._propagate = propagate

    def start_trace(self, name: str, *, input: Any, metadata: Mapping[str, Any]) -> Trace:
        # Propagated values must be short strings; the input goes on the spans themselves.
        attributes: dict[str, Any] = {"trace_name": TRACE_NAME}
        conversation = metadata.get("conversation_id")
        if conversation:
            attributes["session_id"] = str(conversation)
        attributes["metadata"] = {k: str(v)[:200] for k, v in sanitize(dict(metadata)).items()}
        return _LangfuseTrace(self._client, self._propagate, attributes)


def make_langfuse_client(settings: Settings, *, span_exporter: Any = None) -> Any:
    """The SDK client, configured only from our settings (see the module doc).
    `span_exporter` is for tests: an in-memory exporter instead of the export to
    `{LANGFUSE_HOST}/api/public/otel/v1/traces`."""
    from langfuse import Langfuse
    from langfuse.span_filter import is_langfuse_span
    from opentelemetry.sdk.trace import TracerProvider

    assert settings.langfuse_public_key is not None and settings.langfuse_secret_key is not None
    return Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_host,
        timeout=EXPORT_TIMEOUT_S,
        debug=False,
        tracing_enabled=True,
        sample_rate=1.0,
        environment="local",
        tracer_provider=TracerProvider(),  # private: never the global provider
        should_export_span=is_langfuse_span,
        span_exporter=span_exporter,
    )


def make_langfuse_tracer(settings: Settings) -> LangfuseTracer:
    from langfuse import propagate_attributes

    return LangfuseTracer(make_langfuse_client(settings), propagate_attributes)

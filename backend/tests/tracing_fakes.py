"""A Tracer that records traces and spans in memory, for tests (no network)."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecordedSpan:
    name: str
    input: Any
    ended: bool = False
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    error: str | None = None
    skipped: str | None = None

    def end(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        skipped: str | None = None,
    ) -> None:
        assert not self.ended, f"span {self.name!r} ended twice"
        self.ended = True
        self.output, self.metadata = output, dict(metadata or {})
        self.usage = dict(usage) if usage is not None else None
        self.error, self.skipped = error, skipped


@dataclass
class RecordedTrace:
    name: str
    input: Any
    metadata: dict[str, Any]
    spans: list[RecordedSpan] = field(default_factory=list)
    ended: bool = False
    output: Any = None

    @property
    def id(self) -> str | None:
        return "trace-0001"

    def span(self, name: str, *, input: Any = None) -> RecordedSpan:
        span = RecordedSpan(name, input)
        self.spans.append(span)
        return span

    def end(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None) -> None:
        self.ended, self.output = True, output
        self.metadata.update(metadata or {})

    def span_named(self, name: str) -> RecordedSpan:
        (span,) = [s for s in self.spans if s.name == name]
        return span


@dataclass
class RecordingTracer:
    traces: list[RecordedTrace] = field(default_factory=list)

    def start_trace(self, name: str, *, input: Any, metadata: Mapping[str, Any]) -> RecordedTrace:
        trace = RecordedTrace(name, input, dict(metadata))
        self.traces.append(trace)
        return trace

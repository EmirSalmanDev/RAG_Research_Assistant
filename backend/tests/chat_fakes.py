"""A scripted stand-in for `openai.OpenAI().chat.completions` (no API calls).

Each request takes the next scripted reply: an exception to raise, a finished
completion (`text_reply`, `parsed_reply`) or a stream (`stream_reply`). The fake records
every request's keyword arguments, so tests can check what would have been sent.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx2
import openai

from app.llm.client import ChatClient
from app.llm.usage import chat_cost


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Delta:
    content: str | None


@dataclass
class _ChunkChoice:
    delta: _Delta


@dataclass
class Chunk:
    choices: list[_ChunkChoice]
    usage: Usage | None = None


@dataclass
class _Message:
    content: str | None = None
    parsed: Any = None
    refusal: str | None = None


@dataclass
class _Choice:
    message: _Message


@dataclass
class Completion:
    choices: list[_Choice]
    usage: Usage | None


class FakeRawStream:
    """What `create(stream=True)` returns: chunks to iterate, and `close()`."""

    def __init__(self, chunks: list[Chunk], fail_after: int | None = None) -> None:
        self.chunks = chunks
        self.fail_after = fail_after  # raise a connection error after this many chunks
        self.read = 0
        self.closed = False

    def __iter__(self) -> Any:
        for chunk in self.chunks:
            if self.fail_after is not None and self.read >= self.fail_after:
                raise openai.APIConnectionError(request=request())
            self.read += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


def text_reply(text: str, tokens_in: int = 100, tokens_out: int = 10) -> Completion:
    return Completion([_Choice(_Message(content=text))], Usage(tokens_in, tokens_out))


def parsed_reply(obj: Any, tokens_in: int = 100, tokens_out: int = 10) -> Completion:
    return Completion([_Choice(_Message(parsed=obj))], Usage(tokens_in, tokens_out))


def stream_reply(
    pieces: list[str], tokens_in: int = 1000, tokens_out: int = 50, fail_after: int | None = None
) -> FakeRawStream:
    """Text chunks, then the usage-only chunk the API sends last (no choices)."""
    chunks = [Chunk([_ChunkChoice(_Delta(p))]) for p in pieces]
    chunks.append(Chunk([], Usage(tokens_in, tokens_out)))
    return FakeRawStream(chunks, fail_after)


def request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.example/v1/chat/completions")


def status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    response = httpx2.Response(status, request=request(), headers={"retry-after": "0"})
    return cls("invented error", response=response, body=None)


@dataclass
class FakeCompletions:
    replies: list[Any]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _next(self, kind: str, kwargs: dict[str, Any]) -> Any:
        self.calls.append({"kind": kind, **kwargs})
        if not self.replies:
            raise AssertionError(f"unexpected {kind} request: no scripted reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def create(self, **kwargs: Any) -> Any:
        return self._next("stream" if kwargs.get("stream") else "create", kwargs)

    def parse(self, **kwargs: Any) -> Any:
        return self._next("parse", kwargs)


@dataclass
class _Chat:
    completions: FakeCompletions


@dataclass
class FakeOpenAI:
    chat: _Chat


UsageRows = list[tuple[str, str, int, int]]  # (feature, model, tokens_in, tokens_out)


def fake_chat(replies: list[Any]) -> tuple[ChatClient, FakeCompletions, UsageRows]:
    """A ChatClient on scripted replies; usage rows are collected in a list."""
    completions = FakeCompletions(list(replies))
    rows: UsageRows = []

    def record(feature: str, model: str, tokens_in: int, tokens_out: int) -> Decimal:
        rows.append((feature, model, tokens_in, tokens_out))
        return chat_cost(tokens_in, tokens_out)

    client = ChatClient(FakeOpenAI(_Chat(completions)), record, sleep=lambda s: None)
    return client, completions, rows

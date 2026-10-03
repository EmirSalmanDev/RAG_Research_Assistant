"""Chat calls to OpenAI (condense, plan, generate, verify): retries and usage in one place.

Three kinds of call:
- `complete`: one request, plain text back (condense).
- `parse`: one request with Structured Outputs: the API is given the JSON schema of a
  pydantic model and must answer in exactly that shape; the SDK parses it (plan, verify).
- `stream`: the answer arrives token by token (generate), so the user sees it grow.

Every successful request writes one `llm_usage` row (rule 9) with input and output
tokens and the cost at the configured prices. Transient failures (429, 5xx, network)
are retried by our own `call_with_retry`; the SDK's own retries are off (max_retries=0
in `default_openai_client`), so attempts are never multiplied.

A stream is retried only while it is being opened. Once tokens have been sent to the
user, a retry would repeat the answer, so a failure mid-stream is raised instead.
"""

from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import openai
from pydantic import BaseModel

from app.config import get_settings

# Imported from ingestion, where the embedding client was built first (M2). Moving it to
# app/llm/ would change embed.py; it is left there on purpose for now.
from app.ingest.embed import default_openai_client
from app.llm.usage import ChatUsageRecorder, record_chat_usage
from app.sources.http import RetryableError, call_with_retry

ChatMessage = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


@dataclass(frozen=True)
class ChatUsage:
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal


class ChatResponseError(RuntimeError):
    """The API answered, but not with usable output (refused, cut off, or no usage).

    `usage` is set when the request was billed anyway, so the caller can still count it.
    """

    def __init__(self, message: str, usage: ChatUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def _retry_after(err: openai.APIStatusError) -> float | None:
    value = err.response.headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _as_retryable[T](fn: Callable[[], T]) -> Callable[[], T]:
    """Wrap one API request so transient failures become RetryableError.

    The same mapping as DenseEmbedder._request in app/ingest/embed.py (copied, not
    shared, so embed.py stays unchanged). Anything else (bad key, bad request) is raised
    at once and not retried.
    """

    def call() -> T:
        try:
            return fn()
        except openai.RateLimitError as err:
            raise RetryableError(str(err), status=429, retry_after=_retry_after(err)) from err
        except openai.InternalServerError as err:
            raise RetryableError(str(err), status=err.status_code) from err
        except openai.APIConnectionError as err:  # includes timeouts
            raise RetryableError(f"network error: {err}") from err

    return call


class ChatStream:
    """The text pieces of a streamed answer; `usage` is set once the stream has ended.

    With `stream_options={"include_usage": True}` the API sends the token counts in one
    last chunk after the text. If the reader stops early (the browser went away), `close`
    reads the rest of the stream anyway: the tokens are generated and billed either way,
    and only that last chunk tells us how many, so the cost can still be logged.
    """

    def __init__(self, raw: Iterable[Any], on_usage: Callable[[Any], ChatUsage]) -> None:
        self._raw = raw
        self._chunks: Iterator[Any] = iter(raw)
        self._on_usage = on_usage
        self._closed = False
        self.usage: ChatUsage | None = None

    def __iter__(self) -> Generator[str]:
        try:
            for chunk in self._chunks:
                self._take_usage(chunk)
                for choice in chunk.choices:
                    if choice.delta.content:
                        yield choice.delta.content
        finally:
            self.close()

    def _take_usage(self, chunk: Any) -> None:
        if chunk.usage is not None and self.usage is None:
            self.usage = self._on_usage(chunk.usage)

    def close(self) -> None:
        """Drain the rest of the stream (to log its usage), then release the connection."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.usage is None:
                for chunk in self._chunks:
                    self._take_usage(chunk)
        except Exception:  # the stream itself failed: the reader already got that error
            pass
        finally:
            close = getattr(self._raw, "close", None)
            if callable(close):
                close()


class ChatClient:
    """Chat requests with retries and usage logging. Tests pass a fake `client` with the
    shape of `openai.OpenAI().chat.completions` and a fake `record_usage`."""

    def __init__(
        self,
        client: Any = None,
        record_usage: ChatUsageRecorder | None = None,
        *,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        settings = get_settings()
        self._client = client
        self._record_usage = record_usage or record_chat_usage
        self.model = settings.chat_model
        self._temperature = settings.chat_temperature
        self._retry_kwargs: dict[str, Any] = {} if sleep is None else {"sleep": sleep}

    @property
    def client(self) -> Any:
        if self._client is None:  # created lazily: no key is needed until the first call
            self._client = default_openai_client()
        return self._client

    def _log(self, feature: str, usage: Any) -> ChatUsage:
        """Write the llm_usage row for one finished request (this is when money is spent)."""
        if usage is None:
            raise ChatResponseError("the response carried no token usage")
        tokens_in, tokens_out = int(usage.prompt_tokens), int(usage.completion_tokens)
        cost = self._record_usage(feature, self.model, tokens_in, tokens_out)
        return ChatUsage(tokens_in, tokens_out, cost)

    def complete(
        self, messages: Sequence[ChatMessage], *, feature: str, max_tokens: int
    ) -> tuple[str, ChatUsage]:
        """One request; the answer's text and what it cost."""
        completion = call_with_retry(
            _as_retryable(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=self._temperature,
                    max_completion_tokens=max_tokens,
                )
            ),
            **self._retry_kwargs,
        )
        usage = self._log(feature, completion.usage)
        text = completion.choices[0].message.content if completion.choices else None
        if not text or not text.strip():
            raise ChatResponseError("the model returned no text", usage)
        return text.strip(), usage

    def parse[M: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[M],
        *,
        feature: str,
        max_tokens: int,
    ) -> tuple[M, ChatUsage]:
        """One Structured Outputs request: the answer as an instance of `schema`."""
        try:
            completion = call_with_retry(
                _as_retryable(
                    lambda: self.client.chat.completions.parse(
                        model=self.model,
                        messages=list(messages),
                        response_format=schema,
                        temperature=self._temperature,
                        max_completion_tokens=max_tokens,
                    )
                ),
                **self._retry_kwargs,
            )
        except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as err:
            # The request succeeded (and was billed) but the JSON was cut off or filtered.
            billed = None
            if err.completion is not None and err.completion.usage is not None:
                billed = self._log(feature, err.completion.usage)
            raise ChatResponseError(f"no structured output: {type(err).__name__}", billed) from err
        usage = self._log(feature, completion.usage)
        message = completion.choices[0].message
        if message.parsed is None:
            raise ChatResponseError(f"no structured output (refusal: {message.refusal!r})", usage)
        parsed: M = message.parsed
        return parsed, usage

    def stream(
        self, messages: Sequence[ChatMessage], *, feature: str, max_tokens: int
    ) -> ChatStream:
        """Open a streamed request (retried only while opening); iterate it for the text."""
        raw = call_with_retry(
            _as_retryable(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=self._temperature,
                    max_completion_tokens=max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
            ),
            **self._retry_kwargs,
        )
        return ChatStream(raw, lambda usage: self._log(feature, usage))

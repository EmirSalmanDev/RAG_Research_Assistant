"""app/llm/client.py and the chat cost: against a scripted fake OpenAI (no API calls)."""

from decimal import Decimal

import openai
import pytest
from pydantic import BaseModel

from app.ingest.tokens import count_tokens
from app.llm.client import MESSAGE_OVERHEAD_TOKENS, ChatResponseError
from app.llm.usage import chat_cost
from app.sources.http import RetryableError
from tests.chat_fakes import (
    Completion,
    fake_chat,
    parsed_reply,
    request,
    status_error,
    stream_reply,
    text_reply,
)

MESSAGES = [{"role": "user", "content": "invented question"}]


class Answer(BaseModel):
    value: str


def test_chat_cost_uses_input_and_output_prices() -> None:
    # 0.15 USD per 1M input tokens, 0.60 USD per 1M output tokens.
    assert chat_cost(1_000_000, 0) == Decimal("0.150000")
    assert chat_cost(0, 1_000_000) == Decimal("0.600000")
    assert chat_cost(4000, 500) == Decimal("0.000900")  # 0.0006 + 0.0003


def test_complete_returns_text_and_logs_one_row() -> None:
    client, calls, rows = fake_chat([text_reply("  standalone question  ", 120, 8)])
    text, usage = client.complete(MESSAGES, feature="chat-condense", max_tokens=50)
    assert text == "standalone question"
    assert rows == [("chat-condense", "gpt-4o-mini", 120, 8)]
    assert usage.cost_usd == chat_cost(120, 8)
    sent = calls.calls[0]
    assert sent["model"] == "gpt-4o-mini" and sent["temperature"] == 0.0
    assert sent["max_completion_tokens"] == 50


def test_parse_returns_the_model_instance() -> None:
    client, calls, rows = fake_chat([parsed_reply(Answer(value="x"))])
    answer, _ = client.parse(MESSAGES, Answer, feature="chat-plan", max_tokens=100)
    assert answer == Answer(value="x")
    assert calls.calls[0]["response_format"] is Answer
    assert [r[0] for r in rows] == ["chat-plan"]


def test_parse_refusal_is_an_error_but_still_billed() -> None:
    client, _, rows = fake_chat([parsed_reply(None, 100, 10)])
    with pytest.raises(ChatResponseError) as err:
        client.parse(MESSAGES, Answer, feature="chat-plan", max_tokens=100)
    assert len(rows) == 1
    assert err.value.usage is not None and err.value.usage.cost_usd == chat_cost(100, 10)


def test_parse_cut_off_json_logs_its_usage_and_raises() -> None:
    cut = openai.LengthFinishReasonError(completion=text_reply("{", 300, 100))  # type: ignore[arg-type]
    client, _, rows = fake_chat([cut])
    with pytest.raises(ChatResponseError) as err:
        client.parse(MESSAGES, Answer, feature="chat-verify", max_tokens=100)
    assert rows == [("chat-verify", "gpt-4o-mini", 300, 100)]
    assert err.value.usage is not None and err.value.usage.tokens_out == 100


def test_missing_usage_is_an_error() -> None:
    client, _, _ = fake_chat([Completion(text_reply("x").choices, None)])
    with pytest.raises(ChatResponseError):
        client.complete(MESSAGES, feature="chat-condense", max_tokens=10)


def test_transient_errors_are_retried_and_only_success_is_logged() -> None:
    error = status_error(openai.RateLimitError, 429)
    client, calls, rows = fake_chat(
        [error, openai.APIConnectionError(request=request()), text_reply("ok")]
    )
    text, _ = client.complete(MESSAGES, feature="chat-condense", max_tokens=10)
    assert text == "ok" and len(calls.calls) == 3 and len(rows) == 1


def test_gives_up_after_five_attempts() -> None:
    error = status_error(openai.InternalServerError, 503)
    client, calls, rows = fake_chat([error] * 5)
    with pytest.raises(RetryableError):
        client.complete(MESSAGES, feature="chat-condense", max_tokens=10)
    assert len(calls.calls) == 5 and rows == []


def test_bad_request_is_not_retried() -> None:
    client, calls, _ = fake_chat([status_error(openai.BadRequestError, 400)])
    with pytest.raises(openai.BadRequestError):
        client.complete(MESSAGES, feature="chat-condense", max_tokens=10)
    assert len(calls.calls) == 1


def test_stream_yields_text_then_logs_usage_from_the_last_chunk() -> None:
    raw = stream_reply(["Attention ", "is ", "all [1]."], 2000, 30)
    client, calls, rows = fake_chat([raw])
    stream = client.stream(MESSAGES, feature="chat-generate", max_tokens=1000)
    assert "".join(stream) == "Attention is all [1]."
    assert stream.usage is not None and stream.usage.tokens_in == 2000
    assert rows == [("chat-generate", "gpt-4o-mini", 2000, 30)]
    assert raw.closed
    sent = calls.calls[0]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}


def test_stream_opening_is_retried() -> None:
    client, calls, rows = fake_chat([status_error(openai.RateLimitError, 429), stream_reply(["a"])])
    assert "".join(client.stream(MESSAGES, feature="chat-generate", max_tokens=10)) == "a"
    assert len(calls.calls) == 2 and len(rows) == 1


def test_a_reader_that_stops_early_still_gets_usage_logged() -> None:
    """The browser went away after the first token: the rest is drained for its usage."""
    raw = stream_reply(["one ", "two ", "three"], 900, 3)
    client, _, rows = fake_chat([raw])
    stream = client.stream(MESSAGES, feature="chat-generate", max_tokens=10)
    pieces = iter(stream)
    assert next(pieces) == "one "
    pieces.close()  # what happens to a generator the reader abandons
    assert rows == [("chat-generate", "gpt-4o-mini", 900, 3)]
    assert raw.read == len(raw.chunks) and raw.closed


def test_close_before_reading_drains_and_logs() -> None:
    raw = stream_reply(["a", "b"], 10, 2)
    client, _, rows = fake_chat([raw])
    client.stream(MESSAGES, feature="chat-generate", max_tokens=10).close()
    assert len(rows) == 1 and raw.closed


def test_a_failure_mid_stream_is_raised_not_retried() -> None:
    raw = stream_reply(["a", "b", "c"], fail_after=1)
    client, calls, rows = fake_chat([raw])
    stream = client.stream(MESSAGES, feature="chat-generate", max_tokens=10)
    got: list[str] = []
    with pytest.raises(openai.APIConnectionError):
        for piece in stream:
            got.append(piece)
    assert got == ["a"] and len(calls.calls) == 1 and raw.closed
    # The usage chunk never arrived: one estimated row instead (checked below).
    assert [r[0] for r in rows] == ["chat-generate-estimated"]


def test_a_broken_stream_logs_an_estimate_from_the_prompt_and_the_text_received() -> None:
    messages = [
        {"role": "system", "content": "You answer from the sources."},
        {
            "role": "user",
            "content": "<question>\nWhat does the invented paper propose?\n</question>",
        },
    ]
    raw = stream_reply(["The invented paper ", "proposes a ", "method"], fail_after=2)
    client, _, rows = fake_chat([raw])
    stream = client.stream(messages, feature="chat-generate", max_tokens=100)
    with pytest.raises(openai.APIConnectionError):
        list(stream)
    tokens_in = sum(count_tokens(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages)
    tokens_out = count_tokens("The invented paper proposes a ")  # only what arrived
    assert rows == [("chat-generate-estimated", "gpt-4o-mini", tokens_in, tokens_out)]
    assert stream.estimated and stream.usage is not None
    assert stream.usage.cost_usd == chat_cost(tokens_in, tokens_out)


def test_a_complete_stream_is_never_estimated() -> None:
    client, _, rows = fake_chat([stream_reply(["ok"], 10, 1)])
    stream = client.stream(MESSAGES, feature="chat-generate", max_tokens=10)
    assert "".join(stream) == "ok" and not stream.estimated
    assert [r[0] for r in rows] == ["chat-generate"]

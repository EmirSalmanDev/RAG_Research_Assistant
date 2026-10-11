"""Condense stage: turn a follow-up message into a standalone question.

"And what are its limitations?" cannot be searched on its own: "its" points back into
the conversation. One LLM call rewrites it, using the last N messages, into "What are
the limitations of the Transformer (1706.03762)?". This is the chat's only memory: there
is no running summary, so anything older than N messages is not seen.

A first message has no history, so no call is made (the message is already standalone).
The stage has a switch (`condense_enabled`) so the evaluation can measure what it adds.
Errors are raised; the pipeline falls back to the raw message (docs/adr/0006).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from app.llm.client import ChatClient, ChatMessage, ChatResponseError, ChatUsage
from app.rag.prompts import escape_block, load_prompt

CONDENSE_PROMPT = "condense_v1"
FEATURE = "chat-condense"  # llm_usage.feature
MAX_OUTPUT_TOKENS = 200  # a question, not an answer
HISTORY_MESSAGE_CHARS = 2000  # each earlier message is cut to this, to bound the cost
MAX_QUESTION_CHARS = 1000  # a longer "question" means the model did something else


@dataclass(frozen=True)
class HistoryMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class CondenseResult:
    question: str  # the standalone question (the message itself when skipped)
    skipped: str | None = None  # why no call was made, else None
    usage: ChatUsage | None = None
    prompt: str | None = None  # prompt id, when a call was made


def build_messages(message: str, history: Sequence[HistoryMessage]) -> list[ChatMessage]:
    """System prompt, then the conversation and the latest message as data blocks."""
    turns = "\n".join(
        f'<message role="{h.role}">\n{escape_block(h.content[:HISTORY_MESSAGE_CHARS])}\n</message>'
        for h in history
    )
    user = (
        f"<conversation>\n{turns}\n</conversation>\n\n<latest>\n{escape_block(message)}\n</latest>"
    )
    return [
        {"role": "system", "content": load_prompt(CONDENSE_PROMPT).text},
        {"role": "user", "content": user},
    ]


def condense(
    message: str,
    history: Sequence[HistoryMessage],
    *,
    client: ChatClient,
    enabled: bool,
    max_history: int,
) -> CondenseResult:
    """The standalone form of `message`, given the last `max_history` messages."""
    if not enabled:
        return CondenseResult(message, skipped="disabled")
    recent = list(history)[-max_history:] if max_history > 0 else []
    if not recent:
        return CondenseResult(message, skipped="no history")
    text, usage = client.complete(
        build_messages(message, recent), feature=FEATURE, max_tokens=MAX_OUTPUT_TOKENS
    )
    if len(text) > MAX_QUESTION_CHARS:
        raise ChatResponseError(f"condensed question is {len(text)} characters long", usage)
    return CondenseResult(text, usage=usage, prompt=CONDENSE_PROMPT)

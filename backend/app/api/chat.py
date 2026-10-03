"""POST /chat: ask a question, get a cited answer streamed as server-sent events (SSE).

Example (new conversation; -N stops curl from buffering the stream):

    curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \\
         -d '{"message": "How does LoRA reduce the number of trained parameters?"}'

A follow-up sends the `conversation_id` from the first `meta` event. The events and
their order are described in app/rag/pipeline.py; the stream ends with `done` or `error`.

Everything that can be refused is checked BEFORE the stream starts, so it comes back as
a normal HTTP error: 422 for a blank message, 404 for an unknown conversation, 503 when
OPENAI_API_KEY is missing. Once the stream has started the status is 200, and a failure
arrives as an `error` event instead.

The pipeline is synchronous (OpenAI, Qdrant and Postgres clients are blocking calls),
so each step of the generator runs in a worker thread and the event loop stays free.
If the browser disconnects, the generator is closed: its spans are ended, the answer's
token usage is still logged, and nothing is saved for the unfinished turn.
"""

import json
import uuid
from collections.abc import AsyncGenerator, Generator
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.sse import EventSourceResponse, format_sse_event
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app.config import get_settings
from app.rag.condense import HistoryMessage
from app.rag.conversations import ConversationNotFound
from app.rag.pipeline import ChatDeps, ChatEvent, run_chat_turn

router = APIRouter()

MAX_MESSAGE_CHARS = 2000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: uuid.UUID | None = None  # None starts a new conversation


def get_chat_deps() -> ChatDeps:
    """The real clients, store and tracer (tests override this dependency)."""
    return ChatDeps.default()


async def _sse(events: Generator[ChatEvent]) -> AsyncGenerator[bytes]:
    try:
        async for event in iterate_in_threadpool(events):
            yield format_sse_event(event=event.event, data_str=json.dumps(event.data))
    finally:
        # On a disconnect this task is being cancelled; the shield lets the close (ending
        # spans, logging usage) finish in its thread instead of being cancelled as well.
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(events.close)


# Not `response_class=EventSourceResponse`: FastAPI would then expect a generator route.
# This route checks the request first and returns the streaming response itself.
@router.post(
    "/chat",
    response_model=None,
    responses={200: {"content": {"text/event-stream": {}}, "description": "SSE events"}},
)
async def chat(
    body: ChatRequest, deps: Annotated[ChatDeps, Depends(get_chat_deps)]
) -> EventSourceResponse:
    message = body.message.strip()
    if not message:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "message is blank")
    settings = get_settings()
    if settings.openai_api_key is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "OPENAI_API_KEY is not set (put it in .env)"
        )
    history: list[HistoryMessage] = []
    if body.conversation_id is None:
        conversation_id, new_conversation = uuid.uuid4(), True
    else:
        conversation_id, new_conversation = body.conversation_id, False
        try:
            history = await run_in_threadpool(
                deps.store.load_history, conversation_id, settings.chat_history_messages
            )
        except ConversationNotFound as err:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown conversation_id") from err
    events = run_chat_turn(
        message,
        conversation_id=conversation_id,
        new_conversation=new_conversation,
        history=history,
        deps=deps,
        settings=settings,
    )
    return EventSourceResponse(
        _sse(events),
        # Proxies and browsers must pass each event on at once, not buffer the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

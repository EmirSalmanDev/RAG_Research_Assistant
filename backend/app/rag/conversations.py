"""Conversations and messages in Postgres: read the history, save a finished turn.

A turn is saved in ONE transaction at its end: the conversation (if new), the user's
message and the assistant's message together. So the history never holds half a turn:
- a turn that fails (search or generate error) saves nothing, so a retry of the same
  question does not leave a dangling user message behind;
- a refusal is a finished turn: both messages are saved, the assistant's being the
  fixed refusal text with `meta.refusal` set;
- spend is recorded separately (`llm_usage`, one committed row per request), so a failed
  turn's cost is never lost.
"""

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import select

from app.db.models import Conversation, Message
from app.db.session import session_scope
from app.rag.condense import HistoryMessage
from app.rag.meta import MessageMeta

TITLE_CHARS = 80  # a new conversation is titled with the start of its first message


class ConversationNotFound(LookupError):
    pass


@dataclass(frozen=True)
class TurnRecord:
    conversation_id: uuid.UUID
    new_conversation: bool
    question: str  # the user's message as sent
    answer: str  # the assistant's text (the refusal text on a refusal)
    citations: list[dict[str, Any]]
    tokens_in: int
    tokens_out: int
    cost_usd: Decimal
    trace_id: str | None
    meta: MessageMeta


class ChatStore(Protocol):
    def load_history(self, conversation_id: uuid.UUID, limit: int) -> list[HistoryMessage]: ...

    def save_turn(self, turn: TurnRecord) -> int: ...


class SqlChatStore:
    def load_history(self, conversation_id: uuid.UUID, limit: int) -> list[HistoryMessage]:
        """The last `limit` user/assistant messages, oldest first. Raises
        ConversationNotFound for an unknown id."""
        with session_scope() as session:
            if session.get(Conversation, conversation_id) is None:
                raise ConversationNotFound(str(conversation_id))
            if limit <= 0:
                return []
            rows = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .where(Message.role.in_(("user", "assistant")))
                .order_by(Message.id.desc())
                .limit(limit)
            ).all()
            return [HistoryMessage(r.role, r.content) for r in reversed(rows)]  # type: ignore[arg-type]

    def save_turn(self, turn: TurnRecord) -> int:
        """Save the whole turn at once; returns the assistant message's id."""
        with session_scope() as session:
            if turn.new_conversation:
                title = turn.question[:TITLE_CHARS]
                session.add(Conversation(id=turn.conversation_id, title=title))
                session.flush()
            elif session.get(Conversation, turn.conversation_id) is None:
                raise ConversationNotFound(str(turn.conversation_id))  # deleted meanwhile
            session.add(
                Message(conversation_id=turn.conversation_id, role="user", content=turn.question)
            )
            answer = Message(
                conversation_id=turn.conversation_id,
                role="assistant",
                content=turn.answer,
                citations=list(turn.citations),
                trace_id=turn.trace_id,
                tokens_in=turn.tokens_in,
                tokens_out=turn.tokens_out,
                cost_usd=turn.cost_usd,
                meta=turn.meta.model_dump(mode="json"),
            )
            session.add(answer)
            session.flush()  # assigns the id
            return answer.id

"""Saving chat turns in Postgres (scratch database): whole turns or nothing.

The pipeline runs with a scripted fake OpenAI and a fake search, but with the real
store (SqlChatStore) and the real llm_usage logging, so the rows are what /chat writes.
"""

import uuid
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Conversation, LLMUsage, Message
from app.llm.client import ChatClient
from app.rag.conversations import ConversationNotFound, SqlChatStore
from app.rag.meta import load_meta
from app.rag.pipeline import REFUSAL_TEXT, ChatDeps, ChatEvent, run_chat_turn
from tests.chat_fakes import FakeCompletions, FakeOpenAI, _Chat, parsed_reply, stream_reply
from tests.tracing_fakes import RecordingTracer
from tests.unit.test_pipeline import (
    ABSTRACT,
    SECTION,
    FakeDense,
    FakeSearch,
    FakeSparse,
    _plan,
    _verdicts,
)

pytestmark = pytest.mark.integration


def _deps(replies: list[Any], search: FakeSearch) -> ChatDeps:
    # record_usage left out: the real recorder writes llm_usage rows (scratch database).
    client = ChatClient(FakeOpenAI(_Chat(FakeCompletions(replies))), sleep=lambda s: None)
    return ChatDeps(
        chat=client,
        store=SqlChatStore(),
        tracer=RecordingTracer(),
        search_fn=search,
        dense=FakeDense,
        sparse=FakeSparse,
    )


def _turn(deps: ChatDeps, message: str, conversation: uuid.UUID, new: bool) -> list[ChatEvent]:
    history = [] if new else SqlChatStore().load_history(conversation, 6)
    return list(
        run_chat_turn(
            message, conversation_id=conversation, new_conversation=new, history=history, deps=deps
        )
    )


def test_a_turn_is_saved_whole_with_meta_and_usage(migrated_session: Session) -> None:
    conversation = uuid.uuid4()
    replies = [
        parsed_reply(_plan(rephrasings=[]), 300, 40),
        stream_reply(["Invented claim [1]. Abstract claim [2]."], 2000, 60),
        parsed_reply(_verdicts(True, True), 900, 30),
    ]
    events = _turn(
        _deps(replies, FakeSearch([SECTION, ABSTRACT])), "What is X?", conversation, True
    )
    assert events[-1].event == "done"

    assert migrated_session.get(Conversation, conversation).title == "What is X?"  # type: ignore[union-attr]
    rows = migrated_session.scalars(select(Message).order_by(Message.id)).all()
    assert [(m.role, m.content) for m in rows] == [
        ("user", "What is X?"),
        ("assistant", "Invented claim [1]. Abstract claim [2]."),
    ]
    user, answer = rows
    assert user.meta is None and answer.id == events[-1].data["message_id"]
    assert [c["chunk_type"] for c in answer.citations] == ["section", "abstract"]
    assert (answer.tokens_in, answer.tokens_out) == (3200, 130)
    meta = load_meta(answer.meta)
    assert meta is not None and meta.intent == "qa" and meta.verification is not None
    assert meta.prompts == {"plan": "plan_v1", "generate": "generate_v1", "verify": "verify_v1"}
    usage = migrated_session.scalars(select(LLMUsage).order_by(LLMUsage.id)).all()
    # One row per request, the question embedding included; together they are the turn's cost.
    assert [u.feature for u in usage] == [
        "chat-plan",
        "embed-query",
        "chat-generate",
        "chat-verify",
    ]
    assert sum(u.cost_usd for u in usage) == answer.cost_usd

    # A follow-up in the same conversation sees the history, oldest first.
    history = SqlChatStore().load_history(conversation, 6)
    assert [h.role for h in history] == ["user", "assistant"]
    assert SqlChatStore().load_history(conversation, 1)[0].role == "assistant"


def test_a_refusal_is_a_finished_turn(migrated_session: Session) -> None:
    conversation = uuid.uuid4()
    events = _turn(
        _deps([parsed_reply(_plan())], FakeSearch([])), "Unanswerable?", conversation, True
    )
    assert [e.event for e in events] == ["meta", "refusal", "done"]
    answer = migrated_session.scalars(select(Message).where(Message.role == "assistant")).one()
    assert answer.content == REFUSAL_TEXT and answer.citations == []
    assert load_meta(answer.meta).refusal == "no_sources"  # type: ignore[union-attr]


def test_a_failed_turn_leaves_no_history_but_keeps_its_spend(migrated_session: Session) -> None:
    conversation = uuid.uuid4()
    search = FakeSearch([], fail_with=ConnectionError("qdrant down"))
    events = _turn(_deps([parsed_reply(_plan())], search), "Q?", conversation, True)
    assert events[-1].event == "error"
    assert migrated_session.scalar(select(func.count()).select_from(Conversation)) == 0
    assert migrated_session.scalar(select(func.count()).select_from(Message)) == 0
    features = migrated_session.scalars(select(LLMUsage.feature).order_by(LLMUsage.id)).all()
    assert features == ["chat-plan", "embed-query"]


def test_unknown_conversations_are_refused(migrated_session: Session) -> None:
    with pytest.raises(ConversationNotFound):
        SqlChatStore().load_history(uuid.uuid4(), 6)

"""One chat turn: condense -> plan -> search -> rerank -> generate -> verify, then save.

`run_chat_turn` is a generator of events, which POST /chat sends to the browser as
server-sent events as soon as each one exists. Every stream ends with exactly one
`done` or one `error` event:

    meta          conversation id, intent, standalone question, the checked plan
    sources       the numbered sources [1]..[n] given to generate
    token ...     pieces of the answer, as the model writes them
    refusal       instead of tokens, when there is nothing to answer from
    verification  the citation checks
    done          message id, cited sources, abstract_only, tokens and cost, trace id
    error         a stage that could not fall back failed (nothing is saved)

What happens when a stage fails (docs/adr/0006):
- condense falls back to the raw message, plan to the default plan (intent qa, no
  filters, no rephrasings), verify to "unavailable" (the code checks are kept). The
  error is recorded on the stage's span and in the message meta; the turn goes on.
- search, rerank and generate cannot fall back: the stream ends with `error`.
The fallback for questions the papers cannot answer: if no chunk is left after the
filters, a fixed refusal is returned and generate is never called. If generate answers
with the insufficient-sources marker, the same refusal is returned. Abstract-only
sources are not a refusal case: they follow the abstract rule.

Search runs once for the standalone question and once for each rephrasing; the lists
are merged with Reciprocal Rank Fusion (the same k as inside Qdrant) and the pool is cut
to `search_candidates` before the reranker sees it, so rephrasings never make the
(slow) rerank longer.

Cost: every request writes its own llm_usage row at once. The assistant message also
gets the turn's totals: tokens_in / tokens_out of the chat calls, and cost_usd of the
chat calls plus the question embeddings.
"""

import uuid
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal

from app.config import Settings, get_settings
from app.ingest.embed import DenseEmbedder, DenseEncoder, SparseEncoder, get_sparse_embedder
from app.llm.client import ChatClient, ChatUsage
from app.llm.usage import UsageRecorder, embedding_cost, record_embedding_usage
from app.observability.redact import redact_secrets
from app.observability.tracing import Span, Trace, Tracer, get_tracer
from app.rag.citations import (
    Source,
    abstract_only,
    citation_objects,
    cited_numbers,
    number_sources,
    source_object,
)
from app.rag.condense import HistoryMessage, condense
from app.rag.conversations import ChatStore, SqlChatStore, TurnRecord
from app.rag.generate import GenerateStream, generate
from app.rag.meta import STAGES, MessageMeta, PlanMeta, Refusal, Stage
from app.rag.plan import DEFAULT_PLAN, QueryPlan, plan_query
from app.rag.rerank import CrossEncoder, cap_per_paper, get_cross_encoder, rerank
from app.rag.retrieve import RetrievalConfig
from app.rag.search import SearchHit, search
from app.rag.verify import VerifyResult, unavailable, verify

REFUSAL_TEXT = (
    "I could not find enough in the indexed papers to answer this. Try rephrasing the "
    "question, or ask about a topic or paper that the index covers."
)
ERROR_CHARS = 300  # error texts in meta and spans are cut to this

EventName = Literal["meta", "sources", "token", "refusal", "verification", "done", "error"]


@dataclass(frozen=True)
class ChatEvent:
    event: EventName
    data: dict[str, Any]


SearchFn = Callable[..., list[SearchHit]]


def _today() -> date:
    return datetime.now(UTC).date()


def _dense_embedder(record: UsageRecorder) -> DenseEncoder:
    return DenseEmbedder(record_usage=record)


@dataclass
class ChatDeps:
    """Everything a turn talks to; tests replace each part with a fake."""

    chat: ChatClient
    store: ChatStore
    tracer: Tracer
    search_fn: SearchFn = search
    dense: Callable[[UsageRecorder], DenseEncoder] = _dense_embedder
    sparse: Callable[[], SparseEncoder] = get_sparse_embedder
    cross_encoder: Callable[[], CrossEncoder] = get_cross_encoder
    record_embedding: UsageRecorder = record_embedding_usage
    today: Callable[[], date] = _today

    @classmethod
    def default(cls) -> "ChatDeps":
        return cls(chat=ChatClient(), store=SqlChatStore(), tracer=get_tracer())


def describe(err: BaseException) -> str:
    """An error as recorded in meta and on spans: its type and a short message, with
    anything that looks like a key removed (an OpenAI 401 quotes part of the key)."""
    return redact_secrets(f"{type(err).__name__}: {err}")[:ERROR_CHARS]


def fuse_ranked_lists(lists: Sequence[Sequence[SearchHit]], k: int) -> list[SearchHit]:
    """Merge ranked lists with Reciprocal Rank Fusion: each chunk scores the sum of
    1 / (k + position) over the lists it is in (position 0 = the top, as in Qdrant).
    A single list is returned unchanged; ties keep the order of first appearance."""
    if len(lists) == 1:
        return list(lists[0])
    scores: dict[uuid.UUID, float] = {}
    first: dict[uuid.UUID, SearchHit] = {}
    for hits in lists:
        for position, hit in enumerate(hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + position)
            first.setdefault(hit.chunk_id, hit)
    order = sorted(first, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [replace(first[c], score=scores[c]) for c in order]


@dataclass
class _Turn:
    """What a turn has spent and recorded so far."""

    record_embedding: UsageRecorder
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: Decimal = Decimal("0")
    embedding_requests: int = 0
    prompts: dict[Stage, str] = field(default_factory=dict)
    skipped: dict[Stage, str] = field(default_factory=dict)
    errors: dict[Stage, str] = field(default_factory=dict)

    def add(self, usage: ChatUsage | None) -> None:
        if usage is not None:
            self.tokens_in += usage.tokens_in
            self.tokens_out += usage.tokens_out
            self.cost_usd += usage.cost_usd

    def embedding_recorder(self, feature: str, model: str, tokens: int) -> None:
        """Logs a question embedding to llm_usage and counts it in the turn's cost."""
        self.record_embedding(feature, model, tokens)
        self.embedding_requests += 1
        self.cost_usd += embedding_cost(tokens)


def _usage(usage: ChatUsage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "tokens_in": usage.tokens_in,
        "tokens_out": usage.tokens_out,
        "cost_usd": str(usage.cost_usd),
    }


class _TrackedSpan:
    """A span that remembers whether it was ended (see `_Stages.close_missing`)."""

    def __init__(self, span: Span) -> None:
        self._span = span
        self.ended = False

    def end(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        usage: Mapping[str, Any] | None = None,
        error: str | None = None,
        skipped: str | None = None,
    ) -> None:
        self.ended = True
        self._span.end(output=output, metadata=metadata, usage=usage, error=error, skipped=skipped)


class _Stages:
    """Opens and closes the six stage spans and keeps meta's skipped/errors in step."""

    def __init__(self, trace: Trace, turn: _Turn) -> None:
        self._trace = trace
        self._turn = turn
        self._spans: dict[Stage, _TrackedSpan] = {}

    def open(self, stage: Stage, input: Any = None) -> Span:
        span = _TrackedSpan(self._trace.span(stage, input=input))
        self._spans[stage] = span
        return span

    def skip(self, stage: Stage, reason: str, span: Span | None = None) -> None:
        (span or self.open(stage)).end(skipped=reason)
        self._turn.skipped[stage] = reason

    def fail(self, stage: Stage, span: Span, err: BaseException) -> None:
        self._turn.errors[stage] = describe(err)
        span.end(error=describe(err), usage=_usage(getattr(err, "usage", None)))

    def close_missing(self, reason: str) -> None:
        """Every trace gets all six spans: a stage that never started is ended as skipped,
        one still open (the reader went away mid-answer) is ended with an error."""
        for stage in STAGES:
            span = self._spans.get(stage)
            if span is None:
                self.open(stage).end(skipped=reason)
            elif not span.ended:
                span.end(error=reason)


def _error_event(stage: str, err: BaseException) -> ChatEvent:
    """What the browser is told: the stage and the error type, no details."""
    return ChatEvent("error", {"stage": stage, "message": f"{stage} failed: {type(err).__name__}"})


def _ids(hits: Sequence[SearchHit]) -> list[str]:
    return [str(h.chunk_id) for h in hits]


def run_chat_turn(
    message: str,
    *,
    conversation_id: uuid.UUID,
    new_conversation: bool,
    history: Sequence[HistoryMessage],
    deps: ChatDeps,
    settings: Settings | None = None,
) -> Generator[ChatEvent]:
    """Run one turn and yield its events (see the module doc for their order)."""
    s = settings or get_settings()
    config = RetrievalConfig.from_settings()
    turn = _Turn(deps.record_embedding)
    trace = deps.tracer.start_trace(
        "chat-turn",
        input={"message": message},
        metadata={"conversation_id": str(conversation_id), "chat_model": deps.chat.model},
    )
    stages = _Stages(trace, turn)
    outcome: dict[str, Any] = {"status": "ended early"}
    try:
        # 1. condense: a follow-up becomes a standalone question.
        question = message
        span = stages.open("condense", {"message": message, "history": len(history)})
        try:
            condensed = condense(
                message,
                history,
                client=deps.chat,
                enabled=s.condense_enabled,
                max_history=s.chat_history_messages,
            )
        except Exception as err:
            turn.add(getattr(err, "usage", None))
            stages.fail("condense", span, err)  # fall back to the raw message
        else:
            if condensed.skipped:
                stages.skip("condense", condensed.skipped, span)
            else:
                question = condensed.question
                turn.add(condensed.usage)
                turn.prompts["condense"] = condensed.prompt or ""
                span.end(output=question, usage=_usage(condensed.usage))

        # 2. plan: intent, checked filters and rephrasings.
        plan: QueryPlan = DEFAULT_PLAN
        span = stages.open("plan", {"question": question})
        try:
            planned = plan_query(
                question,
                client=deps.chat,
                enabled=s.plan_enabled,
                use_rephrasings=s.plan_rephrasings_enabled,
                categories=s.arxiv_categories,
                today=deps.today(),
            )
        except Exception as err:
            turn.add(getattr(err, "usage", None))
            stages.fail("plan", span, err)  # fall back to the default plan
        else:
            plan = planned.plan
            if planned.skipped:
                stages.skip("plan", planned.skipped, span)
            else:
                turn.add(planned.usage)
                turn.prompts["plan"] = planned.prompt or ""
                span.end(
                    output=PlanMeta.from_plan(plan).model_dump(mode="json"),
                    usage=_usage(planned.usage),
                )
        plan_meta = PlanMeta.from_plan(plan)
        yield ChatEvent(
            "meta",
            {
                "conversation_id": str(conversation_id),
                "new_conversation": new_conversation,
                "intent": plan.intent,
                "standalone_question": question,
                "plan": plan_meta.model_dump(mode="json"),
            },
        )

        # 3. search: the question and its rephrasings, merged, cut to the candidate count.
        queries = [question, *plan.rephrasings]
        span = stages.open(
            "search",
            {"queries": queries, "mode": config.mode, "filters": plan_meta.model_dump(mode="json")},
        )
        try:
            dense = deps.dense(turn.embedding_recorder) if config.mode != "bm25" else None
            sparse = deps.sparse() if config.mode != "dense" else None
            lists = [
                deps.search_fn(
                    q,
                    plan.filters,
                    mode=config.mode,
                    limit=config.candidates,
                    prefetch_limit=config.prefetch_limit,
                    rrf_k=config.rrf_k,
                    dense=dense,
                    sparse=sparse,
                )
                for q in queries
            ]
        except Exception as err:
            stages.fail("search", span, err)
            outcome = {"status": "error", "stage": "search"}
            yield _error_event("search", err)
            return
        pool = fuse_ranked_lists(lists, config.rrf_k)[: config.candidates]
        span.end(
            output=_ids(pool),
            metadata={
                "per_query": [len(h) for h in lists],
                "embedding_requests": turn.embedding_requests,
            },
        )

        # 4. rerank (optional), then at most N chunks per paper and the top sources.
        if not config.rerank:
            stages.skip("rerank", "disabled")
        elif not pool:
            stages.skip("rerank", "no candidates")
        else:
            span = stages.open("rerank", {"candidates": len(pool)})
            try:
                pool = rerank(
                    question, pool, deps.cross_encoder(), batch_size=config.rerank_batch_size
                )
            except Exception as err:
                stages.fail("rerank", span, err)
                outcome = {"status": "error", "stage": "rerank"}
                yield _error_event("rerank", err)
                return
            span.end(output=_ids(pool))
        capped = cap_per_paper(
            pool,
            config.max_chunks_per_paper,
            max_analysis_per_paper=config.max_analysis_chunks_per_paper,
        )
        sources = number_sources(capped[: s.chat_sources])

        if not sources:  # the fallback: nothing to answer from, so no generate call
            stages.skip("generate", "refusal: no sources")
            stages.skip("verify", "refusal: no sources")
            outcome = {"status": "refusal", "refusal": "no_sources"}
            yield from _refuse(
                "no_sources",
                question,
                plan,
                sources,
                turn,
                trace,
                deps,
                conversation_id,
                new_conversation,
                message,
                outcome,
            )
            return
        yield ChatEvent("sources", {"sources": [source_object(x) for x in sources]})

        # 5. generate: streamed; the marker check happens inside GenerateStream.
        span = stages.open(
            "generate",
            {
                "question": question,
                "intent": plan.intent,
                "sources": _ids([x.hit for x in sources]),
            },
        )
        answer: GenerateStream | None = None
        try:
            answer = generate(
                question,
                sources,
                plan.intent,
                client=deps.chat,
                max_tokens=s.chat_max_output_tokens,
                max_tokens_explain=s.chat_max_output_tokens_explain,
            )
            for piece in answer:
                yield ChatEvent("token", {"text": piece})
        except Exception as err:
            billed = answer.usage if answer is not None else None
            turn.add(billed or getattr(err, "usage", None))
            stages.fail("generate", span, err)
            outcome = {"status": "error", "stage": "generate"}
            yield _error_event("generate", err)
            return
        finally:
            if answer is not None:
                answer.close()  # also when the reader went away: logs the usage
        assert answer is not None  # set in the try block, which did not fail
        turn.add(answer.usage)
        turn.prompts["generate"] = answer.prompt
        span.end(output=answer.text, usage=_usage(answer.usage))
        if answer.insufficient:
            stages.skip("verify", "refusal: insufficient sources")
            outcome = {"status": "refusal", "refusal": "insufficient_sources"}
            yield from _refuse(
                "insufficient_sources",
                question,
                plan,
                sources,
                turn,
                trace,
                deps,
                conversation_id,
                new_conversation,
                message,
                outcome,
            )
            return

        # 6. verify: code checks always; the model check falls back to "unavailable".
        span = stages.open("verify", {"answer_chars": len(answer.text), "sources": len(sources)})
        verification: VerifyResult
        try:
            verification, usage = verify(
                answer.text, sources, client=deps.chat, enabled=s.verify_enabled
            )
        except Exception as err:
            turn.add(getattr(err, "usage", None))
            stages.fail("verify", span, err)
            verification = unavailable(answer.text, sources, describe(err))
        else:
            turn.add(usage)
            if verification.status == "skipped":
                stages.skip("verify", verification.reason, span)
            else:
                turn.prompts["verify"] = verification.prompt or ""
                span.end(
                    output={
                        "unsupported": verification.unsupported_sentences,
                        "invalid": verification.invalid_citations,
                    },
                    usage=_usage(usage),
                )
        yield ChatEvent("verification", verification.model_dump(mode="json"))

        # Save the finished turn (one transaction), then report it.
        cited = cited_numbers(answer.text)
        meta = _meta(
            question,
            plan,
            sources,
            turn,
            abstract_only=abstract_only(sources, cited),
            verification=verification,
        )
        record = TurnRecord(
            conversation_id=conversation_id,
            new_conversation=new_conversation,
            question=message,
            answer=answer.text,
            citations=citation_objects(sources, cited),
            tokens_in=turn.tokens_in,
            tokens_out=turn.tokens_out,
            cost_usd=turn.cost_usd,
            trace_id=trace.id,
            meta=meta,
        )
        outcome = {"status": "answered"}
        yield from _save_and_finish(record, deps, outcome)
    finally:
        stages.close_missing("turn ended early")
        trace.end(
            output=outcome,
            metadata={
                "prompts": dict(turn.prompts),
                "skipped": dict(turn.skipped),
                "errors": dict(turn.errors),
            },
        )


def _meta(
    question: str,
    plan: QueryPlan,
    sources: Sequence[Source],
    turn: _Turn,
    *,
    refusal: Refusal | None = None,
    abstract_only: bool = False,
    verification: VerifyResult | None = None,
) -> MessageMeta:
    return MessageMeta(
        intent=plan.intent,
        standalone_question=question,
        plan=PlanMeta.from_plan(plan),
        sources=len(sources),
        refusal=refusal,
        abstract_only=abstract_only,
        prompts=dict(turn.prompts),
        skipped=dict(turn.skipped),
        errors=dict(turn.errors),
        verification=verification,
    )


def _refuse(
    reason: Refusal,
    question: str,
    plan: QueryPlan,
    sources: Sequence[Source],
    turn: _Turn,
    trace: Trace,
    deps: ChatDeps,
    conversation_id: uuid.UUID,
    new_conversation: bool,
    message: str,
    outcome: dict[str, Any],
) -> Iterator[ChatEvent]:
    """The fixed refusal: shown, saved as a finished turn (no citations), reported."""
    yield ChatEvent("refusal", {"reason": reason, "text": REFUSAL_TEXT})
    shown = sources if reason == "insufficient_sources" else []
    record = TurnRecord(
        conversation_id=conversation_id,
        new_conversation=new_conversation,
        question=message,
        answer=REFUSAL_TEXT,
        citations=[],
        tokens_in=turn.tokens_in,
        tokens_out=turn.tokens_out,
        cost_usd=turn.cost_usd,
        trace_id=trace.id,
        meta=_meta(question, plan, shown, turn, refusal=reason),
    )
    yield from _save_and_finish(record, deps, outcome)


def _save_and_finish(
    record: TurnRecord, deps: ChatDeps, outcome: dict[str, Any]
) -> Iterator[ChatEvent]:
    """Save the turn in one transaction; `done` only once it is stored."""
    try:
        message_id = deps.store.save_turn(record)
    except Exception as err:
        outcome.update(status="error", stage="save")  # the trace shows the failed save
        yield _error_event("save", err)
        return
    yield ChatEvent(
        "done",
        {
            "conversation_id": str(record.conversation_id),
            "message_id": message_id,
            "refusal": record.meta.refusal,
            "abstract_only": record.meta.abstract_only,
            "citations": record.citations,
            "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out,
            "cost_usd": str(record.cost_usd),
            "trace_id": record.trace_id,
        },
    )

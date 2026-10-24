"""The chat pipeline with every outside part faked: OpenAI (scripted replies), search,
the embedders, the reranker, the store and the tracer. No network, no stores."""

import uuid
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import openai
import pytest

from app.config import Settings, get_settings
from app.llm.usage import chat_cost, embedding_cost
from app.rag.condense import HistoryMessage
from app.rag.conversations import TurnRecord
from app.rag.pipeline import (
    REFUSAL_TEXT,
    ChatDeps,
    ChatEvent,
    fuse_ranked_lists,
    run_chat_turn,
)
from app.rag.plan import PlanOutput
from app.rag.search import SearchHit
from app.rag.verify import Verdict, VerifyOutput
from tests.chat_fakes import (
    FakeCompletions,
    fake_chat,
    parsed_reply,
    status_error,
    stream_reply,
    text_reply,
)
from tests.tracing_fakes import RecordedTrace, RecordingTracer

CONVERSATION = uuid.UUID("00000000-0000-4000-8000-0000000000aa")
HISTORY = [HistoryMessage("user", "What is LoRA?"), HistoryMessage("assistant", "A method [1].")]
STAGES = ["condense", "plan", "search", "rerank", "generate", "verify"]


def _hit(name: str, arxiv_id: str, chunk_type: str = "section") -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, name),
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        section_path=None if chunk_type == "abstract" else "3 Method",
        chunk_type=chunk_type,
        published_at="2026-09-01T00:00:00+00:00",
        text=f"Invented text {name}.",
        score=1.0,
    )


SECTION = _hit("s1", "2601.00001")
ABSTRACT = _hit("a1", "2601.00002", "abstract")


def _plan(**kw: Any) -> PlanOutput:
    base: dict[str, Any] = {
        "intent": "qa",
        "rephrasings": ["low-rank adapters", "LoRA finetuning"],
        "arxiv_ids": [],
        "categories": [],
        "published_from": None,
        "published_to": None,
    }
    return PlanOutput.model_validate(base | kw)


def _verdicts(*supported: bool) -> VerifyOutput:
    return VerifyOutput(
        verdicts=[Verdict(index=i, supported=v, reason="") for i, v in enumerate(supported)]
    )


@dataclass
class FakeDense:
    record: Any

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.record(feature, "text-embedding-3-small", 12)  # what DenseEmbedder does
        return [[0.0] for _ in texts]


class FakeSparse:
    """Never called: the fake search does not compute BM25 vectors."""

    def embed(self, texts: Sequence[str]) -> list[Any]:
        raise AssertionError("not used by the fake search")


@dataclass
class FakeSearch:
    """Returns `results[query]` (default: `default`); embeds each query like search() does."""

    default: list[SearchHit]
    results: dict[str, list[SearchHit]] = field(default_factory=dict)
    fail_with: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, query: str, filters: Any, **kw: Any) -> list[SearchHit]:
        self.calls.append({"query": query, "filters": filters, **kw})
        if kw.get("dense") is not None:
            kw["dense"].embed([query], "embed-query")
        if self.fail_with is not None:
            raise self.fail_with
        return list(self.results.get(query, self.default))


class FakeStore:
    def __init__(self, fail: bool = False) -> None:
        self.saved: list[TurnRecord] = []
        self.fail = fail

    def load_history(self, conversation_id: uuid.UUID, limit: int) -> list[HistoryMessage]:
        raise AssertionError("the pipeline is given its history")

    def save_turn(self, turn: TurnRecord) -> int:
        if self.fail:
            raise RuntimeError("database down")
        self.saved.append(turn)
        return 42


class ReverseCrossEncoder:
    def __init__(self) -> None:
        self.seen: list[int] = []

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 16) -> list[float]:
        docs = list(documents)
        self.seen.append(len(docs))
        return [float(i) for i in range(len(docs))]


@dataclass
class Harness:
    deps: ChatDeps
    completions: FakeCompletions
    chat_rows: list[tuple[str, str, int, int]]
    embed_rows: list[tuple[str, str, int]]
    search: FakeSearch
    store: FakeStore
    tracer: RecordingTracer
    cross: ReverseCrossEncoder

    @property
    def trace(self) -> RecordedTrace:
        (trace,) = self.tracer.traces
        return trace

    def run(
        self, message: str = "and its limits?", history: list[HistoryMessage] | None = None
    ) -> list[ChatEvent]:
        return list(self.stream(message, HISTORY if history is None else history))

    def stream(self, message: str, history: list[HistoryMessage]) -> Generator[ChatEvent]:
        return run_chat_turn(
            message,
            conversation_id=CONVERSATION,
            new_conversation=False,
            history=history,
            deps=self.deps,
            settings=get_settings(),
        )


def harness(
    replies: list[Any], hits: list[SearchHit] | None = None, *, store_fails: bool = False
) -> Harness:
    client, completions, chat_rows = fake_chat(replies)
    embed_rows: list[tuple[str, str, int]] = []
    search = FakeSearch([SECTION, ABSTRACT] if hits is None else hits)
    store, tracer, cross = FakeStore(store_fails), RecordingTracer(), ReverseCrossEncoder()
    deps = ChatDeps(
        chat=client,
        store=store,
        tracer=tracer,
        search_fn=search,
        dense=FakeDense,
        sparse=FakeSparse,
        cross_encoder=lambda: cross,
        record_embedding=lambda f, m, t: embed_rows.append((f, m, t)),
    )
    return Harness(deps, completions, chat_rows, embed_rows, search, store, tracer, cross)


def _names(events: list[ChatEvent]) -> list[str]:
    return [e.event for e in events]


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    return get_settings()


ANSWER_PIECES = ["Paper 2601.00001 adds adapters [1]. ", "Its abstract says less [2]."]


def full_turn() -> list[Any]:
    """Fresh replies for condense, plan, generate and verify (a stream is read once)."""
    return [
        text_reply("What are the limits of LoRA?", 200, 12),
        parsed_reply(_plan(), 300, 40),
        stream_reply(ANSWER_PIECES, 2000, 60),
        parsed_reply(_verdicts(True, False), 900, 30),
    ]


# --- a full turn ---------------------------------------------------------------------------------


def test_a_full_turn_streams_saves_and_traces_six_stages() -> None:
    h = harness(full_turn())
    events = h.run()
    assert _names(events) == ["meta", "sources", "token", "token", "verification", "done"]

    meta = events[0].data
    assert meta["conversation_id"] == str(CONVERSATION) and meta["intent"] == "qa"
    assert meta["standalone_question"] == "What are the limits of LoRA?"
    assert [s["n"] for s in events[1].data["sources"]] == [1, 2]
    assert "".join(e.data["text"] for e in events if e.event == "token") == "".join(ANSWER_PIECES)
    assert events[4].data["unsupported_sentences"] == [1]

    # Searched three times: the standalone question and the two rephrasings.
    assert [c["query"] for c in h.search.calls] == [
        "What are the limits of LoRA?",
        "low-rank adapters",
        "LoRA finetuning",
    ]

    # Six spans, in order; rerank is off by default, so its span says skipped.
    trace = h.trace
    assert [s.name for s in trace.spans] == STAGES and all(s.ended for s in trace.spans)
    assert trace.span_named("rerank").skipped == "disabled"
    assert trace.span_named("generate").usage == {
        "tokens_in": 2000,
        "tokens_out": 60,
        "cost_usd": str(chat_cost(2000, 60)),
    }
    assert trace.metadata["prompts"] == {
        "condense": "condense_v1",
        "plan": "plan_v1",
        "generate": "generate_v2",
        "verify": "verify_v1",
    }

    # Saved once, with the totals: chat tokens, and chat + embedding cost.
    (saved,) = h.store.saved
    assert saved.question == "and its limits?" and saved.answer == "".join(ANSWER_PIECES).strip()
    assert [c["n"] for c in saved.citations] == [1, 2]
    assert saved.citations[1]["chunk_type"] == "abstract"
    assert (saved.tokens_in, saved.tokens_out) == (200 + 300 + 2000 + 900, 12 + 40 + 60 + 30)
    expected = (
        chat_cost(200, 12)
        + chat_cost(300, 40)
        + chat_cost(2000, 60)
        + chat_cost(900, 30)
        + 3 * embedding_cost(12)
    )
    assert saved.cost_usd == expected
    assert h.embed_rows == [("embed-query", "text-embedding-3-small", 12)] * 3
    assert [r[0] for r in h.chat_rows] == [
        "chat-condense",
        "chat-plan",
        "chat-generate",
        "chat-verify",
    ]
    assert saved.trace_id == "trace-0001" and saved.meta.abstract_only is False
    assert saved.meta.skipped == {"rerank": "disabled"} and saved.meta.errors == {}
    assert saved.meta.verification is not None
    assert saved.meta.verification.status == "checked"

    done = events[-1].data
    assert done["message_id"] == 42 and done["cost_usd"] == str(expected)
    assert done["refusal"] is None and done["abstract_only"] is False


def test_a_first_message_skips_condense_without_a_call() -> None:
    h = harness(list(full_turn()[1:]))
    events = h.run("What is LoRA?", history=[])
    assert events[-1].event == "done"
    assert h.trace.span_named("condense").skipped == "no history"
    assert h.store.saved[0].meta.standalone_question == "What is LoRA?"


# --- the abstract rule ---------------------------------------------------------------------------


def test_abstract_only_comes_from_the_cited_sources() -> None:
    replies = [
        parsed_reply(_plan()),
        stream_reply(["Only the abstract says so [2]."]),
        parsed_reply(_verdicts(True)),
    ]
    h = harness(replies)
    events = h.run("q", history=[])
    assert events[-1].data["abstract_only"] is True  # [2] is the abstract chunk
    assert h.store.saved[0].meta.abstract_only is True


def test_all_abstract_sources_get_the_note_but_are_not_a_refusal() -> None:
    replies = [
        parsed_reply(_plan()),
        stream_reply(["Abstracts only [1]."]),
        parsed_reply(_verdicts(True)),
    ]
    h = harness(replies, hits=[ABSTRACT])
    events = h.run("q", history=[])
    assert _names(events)[-1] == "done" and events[-1].data["refusal"] is None
    user = next(c for c in h.completions.calls if c["kind"] == "stream")["messages"][1]
    assert user["content"].startswith("<note>")


# --- the refusals --------------------------------------------------------------------------------


def test_no_chunks_after_filters_refuses_without_calling_generate() -> None:
    h = harness([parsed_reply(_plan(categories=["cs.CL"]))], hits=[])
    events = h.run("Recent cs.CL papers on X?", history=[])
    assert _names(events) == ["meta", "refusal", "done"]
    assert events[1].data == {"reason": "no_sources", "text": REFUSAL_TEXT}
    assert [c["kind"] for c in h.completions.calls] == ["parse"]  # plan only
    assert h.trace.span_named("generate").skipped == "refusal: no sources"
    assert h.trace.span_named("verify").skipped == "refusal: no sources"
    (saved,) = h.store.saved
    assert (saved.answer, saved.citations, saved.meta.refusal) == (REFUSAL_TEXT, [], "no_sources")
    assert saved.meta.sources == 0 and saved.meta.plan.categories == ["cs.CL"]
    assert events[-1].data["refusal"] == "no_sources"


def test_the_insufficient_marker_becomes_the_same_refusal() -> None:
    h = harness([parsed_reply(_plan()), stream_reply(["INSUFFICIENT", "_SOURCES"], 1500, 5)])
    events = h.run("q", history=[])
    assert _names(events) == ["meta", "sources", "refusal", "done"]  # no token shown
    assert events[2].data["reason"] == "insufficient_sources"
    assert h.trace.span_named("verify").skipped == "refusal: insufficient sources"
    (saved,) = h.store.saved
    assert saved.answer == REFUSAL_TEXT and saved.meta.refusal == "insufficient_sources"
    assert saved.meta.sources == 2 and saved.tokens_in == 100 + 1500  # plan + generate


# --- stages that fall back -----------------------------------------------------------------------


def test_a_failed_condense_falls_back_to_the_raw_message() -> None:
    replies = [status_error(openai.BadRequestError, 400), *full_turn()[1:]]
    h = harness(replies)
    events = h.run("and its limits?")
    assert events[0].data["standalone_question"] == "and its limits?"
    assert events[-1].event == "done"
    assert h.trace.span_named("condense").error is not None
    assert h.store.saved[0].meta.errors["condense"].startswith("BadRequestError")


def test_a_failed_plan_falls_back_to_the_default_plan() -> None:
    replies = [full_turn()[0], parsed_reply(None), *full_turn()[2:]]  # a refusal: bad data
    h = harness(replies)
    events = h.run()
    assert events[0].data["intent"] == "qa" and events[0].data["plan"]["rephrasings"] == []
    assert len(h.search.calls) == 1  # no rephrasings
    assert h.store.saved[0].meta.errors["plan"].startswith("ChatResponseError")
    assert h.store.saved[0].tokens_in == 200 + 100 + 2000 + 900  # the failed plan still counts


def test_a_failed_verify_is_unavailable_and_the_answer_stands() -> None:
    replies = [*full_turn()[:3], parsed_reply(VerifyOutput(verdicts=[]))]  # wrong sentences
    h = harness(replies)
    events = h.run()
    assert _names(events)[-1] == "done"
    verification = events[-2].data
    assert verification["status"] == "unavailable" and "ChatResponseError" in verification["reason"]
    assert h.trace.span_named("verify").error is not None
    assert h.store.saved[0].meta.errors["verify"].startswith("ChatResponseError")


# --- stages that cannot fall back ---------------------------------------------------------------


def _assert_nothing_saved_and_six_spans(h: Harness) -> None:
    assert h.store.saved == []
    assert [s.name for s in h.trace.spans] == STAGES and all(s.ended for s in h.trace.spans)


def test_a_failed_search_ends_with_an_error_and_saves_nothing() -> None:
    h = harness(list(full_turn()[:2]))
    h.search.fail_with = ConnectionError("qdrant down")
    events = h.run()
    assert _names(events) == ["meta", "error"]
    assert events[-1].data == {"stage": "search", "message": "search failed: ConnectionError"}
    _assert_nothing_saved_and_six_spans(h)
    assert h.trace.span_named("generate").skipped == "turn ended early"


def test_a_failed_generate_ends_with_an_error_and_saves_nothing() -> None:
    h = harness([*full_turn()[:2], status_error(openai.BadRequestError, 400)])
    events = h.run()
    assert _names(events) == ["meta", "sources", "error"]
    assert events[-1].data["stage"] == "generate"
    _assert_nothing_saved_and_six_spans(h)


def test_a_stream_that_breaks_midway_ends_with_an_error() -> None:
    h = harness([*full_turn()[:2], stream_reply(ANSWER_PIECES, fail_after=1)])
    events = h.run()
    assert _names(events) == ["meta", "sources", "token", "error"]
    _assert_nothing_saved_and_six_spans(h)
    # No usage chunk arrived, so the spend is logged as an estimate (rule 9).
    assert [r[0] for r in h.chat_rows] == ["chat-condense", "chat-plan", "chat-generate-estimated"]


def test_a_failed_save_is_an_error_not_done() -> None:
    h = harness(full_turn(), store_fails=True)
    events = h.run()
    assert events[-1].event == "error" and events[-1].data["stage"] == "save"
    assert h.trace.output == {"status": "error", "stage": "save"}


def test_a_reader_that_leaves_mid_answer_closes_every_span() -> None:
    raw = stream_reply(ANSWER_PIECES, 2000, 60)
    h = harness([*full_turn()[:2], raw])
    events = h.stream("and its limits?", HISTORY)
    for event in events:
        if event.event == "token":
            break
    events.close()  # what the server does when the browser disconnects
    trace = h.trace
    assert [s.name for s in trace.spans] == STAGES and all(s.ended for s in trace.spans)
    assert trace.span_named("generate").error == "turn ended early"
    assert ("chat-generate", "gpt-4o-mini", 2000, 60) in h.chat_rows  # usage still logged
    assert raw.closed and h.store.saved == []


# --- search: rephrasings, the merge, the cap and rerank -----------------------------------------


def test_fuse_ranked_lists_is_rrf_with_positions_from_zero() -> None:
    a, b, c = _hit("a", "1"), _hit("b", "2"), _hit("c", "3")
    assert fuse_ranked_lists([[a, b]], 60) == [a, b]  # one list: unchanged
    fused = fuse_ranked_lists([[a, b], [b, c]], 60)
    assert [h.chunk_id for h in fused] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 60)  # b: position 1, then 0


def test_the_merged_pool_is_cut_to_the_candidate_count_before_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings(monkeypatch, RERANK_ENABLED="true", SEARCH_CANDIDATES="4", MAX_CHUNKS_PER_PAPER="0")
    replies = [parsed_reply(_plan()), stream_reply(["x [1]."]), parsed_reply(_verdicts(True))]
    h = harness(replies)
    hits = {
        q: [_hit(f"{q}-{i}", f"26{i:02d}.{len(q):05d}") for i in range(4)]
        for q in ("q", "low-rank adapters", "LoRA finetuning")
    }
    h.search.results = hits
    events = h.run("q", history=[])
    assert all(c["limit"] == 4 for c in h.search.calls)  # each list: 4 candidates
    assert h.cross.seen == [4]  # 12 merged chunks, cut to 4 before the reranker
    assert h.trace.span_named("rerank").skipped is None
    assert len(events[1].data["sources"]) == 4


def test_rephrasings_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    _settings(monkeypatch, PLAN_REPHRASINGS_ENABLED="false")
    h = harness([parsed_reply(_plan()), stream_reply(["x [1]."]), parsed_reply(_verdicts(True))])
    h.run("q", history=[])
    assert [c["query"] for c in h.search.calls] == ["q"]


@pytest.mark.parametrize(("chat_sources", "expected"), [("8", 4), ("2", 2)])
def test_sources_are_capped_per_paper_then_at_chat_sources(
    monkeypatch: pytest.MonkeyPatch, chat_sources: str, expected: int
) -> None:
    _settings(monkeypatch, CHAT_SOURCES=chat_sources)
    many = [_hit(f"p{i}", "2601.00009") for i in range(5)] + [_hit("o", "2601.00010")]
    replies = [
        parsed_reply(_plan(rephrasings=[])),
        stream_reply(["x [1]."]),
        parsed_reply(_verdicts(True)),
    ]
    h = harness(replies, hits=many)
    sources = h.run("q", history=[])[1].data["sources"]
    # At most 3 chunks of 2601.00009 (the default cap), then the other paper.
    assert [s["arxiv_id"] for s in sources] == (["2601.00009"] * 3 + ["2601.00010"])[:expected]


@pytest.mark.parametrize(("max_analysis", "expected"), [
    ("1", ["analysis", "section", "section", "section"]),  # one, outside the cap of 3
    ("0", ["section", "section", "section"]),
    ("-1", ["analysis", "analysis", "section"]),  # counted like any chunk (before ADR 0012)
])  # fmt: skip
def test_the_analysis_limit_reaches_the_chat_sources(
    monkeypatch: pytest.MonkeyPatch, max_analysis: str, expected: list[str]
) -> None:
    _settings(monkeypatch, MAX_ANALYSIS_CHUNKS_PER_PAPER=max_analysis)
    paper = "2601.00009"
    hits = [_hit("x1", paper, "analysis"), _hit("x2", paper, "analysis")] + [
        _hit(f"s{i}", paper) for i in range(4)
    ]
    replies = [parsed_reply(_plan(rephrasings=[])), stream_reply(["x [1]."]),
               parsed_reply(_verdicts(True))]  # fmt: skip
    sources = harness(replies, hits=hits).run("q", history=[])[1].data["sources"]
    assert [s["chunk_type"] for s in sources] == expected


def test_disabling_every_optional_stage_skips_their_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _settings(
        monkeypatch, CONDENSE_ENABLED="false", PLAN_ENABLED="false", VERIFY_ENABLED="false"
    )
    h = harness([stream_reply(["x [1]."])])
    events = list(
        run_chat_turn(
            "q",
            conversation_id=CONVERSATION,
            new_conversation=True,
            history=HISTORY,
            deps=h.deps,
            settings=s,
        )
    )
    assert events[-1].event == "done" and [c["kind"] for c in h.completions.calls] == ["stream"]
    skipped = h.store.saved[0].meta.skipped
    assert skipped == {
        "condense": "disabled",
        "plan": "disabled",
        "rerank": "disabled",
        "verify": "disabled",
    }
    assert h.store.saved[0].cost_usd == chat_cost(1000, 50) + embedding_cost(12)

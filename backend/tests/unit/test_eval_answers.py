"""The answer evaluation (eval/answers.py): a real chat turn on scripted fakes, the answer
metrics on a scripted judge, the aggregates per split, and the cost estimate.

No network and no stores: OpenAI is a scripted fake, search returns invented chunks, and
the evaluation's chat store keeps nothing."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pytest

from app.config import get_settings
from app.llm.usage import chat_cost, embedding_cost
from app.observability.tracing import NoopTracer
from app.rag.citations import Source
from app.rag.generate import INSUFFICIENT_MARKER
from app.rag.pipeline import ChatDeps
from app.rag.plan import PlanOutput
from app.rag.search import SearchHit
from app.rag.verify import Verdict, VerifyOutput
from eval.answers import (
    AnswerSample,
    NullChatStore,
    SampleScores,
    TokenStats,
    aggregate,
    citation_items,
    collect_sample,
    estimate,
    is_unanswerable,
    run,
    score_sample,
    select_questions,
)
from eval.golden import GoldenItem, SetQuestion
from eval.judge import (
    AttributedOutput,
    AttributedVerdict,
    Judge,
    RelevancyOutput,
    StatementsOutput,
    SupportOutput,
    SupportVerdict,
    UsefulOutput,
    UsefulVerdict,
)
from tests.chat_fakes import fake_chat, parsed_reply, stream_reply


def _hit(name: str, arxiv_id: str, chunk_type: str, text: str) -> SearchHit:
    return SearchHit(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_URL, name), arxiv_id=arxiv_id, title=f"Paper {arxiv_id}",
        section_path=None if chunk_type == "abstract" else "3 Method", chunk_type=chunk_type,
        published_at="2026-09-01T00:00:00+00:00", text=text, score=1.0,
    )  # fmt: skip


SEC = _hit("sec", "2601.00001", "section", "LoRA trains two low-rank matrices per layer.")
ABS = _hit("abs", "2601.00002", "abstract", "The pretrained weights stay frozen.")
QUESTION = "How does LoRA fine-tune a model?"
REFERENCE = "LoRA trains two low-rank matrices. The pretrained weights stay frozen."


def _question(
    qid: str = "gen-001", split: Any = "generated", refs: bool = True, **kw: Any
) -> SetQuestion:
    item = GoldenItem(
        id=qid, question=kw.pop("question", QUESTION),
        reference_chunk_ids=[SEC.chunk_id] if refs else [], **kw,
    )  # fmt: skip
    return SetQuestion(item, split)


def _plan() -> PlanOutput:
    return PlanOutput.model_validate(
        {"intent": "qa", "rephrasings": ["low-rank adapters", "LoRA finetuning"], "arxiv_ids": [],
         "categories": [], "published_from": None, "published_to": None}
    )  # fmt: skip


@dataclass
class FakeSearch:
    hits: list[SearchHit]
    fail: Exception | None = None
    calls: int = 0

    def __call__(self, query: str, filters: Any, **kw: Any) -> list[SearchHit]:
        self.calls += 1
        if kw.get("dense") is not None:
            kw["dense"].embed([query], "embed-query")
        if self.fail is not None:
            raise self.fail
        return list(self.hits)


@dataclass
class FakeQueryDense:
    record: Any

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.record(feature, "text-embedding-3-small", 12)
        return [[0.0] for _ in texts]


class Clock:
    """Each call moves time on by 0.25 s, so latencies are exact."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        self.now += 0.25
        return self.now


def _deps(replies: list[Any], search: FakeSearch) -> tuple[ChatDeps, list[Any]]:
    client, _, rows = fake_chat(replies)
    deps = ChatDeps(
        chat=client, store=NullChatStore(), tracer=NoopTracer(), search_fn=search,
        dense=FakeQueryDense, sparse=lambda: None,  # type: ignore[arg-type,return-value]
        record_embedding=lambda f, m, t: None,
    )  # fmt: skip
    return deps, rows


ANSWER = ["LoRA trains two matrices [1]. ", "The weights stay frozen [2][9]. ", "That is all."]


def _turn_replies(pieces: list[str] = ANSWER) -> list[Any]:
    verdicts = VerifyOutput(verdicts=[Verdict(index=0, supported=True, reason=""),
                                      Verdict(index=1, supported=True, reason="")])  # fmt: skip
    return [
        parsed_reply(_plan(), 300, 40),
        stream_reply(pieces, 2000, 60),
        parsed_reply(verdicts, 900, 30),
    ]


# --- collecting one turn ----------------------------------------------------------------------


def test_a_turn_gives_the_answer_its_numbered_sources_with_text_and_its_cost() -> None:
    search = FakeSearch([SEC, ABS])
    deps, rows = _deps(_turn_replies(), search)
    sample = collect_sample(_question(), deps, get_settings(), clock=Clock())
    assert sample.answer == "".join(ANSWER).strip()
    assert (sample.refusal, sample.error, sample.intent) == (None, None, "qa")
    assert [(s.n, s.hit.chunk_id, s.hit.text) for s in sample.sources] == [
        (1, SEC.chunk_id, SEC.text), (2, ABS.chunk_id, ABS.text),
    ]  # fmt: skip
    # plan, generate, verify (a first message has no condense), plus three query embeddings
    assert [r[0] for r in rows] == ["chat-plan", "chat-generate", "chat-verify"]
    expected = chat_cost(300, 40) + chat_cost(2000, 60) + chat_cost(900, 30)
    assert sample.cost_usd == expected + 3 * embedding_cost(12)
    assert search.calls == 3  # the question and two rephrasings
    assert sample.latency_ms > 0 and sample.first_token_ms is not None
    assert sample.first_token_ms < sample.latency_ms


def test_no_chunks_is_the_no_sources_refusal() -> None:
    deps, rows = _deps([parsed_reply(_plan(), 300, 40)], FakeSearch([]))
    sample = collect_sample(_question(), deps, get_settings())
    assert (sample.refusal, sample.answer, sample.sources) == ("no_sources", None, [])
    assert [r[0] for r in rows] == ["chat-plan"]  # generate was never called


def test_the_insufficient_marker_is_the_other_refusal_and_keeps_the_sources() -> None:
    replies = [parsed_reply(_plan(), 300, 40), stream_reply([INSUFFICIENT_MARKER], 2000, 5)]
    deps, _ = _deps(replies, FakeSearch([SEC]))
    sample = collect_sample(_question(), deps, get_settings())
    assert (sample.refusal, sample.answer) == ("insufficient_sources", None)
    assert [s.hit.chunk_id for s in sample.sources] == [SEC.chunk_id]


def test_a_failed_stage_is_an_error_not_a_refusal() -> None:
    deps, _ = _deps([parsed_reply(_plan(), 300, 40)], FakeSearch([], fail=RuntimeError("down")))
    sample = collect_sample(_question(), deps, get_settings())
    assert (sample.error, sample.refusal, sample.answer) == (
        "search failed: RuntimeError", None, None,
    )  # fmt: skip


def test_the_evaluation_store_keeps_nothing() -> None:
    store = NullChatStore()
    assert store.load_history(uuid.uuid4(), 6) == []
    assert store.save_turn(object()) == 0  # type: ignore[arg-type]


# --- scoring one answer -----------------------------------------------------------------------


@dataclass
class FakeEmbed:
    """Question -> [1, 0]; 'on topic' -> [1, 0]; anything else -> [0, 1]."""

    calls: list[tuple[list[str], str]] = field(default_factory=list)

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.calls.append((list(texts), feature))
        return [[1.0, 0.0] if t in (QUESTION, "on topic") else [0.0, 1.0] for t in texts]


def _support(*oks: bool) -> SupportOutput:
    return SupportOutput(
        verdicts=[SupportVerdict(index=i, supported=ok, reason="" if ok else "no")
                  for i, ok in enumerate(oks)]
    )  # fmt: skip


def _judge_replies() -> list[Any]:
    """statements, faithfulness, relevancy, context precision, context recall, citations."""
    return [
        parsed_reply(StatementsOutput(statements=["LoRA trains matrices.", "Weights frozen."])),
        parsed_reply(_support(True, False)),
        parsed_reply(RelevancyOutput(questions=["on topic", "off topic"], noncommittal=False)),
        parsed_reply(
            UsefulOutput(
                verdicts=[
                    UsefulVerdict(n=1, useful=True, reason="states it"),
                    UsefulVerdict(n=2, useful=False, reason="other"),
                ]
            )
        ),  # fmt: skip
        parsed_reply(
            AttributedOutput(
                verdicts=[
                    AttributedVerdict(index=0, attributed=True, reason=""),
                    AttributedVerdict(index=1, attributed=False, reason=""),
                ]
            )
        ),  # fmt: skip
        parsed_reply(_support(True, True)),
    ]


def _sample(
    question: SetQuestion | None = None, answer: str | None = None, **kw: Any
) -> AnswerSample:
    base: dict[str, Any] = {
        "question": question or _question(reference_answer=REFERENCE),
        "sources": [Source(1, SEC), Source(2, ABS)],
        "answer": "".join(ANSWER).strip() if answer is None else answer,
        "refusal": None, "error": None, "intent": "qa", "cost_usd": Decimal("0.001"),
        "latency_ms": 1000.0, "first_token_ms": 400.0,
    }  # fmt: skip
    return AnswerSample(**(base | kw))


def _judge(replies: list[Any]) -> tuple[Judge, list[tuple[str, str, int, int]]]:
    client, _, rows = fake_chat(replies, model="invented-judge", temperature=0.0)
    return Judge(client, max_tokens=500, relevancy_questions=3), rows


def test_every_answer_metric_from_scripted_verdicts() -> None:
    judge, rows = _judge(_judge_replies())
    embed = FakeEmbed()
    scores = score_sample(_sample(), judge, embed)
    assert scores.metrics == pytest.approx(
        {
            "faithfulness": 0.5,  # 1 of 2 statements supported
            "response_relevancy": 0.5,  # cosine 1 and 0 with the question
            "context_precision": 1.0,  # the only useful source is ranked first
            "context_recall": 0.5,  # 1 of 2 reference sentences found
            "citation_accuracy": 2 / 3,  # [1] and [2] supported, [9] points at no source
            "uncited_sentence_share": 1 / 3,  # "That is all." cites nothing
        }
    )
    assert scores.errors == {}
    assert [r[0] for r in rows] == [
        "eval-judge-statements", "eval-judge-faithfulness", "eval-judge-relevancy",
        "eval-judge-context_precision", "eval-judge-context_recall", "eval-judge-citations",
    ]  # fmt: skip
    assert embed.calls == [([QUESTION, "on topic", "off topic"], "eval-embed-relevancy")]
    assert scores.details["citation_accuracy"]["invalid_citations"] == 1
    assert (scores.details["uncited_sentences"], scores.details["sentences"]) == (1, 3)
    assert scores.details["faithfulness"][1] == {
        "statement": "Weights frozen.", "supported": False, "reason": "no",
    }  # fmt: skip


def test_without_a_reference_answer_there_is_no_context_precision_or_recall() -> None:
    replies = _judge_replies()
    del replies[3:5]  # no precision or recall call
    judge, rows = _judge(replies)
    scores = score_sample(_sample(_question()), judge, FakeEmbed())
    assert set(scores.metrics) == {
        "faithfulness", "response_relevancy", "citation_accuracy", "uncited_sentence_share",
    }  # fmt: skip
    assert len(rows) == 4


def test_a_failing_judge_step_loses_only_its_own_metric() -> None:
    replies = _judge_replies()
    replies[1] = parsed_reply(_support(True))  # one verdict for two statements
    judge, _ = _judge(replies)
    scores = score_sample(_sample(), judge, FakeEmbed())
    assert scores.metrics["faithfulness"] is None
    assert scores.errors["faithfulness"].startswith("JudgeError: faithfulness")
    assert scores.metrics["citation_accuracy"] == pytest.approx(2 / 3)


def test_a_refused_or_failed_question_gets_no_answer_metrics_and_no_judge_call() -> None:
    judge, rows = _judge([])
    for kw in ({"refusal": "no_sources", "answer": None}, {"error": "x failed", "answer": None}):
        sample = AnswerSample(**{**_sample().__dict__, **kw})
        assert score_sample(sample, judge, FakeEmbed()).metrics == {}
    assert rows == []


def test_citation_items_split_valid_and_invalid_numbers() -> None:
    items, invalid = citation_items(
        "A [1]. B [1, 2]. C [7]. D [1][7].", [Source(1, SEC), Source(2, ABS)]
    )
    assert [(i.index, i.source, i.sentence_cites) for i in items] == [
        (0, 1, (1,)), (1, 1, (1, 2)), (1, 2, (1, 2)),
        (3, 1, (1,)),  # [7] does not exist: [1] is judged as the sentence's only source
    ]  # fmt: skip
    assert invalid == 2


@pytest.mark.parametrize("questions", [[], ["on topic", "off topic"]])
def test_a_noncommittal_answer_scores_zero_without_an_embeddings_request(
    questions: list[str],
) -> None:
    replies = _judge_replies()
    replies[2] = parsed_reply(RelevancyOutput(questions=questions, noncommittal=True))
    judge, _ = _judge(replies)
    embed = FakeEmbed()
    scores = score_sample(_sample(), judge, embed)
    assert scores.metrics["response_relevancy"] == 0.0
    assert embed.calls == []  # the score is 0 whatever the vectors: nothing is paid for


def test_a_refused_or_cut_off_judge_answer_loses_only_that_metric() -> None:
    replies = _judge_replies()
    replies[0] = parsed_reply(None, 1500, 40)  # the statements call is refused (billed)
    del replies[1]  # so no faithfulness call follows
    judge, rows = _judge(replies)
    scores = score_sample(_sample(), judge, FakeEmbed())
    assert scores.metrics["faithfulness"] is None
    assert scores.errors["faithfulness"].startswith("ChatResponseError")
    assert scores.metrics["context_recall"] == 0.5
    assert len(judge.spent) == len(rows) == 5  # the refused call is counted too


def test_any_other_judge_failure_stops_the_run() -> None:
    replies = _judge_replies()
    replies[0] = RuntimeError("invented: the model does not exist")
    judge, _ = _judge(replies)
    with pytest.raises(RuntimeError, match="does not exist"):
        score_sample(_sample(), judge, FakeEmbed())


def test_a_limit_needs_one_split() -> None:
    questions = [_question("q01", "heldout"), _question("gen-001"), _question("gen-002")]
    with pytest.raises(ValueError, match="--split"):
        select_questions(questions, "all", 1)
    assert [q.item.id for q in select_questions(questions, "generated", 1)] == ["gen-001"]
    assert len(select_questions(questions, "all", None)) == 3


def test_run_hands_every_scored_question_on_at_once() -> None:
    stored: list[str] = []
    deps, _ = _deps([parsed_reply(_plan(), 300, 40)] * 2, FakeSearch([]))  # two refusals
    judge, _ = _judge([])
    rows = run(
        [_question("gen-001"), _question("gen-002", question="Another question?")],
        deps=deps, judge=judge, embed=FakeEmbed(), settings=get_settings(),
        on_row=lambda row: stored.append(row[0].question.item.id),
    )  # fmt: skip
    assert stored == ["gen-001", "gen-002"] and len(rows) == 2


# --- aggregates -------------------------------------------------------------------------------


def _row(
    question: SetQuestion, *, refusal: str | None = None, error: str | None = None,
    faithfulness: float | None = None, latency: float = 1000.0, cost: str = "0.001",
) -> tuple[AnswerSample, SampleScores]:  # fmt: skip
    answered = refusal is None and error is None
    sample = _sample(
        question, answer=None if not answered else "x [1].", refusal=refusal, error=error,
        latency_ms=latency, cost_usd=Decimal(cost),
    )  # fmt: skip
    metrics: dict[str, float | None] = {}
    if faithfulness is not None:
        metrics["faithfulness"] = faithfulness
    return sample, SampleScores(metrics=metrics)


def test_aggregate_per_split_with_refusal_accuracy_and_false_refusals() -> None:
    rows = [
        _row(_question("q01", "heldout"), faithfulness=1.0, latency=100.0),
        _row(_question("q02", "heldout", refs=False, type="unanswerable"), refusal="no_sources"),
        _row(_question("gen-001", refs=False), faithfulness=0.5, latency=300.0),  # answered it
        _row(_question("gen-002"), refusal="insufficient_sources", latency=200.0),  # false refusal
        _row(_question("gen-003"), error="search failed: RuntimeError"),
    ]
    out = aggregate(rows)
    assert set(out) == {"all", "heldout", "generated"}
    every = out["all"]
    assert (every["faithfulness"], every["n_faithfulness"]) == (0.75, 2)
    assert (every["refusal_accuracy"], every["n_unanswerable"]) == (0.5, 2)
    assert (every["false_refusal_rate"], every["n_answerable"]) == (0.5, 2)
    assert (every["errors"], every["n_questions"]) == (1, 5)
    assert every["context_precision"] is None and every["n_context_precision"] == 0
    assert (every["latency_ms_p50"], every["latency_ms_p95"]) == (200.0, 1000.0)
    assert every["cost_per_answer_usd"] == "0.001000"
    assert out["heldout"]["refusal_accuracy"] == 1.0
    assert out["generated"]["refusal_accuracy"] == 0.0
    assert out["generated"]["false_refusal_rate"] == 1.0


def test_aggregate_leaves_out_an_empty_split() -> None:
    assert set(aggregate([_row(_question("q01", "heldout"), faithfulness=1.0)])) == {
        "all", "heldout",
    }  # fmt: skip


def test_a_question_without_reference_chunks_is_unanswerable() -> None:
    assert is_unanswerable(_question(refs=False))
    assert is_unanswerable(_question(refs=False, type="unanswerable"))
    assert not is_unanswerable(_question())


# --- the estimate -----------------------------------------------------------------------------

STATS = {
    "chat-plan": TokenStats(8, 600, 640, 60, 85),
    "chat-generate": TokenStats(8, 2800, 3400, 300, 439),
    "chat-verify": TokenStats(8, 2200, 3500, 80, 116),
}


def test_the_estimate_from_measured_turns() -> None:
    questions = [_question("gen-001", reference_answer=REFERENCE), _question("gen-002")]
    est = estimate(questions, STATS, get_settings())
    assert (est.questions, est.with_reference) == (2, 1)
    assert est.chat_requests == 3 * 2 + 4 * 2 + 2  # turns, 4 judge calls each, 2 with a reference
    assert est.embedding_requests == 3 * 2 + 2
    # Turn: 5,600 in / 440 out at 0.15 / 0.60 per 1M = $0.001104, twice. Judge, sources
    # S = 2,800 and answer A = 300: with a reference 12,400 in / 1,800 out = $0.00294;
    # without 6,800 in / 1,200 out = $0.00174. Total $0.006888.
    assert est.typical_usd == Decimal("0.0069")
    # Bound: largest inputs, every output at its cap (chat 3 x 1,600, judge 2,000 a call):
    # turn 7,540 / 4,800 = $0.004011, twice; judge 20,000 / 12,000 = $0.0102 and
    # 13,200 / 8,000 = $0.00678. Total $0.025002.
    assert est.bound_usd == Decimal("0.0250")


def test_no_measured_turns_means_no_estimate() -> None:
    with pytest.raises(ValueError, match="chat-verify"):
        estimate(
            [_question()], {k: v for k, v in STATS.items() if k != "chat-verify"}, get_settings()
        )

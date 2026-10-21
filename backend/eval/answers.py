"""Answer evaluation: each question through the chat pipeline, then the answer metrics.

    python -m eval.answers --estimate [--split heldout|generated|all] [--limit N]
    python -m eval.answers --max-usd X [--split ...] [--limit N] [--name NAME]  # PAID: ask first

`--limit N` takes the first N questions of ONE split, so it needs `--split heldout` or
`--split generated`: a quick run never silently consists of held-out questions only.

For every question, in order:
1. `collect_sample`: one chat turn exactly as POST /chat runs it (condense, plan, search,
   rerank, generate, verify, with the configured switches), but nothing is saved (a store
   that writes nothing) and nothing is traced. The sources given to generate, with their
   texts, the answer or the refusal, the cost and the latency are kept.
2. `score_sample`: the judge (`eval/judge.py`) scores the answer:
   - faithfulness: share of the answer's statements the sources support;
   - response relevancy: mean cosine similarity of the question to questions written back
     from the answer (0 for a noncommittal answer);
   - context precision and context recall: only for questions with a reference answer;
   - citation accuracy (own): share of the answer's [n] whose source supports the sentence
     (a [n] pointing at no source counts as unsupported);
   - uncited sentence share: sentences without any [n] / sentences. Citation accuracy
     cannot see an answer that cites nothing, so this is reported next to it.
   A refused question gets no answer metrics, only its refusal and, with a reference
   answer, the two context metrics (they judge the sources, not the answer; no sources at
   all gives context recall 0 without a call). A failed turn gets none. A judge step whose answer
   does not fit what was asked loses only that metric (its reason is stored); any other
   failure (an API error, a wrong model name) stops the run, so a broken judge never
   pays for a whole run of empty scores.
3. `aggregate`: per split ("heldout" = golden.jsonl, "generated", and "all"): the mean
   of every metric over the questions where it is defined (and how many those are), and:
   - refusal accuracy (own): share of the unanswerable questions that got the fallback
     (the fixed refusal), the "Behaviour" metric of the evaluation plan;
   - false refusal rate: share of the answerable questions that got it. Reported next to
     refusal accuracy because refusing everything would score 1.0 on that alone;
   - p50 / p95 latency of a turn and the mean cost of an answer (the turn's own cost from
     its `done` event; a failed turn has none), and the judge's cost apart from it.
4. Storage: the eval_runs row (name "answers:<name>", the configuration, the git commit)
   is written first with status "running"; every question's eval_results row is
   committed as soon as it is scored, so a stopped run keeps what it paid for. At the end
   the metrics are written with status "complete" (or "stopped" and the reason), and the
   run's total spend is read back from the llm_usage rows written during it.

Cost: each question costs one chat turn (about $0.001, measured), and each answered one up
to six judge calls and one embeddings request more. Every request is logged to
llm_usage. `--estimate` prints the number of requests and the cost from measured token
counts without any request; the real run spends money: ask first (CLAUDE.md).
"""

import argparse
import hashlib
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import EvalResult, EvalRun, LLMUsage
from app.ingest.embed import DenseEncoder
from app.llm.client import ChatClient, ChatResponseError
from app.observability.tracing import NoopTracer
from app.rag.citations import Source, split_sentences
from app.rag.condense import HistoryMessage
from app.rag.conversations import TurnRecord
from app.rag.generate import GENERATE_PROMPTS
from app.rag.pipeline import ChatDeps, describe, run_chat_turn
from app.rag.retrieve import RetrievalConfig
from app.rag.search import SearchHit
from eval.golden import GENERATED_PATH, GOLDEN_PATH, SetQuestion, load_question_set
from eval.judge import PROMPTS as JUDGE_PROMPTS
from eval.judge import CitationItem, Judge, JudgeError, default_judge
from eval.llm import EvalEmbedder, eval_cost
from eval.metrics import average_precision, mean, response_relevancy, share
from eval.run import git_commit, percentile, sync_items

ANSWER_METRICS = (
    "faithfulness",
    "response_relevancy",
    "context_precision",
    "context_recall",
    "citation_accuracy",
    "uncited_sentence_share",  # lower is better; computed in code, no judge call
)
SPLITS: tuple[str, ...] = ("all", "heldout", "generated")
RELEVANCY_FEATURE = "eval-embed-relevancy"  # llm_usage.feature of the relevancy embeddings


def is_unanswerable(question: SetQuestion) -> bool:
    """No reference chunk means the papers cannot answer it (eval/golden.py)."""
    return question.item.type == "unanswerable" or not question.item.reference_chunk_ids


# --- 1. one chat turn, nothing saved ----------------------------------------------------------


class NullChatStore:
    """A ChatStore that keeps nothing: an evaluation turn leaves no conversation behind."""

    def load_history(self, conversation_id: uuid.UUID, limit: int) -> list[HistoryMessage]:
        return []

    def save_turn(self, turn: TurnRecord) -> int:
        return 0


@dataclass(frozen=True)
class AnswerSample:
    question: SetQuestion
    sources: list[Source]  # what generate was given, [1]..[n], with the chunk texts
    answer: str | None  # None when refused or failed
    refusal: str | None  # "no_sources" | "insufficient_sources"
    error: str | None  # "<stage> failed: <ErrorType>"
    intent: str | None
    cost_usd: Decimal  # the turn's own cost (chat calls + question embeddings)
    latency_ms: float  # from the question to the turn's last event
    first_token_ms: float | None


def collect_sample(
    question: SetQuestion,
    deps: ChatDeps,
    settings: Settings,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> AnswerSample:
    """Run one chat turn for `question` and keep what the metrics need.

    The search function is wrapped to remember every chunk it returned, so the sources
    named in the stream's `sources` event can be given their texts.
    """
    seen: dict[uuid.UUID, SearchHit] = {}
    inner = deps.search_fn

    def recording_search(*args: Any, **kwargs: Any) -> list[SearchHit]:
        hits = inner(*args, **kwargs)
        for hit in hits:
            seen.setdefault(hit.chunk_id, hit)
        return hits

    turn_deps = replace(deps, search_fn=recording_search)
    start = clock()
    first_token: float | None = None
    pieces: list[str] = []
    numbered: list[dict[str, Any]] = []
    refusal = error = intent = None
    cost = Decimal("0")
    for event in run_chat_turn(
        question.item.question,
        conversation_id=uuid.uuid4(),
        new_conversation=True,
        history=[],
        deps=turn_deps,
        settings=settings,
    ):
        data = event.data
        if event.event == "meta":
            intent = data["intent"]
        elif event.event == "sources":
            numbered = data["sources"]
        elif event.event == "token":
            first_token = clock() if first_token is None else first_token
            pieces.append(data["text"])
        elif event.event == "refusal":
            refusal = data["reason"]
        elif event.event == "error":
            error = data["message"]
        elif event.event == "done":
            cost = Decimal(data["cost_usd"])
    elapsed = (clock() - start) * 1000
    sources = [Source(int(s["n"]), seen[uuid.UUID(s["chunk_id"])]) for s in numbered]
    answered = refusal is None and error is None
    return AnswerSample(
        question=question,
        sources=sources,
        answer="".join(pieces).strip() if answered else None,
        refusal=refusal,
        error=error,
        intent=intent,
        cost_usd=cost,
        latency_ms=round(elapsed, 1),
        first_token_ms=None if first_token is None else round((first_token - start) * 1000, 1),
    )


# --- 2. the answer metrics of one question ----------------------------------------------------


@dataclass
class SampleScores:
    metrics: dict[str, float | None] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)  # verdicts, for the per-question view
    errors: dict[str, str] = field(default_factory=dict)  # metric -> why it is missing


def citation_items(answer: str, sources: Sequence[Source]) -> tuple[list[CitationItem], int]:
    """Every [n] of the answer as (sentence, source) items for the judge, and the number
    of [n] that point at no source (they count as unsupported without asking).

    An item names only the sentence's VALID source numbers: a sentence citing [1] and a
    missing [7] is judged by the one-source rule for [1], not the lenient several-sources
    rule (judge_citation_v1)."""
    valid = {s.n for s in sources}
    items: list[CitationItem] = []
    invalid = 0
    for sentence in split_sentences(answer):
        cites = tuple(n for n in sentence.citations if n in valid)
        invalid += len(sentence.citations) - len(cites)
        items += [CitationItem(sentence.index, sentence.text, n, cites) for n in cites]
    return items, invalid


def score_sample(sample: AnswerSample, judge: Judge, embed: DenseEncoder) -> SampleScores:
    """The metrics of one question: nothing for a failed turn; for a refusal only the two
    context metrics; for an answer all of them.

    Context precision and context recall judge the sources against the reference answer,
    not the answer (as in RAGAS), so they are computed whenever the turn retrieved
    something, also when the generator then refused (insufficient sources). A refusal
    because nothing was retrieved (no sources) has context recall 0 (nothing can state the
    reference) without a judge call, and no context precision (no source to rank). So a
    configuration that makes the generator refuse more often cannot hide those questions
    from the context metrics (docs/adr/0012, decision 12).

    Each metric is computed on its own: if the judge's answer for one does not fit (a
    wrong verdict list, a refusal, output cut off), that metric is recorded as missing
    with the reason and the others still count. Any other error (an API or network
    error, a wrong model name) is raised: the run stops instead of paying on for nothing.
    """
    scores = SampleScores()
    if sample.error is not None:
        return scores
    item = sample.question.item
    sources = sample.sources

    def attempt(metric: str, compute: Callable[[], float | None]) -> None:
        try:
            scores.metrics[metric] = compute()
        except (JudgeError, ChatResponseError) as err:  # this answer only: reason stored
            scores.metrics[metric] = None
            scores.errors[metric] = describe(err)

    def faithfulness() -> float | None:
        statements = judge.statements(item.question, answer)
        verdicts = judge.faithfulness(statements, sources)
        scores.details["faithfulness"] = [
            {"statement": s, "supported": v.ok, "reason": v.reason}
            for s, v in zip(statements, verdicts, strict=True)
        ]
        return share([v.ok for v in verdicts])

    def relevancy() -> float | None:
        questions, noncommittal = judge.relevancy(answer)
        scores.details["response_relevancy"] = {
            "questions": questions,
            "noncommittal": noncommittal,
        }
        if noncommittal or not questions:  # 0 or undefined: nothing to embed
            return response_relevancy([], [], noncommittal=noncommittal)
        vectors = embed.embed([item.question, *questions], RELEVANCY_FEATURE)
        return response_relevancy(vectors[0], vectors[1:], noncommittal=noncommittal)

    def precision(reference: str) -> float | None:
        verdicts = judge.context_precision(item.question, reference, sources)
        scores.details["context_precision"] = [
            {"n": v.key, "useful": v.ok, "reason": v.reason} for v in verdicts
        ]
        return average_precision([v.ok for v in verdicts]) if verdicts else None

    def recall(reference: str) -> float | None:
        sentences = [s.text for s in split_sentences(reference)]
        verdicts = judge.context_recall(item.question, sentences, sources)
        scores.details["context_recall"] = [
            {"sentence": s, "attributed": v.ok, "reason": v.reason}
            for s, v in zip(sentences, verdicts, strict=True)
        ]
        return share([v.ok for v in verdicts])

    def citations() -> float | None:
        items, invalid = citation_items(answer, sources)
        verdicts = judge.citations(items, sources)
        scores.details["citation_accuracy"] = {
            "items": [
                {"sentence": it.index, "n": it.source, "supported": v.ok, "reason": v.reason}
                for it, v in zip(items, verdicts, strict=True)
            ],
            "invalid_citations": invalid,
        }
        return share([v.ok for v in verdicts] + [False] * invalid)

    if sample.answer is None:  # refused: the context metrics only
        if item.reference_answer:
            reference = item.reference_answer
            if sources:
                attempt("context_precision", lambda: precision(reference))
            attempt("context_recall", lambda: recall(reference))
        return scores
    answer = sample.answer
    attempt("faithfulness", faithfulness)
    attempt("response_relevancy", relevancy)
    if item.reference_answer:
        reference = item.reference_answer
        attempt("context_precision", lambda: precision(reference))
        attempt("context_recall", lambda: recall(reference))
    attempt("citation_accuracy", citations)
    sentences = split_sentences(answer)
    uncited = sum(1 for s in sentences if not s.citations)
    scores.details["uncited_sentences"] = uncited
    scores.details["sentences"] = len(sentences)
    scores.metrics["uncited_sentence_share"] = uncited / len(sentences) if sentences else None
    return scores


# --- 3. aggregates per split ------------------------------------------------------------------


Row = tuple[AnswerSample, SampleScores]


def aggregate(
    rows: Sequence[Row], metrics: Sequence[str] = ANSWER_METRICS
) -> dict[str, dict[str, Any]]:
    """Run metrics for "all", "heldout" and "generated" (a split without questions is
    left out). Means are over the questions where a metric is defined; `n_<metric>` says
    how many those were, so a mean over 3 questions is never mistaken for one over 60.
    `metrics` names the per-question values to average (the ablation runner adds its
    source metrics)."""
    out: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        chosen = [r for r in rows if split == "all" or r[0].question.split == split]
        if not chosen:
            continue
        m: dict[str, Any] = {"n_questions": len(chosen)}
        for key in metrics:
            values = [v for _, s in chosen if (v := s.metrics.get(key)) is not None]
            m[key] = mean(values)
            m[f"n_{key}"] = len(values)
        ok = [a for a, _ in chosen if a.error is None]
        unanswerable = [a for a in ok if is_unanswerable(a.question)]
        answerable = [a for a in ok if not is_unanswerable(a.question)]
        m["refusal_accuracy"] = share([a.refusal is not None for a in unanswerable])
        m["n_unanswerable"] = len(unanswerable)
        m["false_refusal_rate"] = share([a.refusal is not None for a in answerable])
        m["n_answerable"] = len(answerable)
        m["errors"] = len(chosen) - len(ok)
        latencies = [a.latency_ms for a in ok]
        m["latency_ms_p50"] = percentile(latencies, 50) if latencies else None
        m["latency_ms_p95"] = percentile(latencies, 95) if latencies else None
        m["cost_per_answer_usd"] = (
            str((sum((a.cost_usd for a in ok), Decimal("0")) / len(ok)).quantize(Decimal("1e-6")))
            if ok
            else None
        )
        out[split] = m
    return out


# --- 4. storing a run -------------------------------------------------------------------------


def run_config(
    settings: Settings, judge: Judge, questions: Sequence[SetQuestion]
) -> dict[str, Any]:
    """Everything that decides the numbers, stored on the run so runs can be compared."""

    def sha(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    return {
        "retrieval": asdict(RetrievalConfig.from_settings()),
        "chat_model": settings.chat_model,
        "chat_sources": settings.chat_sources,
        "stages": {
            "condense": settings.condense_enabled,
            "plan": settings.plan_enabled,
            "plan_rephrasings": settings.plan_rephrasings_enabled,
            "verify": settings.verify_enabled,
        },
        "generate_prompts": dict(GENERATE_PROMPTS),
        "judge": {
            "model": judge.model,
            "temperature": settings.eval_temperature,
            "prompts": dict(JUDGE_PROMPTS),
            "relevancy_questions": judge.relevancy_questions,
        },
        "embedding_model": settings.embedding_model,
        "relevancy_embedding_model": settings.eval_embedding_model,
        "questions": {
            split: sum(1 for q in questions if q.split == split)
            for split in ("heldout", "generated")
        },
        "golden_sha256": sha(GOLDEN_PATH),
        "generated_sha256": sha(GENERATED_PATH),
    }


def _last_usage_id(session: Session) -> int:
    return int(session.scalar(select(func.coalesce(func.max(LLMUsage.id), 0))) or 0)


def last_usage_id(session: Session) -> int:
    """The newest llm_usage id (0 when empty): a run's spend is every row after it."""
    return _last_usage_id(session)


def spend_since(session: Session, after_id: int) -> Decimal:
    """USD of every llm_usage row written after `after_id`."""
    value = session.scalar(
        select(func.coalesce(func.sum(LLMUsage.cost_usd), 0)).where(LLMUsage.id > after_id)
    )
    return Decimal(str(value))


def start_run(
    session: Session, name: str, *, config: dict[str, Any], git_sha: str | None
) -> EvalRun:
    """The eval_runs row ("answers:<name>"), written before the first paid call with
    status "running" and the last llm_usage id, so the run's spend can be read back."""
    run = EvalRun(
        name=f"answers:{name}",
        git_sha=git_sha,
        config=config,
        metrics={"status": "running", "llm_usage_after_id": _last_usage_id(session)},
    )
    session.add(run)
    session.flush()
    return run


def hit_record(hit: SearchHit, n: int | None = None) -> dict[str, Any]:
    """A retrieved chunk as stored with a result: what the per-question view shows. The
    text is kept because the judge's verdicts are about it, and because a chunk id of
    another chunking (an ablation index) means another text than in the chunks table."""
    return {
        **({"n": n} if n is not None else {}),
        "chunk_id": str(hit.chunk_id),
        "arxiv_id": hit.arxiv_id,
        "title": hit.title,
        "section_path": hit.section_path,
        "chunk_type": hit.chunk_type,
        "score": hit.score,
        "text": hit.text,
    }


def result_metrics(row: Row) -> dict[str, Any]:
    """One question's eval_results.metrics: its scores, how the turn went, the judge's
    verdicts and the sources the answer was given (with their texts)."""
    sample, scores = row
    item = sample.question.item
    return {
        **scores.metrics,
        "golden_id": item.id,
        "split": sample.question.split,
        "type": item.type,
        "refusal": sample.refusal,
        "error": sample.error,
        "intent": sample.intent,
        "latency_ms": sample.latency_ms,
        "first_token_ms": sample.first_token_ms,
        "cost_usd": str(sample.cost_usd),
        "judge_errors": scores.errors,
        "details": scores.details,
        "sources": [hit_record(s.hit, s.n) for s in sample.sources],
    }


def store_result(session: Session, run_id: int, item_id: int, row: Row) -> None:
    """One question's eval_results row."""
    sample, _ = row
    session.add(
        EvalResult(
            run_id=run_id,
            item_id=item_id,
            retrieved_chunk_ids=[s.hit.chunk_id for s in sample.sources],
            answer=sample.answer,
            metrics=result_metrics(row),
        )
    )
    session.flush()


def finish_run(
    session: Session,
    run_id: int,
    rows: Sequence[Row],
    *,
    judge_cost_usd: Decimal,
    status: str = "complete",
) -> EvalRun:
    """The run's metrics, its status and its real spend: every llm_usage row written
    since the run started (the chat turns, the judge, the embeddings; also anything
    else that used the same database meanwhile, so run evaluations alone)."""
    run = session.get(EvalRun, run_id)
    if run is None:
        raise LookupError(f"eval_runs {run_id} does not exist")
    after = int(run.metrics.get("llm_usage_after_id", 0))
    spend = session.scalar(
        select(func.coalesce(func.sum(LLMUsage.cost_usd), 0)).where(LLMUsage.id > after)
    )
    run.metrics = {  # a new dict, so SQLAlchemy sees the JSONB change
        **aggregate(rows),
        "status": status,
        "llm_usage_after_id": after,
        "llm_usage_until_id": _last_usage_id(session),
        "spend_usd": str(spend),
        "judge_cost_usd": str(judge_cost_usd),
    }
    session.flush()
    return run


def store_run(
    session: Session,
    name: str,
    rows: Sequence[Row],
    *,
    config: dict[str, Any],
    judge_cost_usd: Decimal,
    git_sha: str | None,
) -> EvalRun:
    """A whole run at once (start, every result, finish), in one session."""
    item_ids = sync_items(session, [a.question.item for a, _ in rows])
    run = start_run(session, name, config=config, git_sha=git_sha)
    for row in rows:
        store_result(session, run.id, item_ids[row[0].question.item.id], row)
    return finish_run(session, run.id, rows, judge_cost_usd=judge_cost_usd)


# --- the cost estimate ------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenStats:
    """Measured tokens of one llm_usage feature: requests, mean and max in and out."""

    requests: int
    mean_in: float
    max_in: int
    mean_out: float
    max_out: int


TURN_FEATURES = ("chat-plan", "chat-generate", "chat-verify")


def measured_stats(session: Session) -> dict[str, TokenStats]:
    """Token counts of the real chat turns so far (llm_usage; read-only)."""
    rows = session.execute(
        select(
            LLMUsage.feature,
            func.count(),
            func.avg(LLMUsage.tokens_in),
            func.max(LLMUsage.tokens_in),
            func.avg(LLMUsage.tokens_out),
            func.max(LLMUsage.tokens_out),
        )
        .where(LLMUsage.feature.in_(TURN_FEATURES))
        .group_by(LLMUsage.feature)
    ).all()
    return {
        f: TokenStats(int(n), float(ai), int(mi), float(ao), int(mo))
        for f, n, ai, mi, ao, mo in rows
    }


@dataclass(frozen=True)
class Estimate:
    questions: int
    with_reference: int
    chat_requests: int
    embedding_requests: int
    typical_usd: Decimal
    bound_usd: Decimal


def estimate(
    questions: Sequence[SetQuestion], stats: dict[str, TokenStats], settings: Settings
) -> Estimate:
    """Requests and cost of an answer run, from measured chat turns (no request is made).

    Turn: plan + generate + verify as measured (typical = means; bound = the largest
    measured input and the configured output caps), and up to three question embeddings
    (a fraction of a cent, counted as requests only). Judge, per answered question, sized
    from the measured generate call, whose input is the numbered sources plus the
    question (S) and whose output is the answer (A): statements (A in, A out),
    faithfulness (S + A in), relevancy (A in), citations (S + A in), plus context
    precision and recall (S in each) when there is a reference answer. Typical judge
    outputs are taken as the answer's size; the bound uses the judge's output cap on
    every call and assumes every question is answered. The bound is not a hard limit:
    its inputs are the largest MEASURED ones, and a turn whose sources are longer than
    any measured turn costs more (each source chunk is at most `chunk_hard_max_tokens`).
    """
    missing = [f for f in TURN_FEATURES if f not in stats]
    if missing:
        raise ValueError(f"no measured chat turns for {missing}; cannot estimate from data")
    n = len(questions)
    refs = sum(1 for q in questions if q.item.reference_answer)
    plan, gen, ver = (stats[f] for f in TURN_FEATURES)
    s = settings

    def chat_usd(tokens_in: float, tokens_out: float) -> Decimal:
        price_in = Decimal(str(s.chat_price_in_per_1m))
        price_out = Decimal(str(s.chat_price_out_per_1m))
        cost = Decimal(str(tokens_in)) * price_in + Decimal(str(tokens_out)) * price_out
        return cost / 1_000_000

    turn_typical = chat_usd(
        plan.mean_in + gen.mean_in + ver.mean_in, plan.mean_out + gen.mean_out + ver.mean_out
    )
    turn_bound = chat_usd(
        plan.max_in + gen.max_in + ver.max_in,
        3 * s.chat_max_output_tokens_explain,  # each call's output cap (explain is the largest)
    )

    def judge_usd(sources: float, answer: float, out_each: float, with_ref: bool) -> Decimal:
        calls_in = [answer, sources + answer, answer, sources + answer]
        if with_ref:
            calls_in += [sources, sources]
        tokens_in = sum(calls_in)
        return eval_cost(int(tokens_in), int(out_each * len(calls_in)))

    typical = Decimal("0")
    bound = Decimal("0")
    for q in questions:
        has_ref = bool(q.item.reference_answer)
        typical += turn_typical + judge_usd(gen.mean_in, gen.mean_out, gen.mean_out, has_ref)
        bound += turn_bound + judge_usd(gen.max_in, s.chat_max_output_tokens_explain,
                                        s.eval_judge_max_output_tokens, has_ref)  # fmt: skip
    judge_calls = 4 * n + 2 * refs
    return Estimate(
        questions=n,
        with_reference=refs,
        chat_requests=3 * n + judge_calls,
        embedding_requests=3 * n + n,  # up to 3 question embeddings + 1 relevancy request
        typical_usd=typical.quantize(Decimal("1e-4")),
        bound_usd=bound.quantize(Decimal("1e-4")),
    )


# --- the whole run ----------------------------------------------------------------------------


def run(
    questions: Sequence[SetQuestion],
    *,
    deps: ChatDeps,
    judge: Judge,
    embed: DenseEncoder,
    settings: Settings,
    progress: Callable[[str], None] = lambda line: None,
    on_row: Callable[[Row], None] = lambda row: None,
) -> list[Row]:
    """Collect and score every question, in order; `on_row` stores each one at once."""
    rows: list[Row] = []
    for k, question in enumerate(questions, start=1):
        sample = collect_sample(question, deps, settings)
        scores = score_sample(sample, judge, embed)
        rows.append((sample, scores))
        on_row((sample, scores))
        state = sample.error or (f"refused ({sample.refusal})" if sample.refusal else "answered")
        progress(f"[{k}/{len(questions)}] {question.item.id}: {state}")
    return rows


def select_questions(
    questions: Sequence[SetQuestion], split: str, limit: int | None
) -> list[SetQuestion]:
    """The questions of `split` ("all": both), the first `limit` of them. A limit needs
    one split: the held-out questions come first in the set, so "all" with a limit
    would quietly be a held-out-only run."""
    if limit is not None and split == "all":
        raise ValueError("--limit needs --split heldout or --split generated")
    chosen = [q for q in questions if split == "all" or q.split == split]
    return chosen[:limit] if limit is not None else chosen


def _table(metrics: dict[str, dict[str, Any]]) -> str:
    keys = (*ANSWER_METRICS, "refusal_accuracy", "false_refusal_rate")
    lines = ["| split | n | " + " | ".join(keys) + " |", "|" + " --- |" * (len(keys) + 2)]
    for split, m in metrics.items():
        cells = ["n/a" if m.get(k) is None else f"{m[k]:.3f}" for k in keys]
        lines.append(f"| {split} | {m['n_questions']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.answers", description=__doc__.split("\n")[0])
    parser.add_argument("--estimate", action="store_true", help="requests and cost only (free)")
    parser.add_argument("--split", choices=SPLITS, default="all")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--name", default="default", help="stored as eval_runs.name 'answers:NAME'")
    parser.add_argument("--max-usd", default=None, help="hard spend cap (needed for a real run)")
    args = parser.parse_args(argv)

    from app.db.session import session_scope

    settings = get_settings()
    try:
        questions = select_questions(load_question_set(), args.split, args.limit)
    except ValueError as err:
        parser.error(str(err))
    if not questions:
        parser.error(f"no questions in split {args.split!r}")
    if args.estimate:
        with session_scope() as session:
            est = estimate(questions, measured_stats(session), settings)
        print(
            f"{est.questions} questions ({est.with_reference} with a reference answer): "
            f"at most {est.chat_requests} chat requests and {est.embedding_requests} embeddings "
            f"requests; about ${est.typical_usd} (bound ${est.bound_usd}). Nothing was sent."
        )
        return 0
    if not args.max_usd:
        parser.error("a real run needs --max-usd, the hard spend cap (see --estimate)")

    from eval.budget import BudgetExhausted, install_budget

    # The egress guard refuses any request that could take the worst case past the cap
    # (eval/budget.py); a question during which it refused one is not stored.
    budget = install_budget(Decimal(args.max_usd), settings)
    judge = default_judge()
    deps = ChatDeps(chat=ChatClient(), store=NullChatStore(), tracer=NoopTracer())
    with session_scope() as session:
        item_ids = sync_items(session, [q.item for q in questions])
        run_id = start_run(
            session, args.name, config=run_config(settings, judge, questions),
            git_sha=git_commit(),
        ).id  # fmt: skip

    rows: list[Row] = []

    def keep(row: Row) -> None:
        budget.check()
        rows.append(row)
        with session_scope() as session:  # committed per question
            store_result(session, run_id, item_ids[row[0].question.item.id], row)

    status = "complete"
    try:
        run(
            questions, deps=deps, judge=judge, embed=EvalEmbedder(), settings=settings,
            progress=lambda line: print(line, file=sys.stderr, flush=True), on_row=keep,
        )  # fmt: skip
    except BaseException as err:  # also Ctrl-C: record what was done, then stop
        if budget.guard.tripped is not None and not isinstance(err, BudgetExhausted):
            status = f"stopped: budget: the egress guard stopped the run: {budget.guard.tripped}"
            raise BudgetExhausted(status) from err
        status = f"stopped: {describe(err)}"
        raise
    finally:
        judge_cost = sum((u.cost_usd for u in judge.spent), Decimal("0"))
        with session_scope() as session:
            done = finish_run(session, run_id, rows, judge_cost_usd=judge_cost, status=status)
            print(
                f"eval_runs {run_id}: answers:{args.name}, {status}, {len(rows)} of "
                f"{len(questions)} questions; spend ${done.metrics['spend_usd']} "
                f"(judge ${judge_cost})"
            )
        if rows:
            print(_table(aggregate(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

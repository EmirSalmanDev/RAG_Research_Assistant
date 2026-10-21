"""The ablation runner (`make eval-ablations`): the plan's 8 ablations and the reranker
comparisons, step by step, each run stored in eval_runs with its configuration and commit.

    python -m eval.ablations list                                   # steps and variants (free)
    python -m eval.ablations estimate --step retrieval              # requests, cost, cap (free)
    python -m eval.ablations run --step retrieval --max-usd 1.50    # PAID: ask first
    python -m eval.ablations select --step retrieval                # the rule, on stored runs

Steps, in order (eval/variants.py): retrieval (ablations 1-5), rewriting (6), chunking (7),
embedding (8), and the reranker comparison. A step after the first runs on the winner of
the step before ("best of the above"), which `select` chooses by the rule fixed in
`eval/selection.py` on GENERATED questions only and stores as a `selection:<step>` row;
`--base VARIANT` names a base by hand instead (recorded as such). The chunking and embedding
variants search their own collections, built first with `python -m eval.indexes`.

Money: the answers part of every variant is paid (a chat turn and the judge per question),
and so is embedding questions missing from the local cache (`--embed`). A paid run needs
`--max-usd`: the egress guard then refuses any request that could take the run's worst-case
spend over it, so the cap is hard (eval/budget.py). `estimate` prints the requests, the
expected cost and the cap a full run needs, without any request.

The reranker comparison runs every variant in its own process (`--isolate`, the default for
that step), so each run's peak memory is that variant's own; the cap is shared: each child
gets what the ones before it left.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import EvalItem, EvalResult, EvalRun
from app.rag.rerank import RerankerUnavailable, model_cache_dir
from eval.answers import Estimate, TokenStats, estimate, measured_stats, select_questions
from eval.budget import BudgetExhausted, install_budget
from eval.golden import GoldenError, SetQuestion, for_tuning, load_question_set
from eval.run import CACHE_PATH, QueryVectorCache, embedding_estimate, git_commit
from eval.runner import (
    ConfigMismatch,
    IndexMissing,
    MissingQueryVectors,
    PartResult,
    RunnerDeps,
    SqlRunStore,
    Suite,
    run_part,
)
from eval.selection import (
    Decision,
    RerankerResult,
    VariantResults,
    choose,
    qualify_reranker,
)
from eval.variants import (
    PARTS,
    PREVIOUS,
    RERANKER_BASE,
    STEPS,
    IndexSpec,
    Part,
    Resolved,
    Step,
    find,
    resolve,
    resolved_from_record,
    variants_of,
)

EXIT_SETUP, EXIT_VECTORS, EXIT_RERANKER, EXIT_BUDGET, EXIT_MISMATCH = 2, 3, 4, 5, 6


# --- which variants a step runs, on which base ------------------------------------------------


def latest_selection(session: Session, step: Step) -> EvalRun | None:
    return session.scalars(
        select(EvalRun).where(EvalRun.name == f"selection:{step}").order_by(EvalRun.id.desc())
    ).first()


def base_for(
    step: Step, s: Settings, session: Session | None, by_hand: str | None
) -> Resolved | None:
    """The base a step builds on: none for the retrieval step, a4 for the reranker step,
    else the stored winner of the step before (or the variant named by hand)."""
    if step == "retrieval":
        return None
    if step == "reranker":
        return resolve(find(RERANKER_BASE, s), s)
    if by_hand is not None:
        variant = find(by_hand, s)
        earlier = base_for(variant.step, s, session, None) if variant.step in PREVIOUS else None
        return resolve(variant, s, earlier)
    previous = PREVIOUS[step]
    row = latest_selection(session, previous) if session is not None else None
    if row is None:
        raise LookupError(
            f"no selection for step {previous!r} yet: run `select --step {previous}` first, "
            "or name the base with --base VARIANT"
        )
    return resolved_from_record(row.config["winner_resolved"], s)


def plan_step(
    step: Step, s: Settings, base: Resolved | None, only: Sequence[str] | None = None
) -> list[Resolved]:
    chosen = variants_of(step, s)
    if only:
        unknown = sorted(set(only) - {v.id for v in chosen})
        if unknown:
            raise KeyError(f"not variants of step {step!r}: {unknown}")
        chosen = [v for v in chosen if v.id in only]
    return [resolve(v, s, base) for v in chosen]


# --- estimate ---------------------------------------------------------------------------------


def reranker_cached(model: str, s: Settings) -> bool:
    """Are the model's files in the local FastEmbed cache (no load, no network)?"""
    repo = model_cache_dir() / f"models--{model.replace('/', '--')}"
    return any(repo.glob("snapshots/*/onnx/*.onnx")) or any(repo.glob("snapshots/*/*.onnx"))


def guard_bound_per_question(stats: Mapping[str, TokenStats], s: Settings) -> Decimal:
    """What one question adds to the egress guard's worst-case bound, roughly: every
    request's output cap at the output price plus its input bytes (4 bytes per measured
    token, doubled for JSON and the prompts) at the input price. The guard counts real
    bytes, so this only suggests a cap; the cap itself is enforced by the guard."""
    price_in = Decimal(str(max(s.chat_price_in_per_1m, s.eval_price_in_per_1m))) / 1_000_000
    price_out = Decimal(str(max(s.chat_price_out_per_1m, s.eval_price_out_per_1m))) / 1_000_000
    largest_in = max(st.max_in for st in stats.values())
    chat_out = 3 * s.chat_max_output_tokens_explain
    judge_out = 6 * s.eval_judge_max_output_tokens
    bytes_in = 8 * largest_in * (3 + 6)  # 3 chat calls and up to 6 judge calls
    return (Decimal(chat_out + judge_out) * price_out + Decimal(bytes_in) * price_in).quantize(
        Decimal("1e-4")
    )


def estimate_lines(
    resolved: Sequence[Resolved], questions: Sequence[SetQuestion], s: Settings,
    stats: Mapping[str, TokenStats] | None, parts: Sequence[Part],
) -> tuple[list[str], Decimal]:  # fmt: skip
    """One line per variant and the suggested --max-usd for all of them (free)."""
    lines: list[str] = []
    cap = Decimal("0")
    for r in resolved:
        notes: list[str] = []
        overrides = r.overrides(s)
        model = overrides.get("embedding_model", s.embedding_model)
        dim = overrides.get("embedding_dim", s.embedding_dim)
        mode = overrides.get("search_mode", s.search_mode)
        if "retrieval" in parts and "retrieval" in r.variant.parts and mode != "bm25":
            cache = QueryVectorCache(CACHE_PATH, model, dim)
            missing = cache.missing([q.item.question for q in questions])
            if missing:
                tokens, requests, cost = embedding_estimate(missing)
                notes.append(f"{len(missing)} question vectors missing ({tokens} tokens, "
                             f"~${cost} at {s.embedding_model} prices; --embed)")  # fmt: skip
        rerank_model = overrides.get("reranker_model", s.reranker_model)
        if overrides.get("rerank_enabled", s.rerank_enabled) and not reranker_cached(
            rerank_model, s
        ):
            notes.append(f"reranker {rerank_model} not downloaded (fetch-reranker; ask first)")
        if r.index != IndexSpec.configured(s):
            notes.append(f"index {r.index.collection(s)} (build: python -m eval.indexes)")
        answers = "answers" in parts and "answers" in r.variant.parts
        if answers and stats is not None:
            est: Estimate = estimate(questions, dict(stats), s)
            bound = guard_bound_per_question(stats, s) * len(questions)
            cap += bound
            notes.insert(0, f"answers: {est.chat_requests} chat + {est.embedding_requests} "
                            f"embedding requests, ~${est.typical_usd} (measured bound "
                            f"${est.bound_usd}; cap for the guard ~${bound})")  # fmt: skip
        elif answers:
            notes.insert(0, "answers: no measured chat turns, no estimate")
        lines.append(f"{r.variant.id:18} " + ("; ".join(notes) or "free"))
    return lines, cap


# --- run --------------------------------------------------------------------------------------


def _print_results(results: Sequence[PartResult]) -> None:
    for r in results:
        overall = r.metrics.get("all", {})
        keys = ("recall@8", "coverage@8", "context_recall", "faithfulness", "source_coverage")
        shown = ", ".join(f"{k} {overall[k]:.3f}" for k in keys if overall.get(k) is not None)
        spend = r.metrics.get("spend_usd")
        print(f"eval_runs {r.run_id}: {r.name}, {r.status}; {shown}; spend ${spend}")


def run_variants(
    resolved: Sequence[Resolved], questions: Sequence[SetQuestion], parts: Sequence[Part],
    deps: RunnerDeps, suite: Suite, *, embed: bool, out: list[PartResult] | None = None,
) -> list[PartResult]:  # fmt: skip
    """Every part of every variant in this process, in order; stops at the first failure.
    Finished parts are appended to `out` as they finish (so a caller sees them after a stop)."""
    out = [] if out is None else out
    for r in resolved:
        for part in parts:
            if part in r.variant.parts:
                out.append(run_part(part, r, questions, deps, suite, embed_missing_vectors=embed))
    return out


def _child(
    args: argparse.Namespace, variant: str, suite: Suite, max_usd: Decimal | None, summary: Path
) -> int:
    """Run one variant in a fresh process (its own memory peak); returns its exit code."""
    cmd = [
        sys.executable, "-m", "eval.ablations", "run", "--step", args.step, "--variants", variant,
        "--parts", ",".join(args.parts), "--split", args.split, "--no-isolate",
        "--suite-id", suite.id, "--suite-started", suite.started_at, "--summary-out", str(summary),
    ]  # fmt: skip
    if args.base:
        cmd += ["--base", args.base]
    if args.embed:
        cmd.append("--embed")
    if max_usd is not None:
        cmd += ["--max-usd", str(max_usd)]
    return subprocess.run(cmd, check=False).returncode


def cmd_run(args: argparse.Namespace) -> int:
    from app.db.session import session_scope

    s = get_settings()
    questions = select_questions(load_question_set(), args.split, None)
    with session_scope() as session:
        base = base_for(args.step, s, session, args.base)
    resolved = plan_step(args.step, s, base, args.variants)
    paid = args.embed or any(
        "answers" in r.variant.parts for r in resolved if "answers" in args.parts
    )
    max_usd = Decimal(args.max_usd) if args.max_usd else None
    if paid and max_usd is None:
        print("a paid run needs --max-usd (see `estimate`); nothing was run", file=sys.stderr)
        return EXIT_SETUP
    suite = Suite("ablation", args.step, **({"id": args.suite_id, "started_at": args.suite_started}
                                            if args.suite_id else {}))  # fmt: skip

    isolate = args.isolate if args.isolate is not None else args.step == "reranker"
    if isolate and len(resolved) > 0:
        left = max_usd
        for r in resolved:
            with tempfile.TemporaryDirectory() as tmp:
                summary = Path(tmp) / "summary.json"
                code = _child(args, r.variant.id, suite, left, summary)
                used = (
                    Decimal(json.loads(summary.read_text())["bound_usd"])
                    if summary.is_file()
                    else None
                )
            if code != 0:
                print(
                    f"{r.variant.id}: stopped (exit {code}); later variants not run",
                    file=sys.stderr,
                )
                return code
            if left is not None and used is not None:
                left -= used
        return 0

    deps = cli_deps(max_usd if paid else None, s)
    results: list[PartResult] = []
    try:
        code = guarded(
            lambda: run_variants(
                resolved, questions, args.parts, deps, suite, embed=args.embed, out=results
            )
        )
    finally:
        if args.summary_out:
            bound = deps.budget.bound_usd if deps.budget is not None else Decimal("0")
            Path(args.summary_out).write_text(json.dumps({"bound_usd": str(bound)}))
    _print_results(results)
    return code


def cli_deps(max_usd: Decimal | None, s: Settings) -> RunnerDeps:
    """The real stores, progress on stderr, and the hard spend cap when one is given."""
    deps = RunnerDeps(
        store=SqlRunStore(), progress=lambda line: print(line, file=sys.stderr, flush=True)
    )
    if max_usd is not None:
        deps.budget = install_budget(max_usd, s)
    return deps


def guarded(work: Callable[[], object]) -> int:
    """Run `work`; the known ways a run stops become a message and an exit code (what was
    done before is stored, with the run marked stopped)."""
    try:
        work()
    except MissingQueryVectors as err:
        print(err, file=sys.stderr)
        return EXIT_VECTORS
    except RerankerUnavailable as err:
        print(f"reranker: {err}", file=sys.stderr)
        return EXIT_RERANKER
    except BudgetExhausted as err:
        print(f"stopped by the spend cap: {err}", file=sys.stderr)
        return EXIT_BUDGET
    except ConfigMismatch as err:
        print(str(err), file=sys.stderr)
        return EXIT_MISMATCH
    except (IndexMissing, GoldenError, LookupError, KeyError) as err:
        print(str(err), file=sys.stderr)
        return EXIT_SETUP
    return 0


# --- select -----------------------------------------------------------------------------------


def _answers_runs(session: Session, step: Step, suite_id: str | None) -> dict[str, EvalRun]:
    """The newest complete answers run of each variant of `step` (one suite if named)."""
    rows = session.scalars(
        select(EvalRun).where(EvalRun.name.like("ablation:%:answers")).order_by(EvalRun.id.desc())
    ).all()
    out: dict[str, EvalRun] = {}
    for run in rows:
        cfg = run.config or {}
        if cfg.get("suite", {}).get("step") != step or run.metrics.get("status") != "complete":
            continue
        if suite_id is not None and cfg["suite"]["id"] != suite_id:
            continue
        out.setdefault(cfg["variant"]["variant"], run)
    return out


def _generated_results(
    session: Session, run: EvalRun, tuning_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Per generated question of the run, its stored metrics. Held-out questions are
    dropped here, before the rule sees anything (the rule checks again)."""
    rows = session.execute(
        select(EvalItem.id, EvalResult.metrics)
        .join(EvalItem, EvalItem.id == EvalResult.item_id)
        .where(EvalResult.run_id == run.id)
    ).all()
    out: dict[str, dict[str, Any]] = {}
    for _, metrics in rows:
        qid = str(metrics.get("golden_id"))
        if qid in tuning_ids and metrics.get("split") == "generated":
            out[qid] = dict(metrics)
    return out


def _same_basis(runs: Mapping[str, EvalRun]) -> None:
    """Runs compared must share the question set, the judge and the code."""

    def key(run: EvalRun) -> tuple[Any, ...]:
        cfg = run.config
        return (cfg.get("generated_sha256"), json.dumps(cfg.get("judge"), sort_keys=True),
                run.git_sha)  # fmt: skip

    keys = {key(r) for r in runs.values()}
    if len(keys) > 1:
        raise ValueError("the runs differ in question set, judge or commit; run the step again "
                         f"as one suite: {sorted(map(str, keys))}")  # fmt: skip


def select_step(session: Session, step: Step, s: Settings, suite_id: str | None) -> dict[str, Any]:
    """Apply the rule to the stored runs of a step; store and return the decision."""
    tuning_ids = {q.item.id for q in for_tuning(load_question_set())}
    if not tuning_ids:
        raise ValueError("no generated questions: the rule only looks at generated questions")
    if step == "reranker":
        return _select_reranker(session, s, suite_id, tuning_ids)
    variants = variants_of(step, s)
    runs = _answers_runs(session, step, suite_id)
    missing = [v.id for v in variants if v.id not in runs]
    if missing:
        raise LookupError(f"no complete answers run for {missing} in step {step!r}")
    _same_basis(runs)
    results = {
        v.id: VariantResults(
            v.id, runs[v.id].id, _generated_results(session, runs[v.id], tuning_ids)
        )
        for v in variants
    }
    incumbent = next(v for v in variants if v.incumbent)
    decision: Decision = choose(
        results[incumbent.id], [results[v.id] for v in variants if not v.incumbent]
    )
    winner_run = runs[decision.winner]
    record = {
        **decision.as_dict(),
        "status": "complete",
        "runs": {v: r.id for v, r in runs.items()},
    }
    session.add(EvalRun(
        name=f"selection:{step}", git_sha=git_commit(),
        config={"step": step, "questions": "generated only (for_tuning)",
                "winner_resolved": winner_run.config["variant"],
                "from_suite": winner_run.config.get("suite")},
        metrics=record,
    ))  # fmt: skip
    session.flush()
    return record


def _select_reranker(
    session: Session, s: Settings, suite_id: str | None, tuning_ids: set[str]
) -> dict[str, Any]:
    variants = variants_of("reranker", s)
    retrieval: dict[str, EvalRun] = {}
    for run in session.scalars(
        select(EvalRun).where(EvalRun.name.like("ablation:%:retrieval")).order_by(EvalRun.id.desc())
    ):
        cfg = run.config or {}
        if (
            cfg.get("suite", {}).get("step") != "reranker"
            or run.metrics.get("status") != "complete"
        ):
            continue
        if suite_id is not None and cfg["suite"]["id"] != suite_id:
            continue
        retrieval.setdefault(cfg["variant"]["variant"], run)
    answers = _answers_runs(session, "reranker", suite_id)
    missing = [v.id for v in variants if v.id not in retrieval]
    if missing:
        raise LookupError(f"no complete retrieval run for {missing} in the reranker step")

    def lists(run: EvalRun) -> dict[str, list[str]]:
        return {qid: [h["chunk_id"] for h in m.get("hits", [])]
                for qid, m in _generated_results(session, run, tuning_ids).items()}  # fmt: skip

    reference = next(v for v in variants if v.incumbent)
    ref_lists = lists(retrieval[reference.id])

    def result(v_id: str) -> RerankerResult:
        run = retrieval[v_id]
        variant = find(v_id, s)
        effective = run.config["effective"]
        median = run.metrics.get("all", {}).get("rerank_ms_p50")
        ranking_changes = (
            "search_candidates" in variant.overrides or "reranker_model" in variant.overrides
        )
        answer_run = answers.get(v_id)
        return RerankerResult(
            variant=v_id,
            candidates=int(effective["retrieval"]["candidates"]),
            median_rerank_s=None if median is None else round(float(median) / 1000, 2),
            peak_rss_mib=run.metrics.get("peak_rss_mib"),
            isolated=not run.config.get("process", {}).get("runs_before"),
            ranking_changes=ranking_changes,
            same_lists_as_reference=None if ranking_changes else lists(run) == ref_lists,
            answers=(VariantResults(v_id, answer_run.id,
                                    _generated_results(session, answer_run, tuning_ids))
                     if answer_run is not None else None),
        )  # fmt: skip

    record = qualify_reranker(
        result(reference.id), [result(v.id) for v in variants if not v.incumbent]
    )
    record["status"] = "complete"
    record["runs"] = {v: r.id for v, r in retrieval.items()}
    session.add(EvalRun(name="selection:reranker", git_sha=git_commit(),
                        config={"step": "reranker", "questions": "generated only (for_tuning)"},
                        metrics=record))  # fmt: skip
    session.flush()
    return record


# --- command line -----------------------------------------------------------------------------


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.ablations", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="steps and variants (free)")
    for name in ("estimate", "run", "select"):
        p = sub.add_parser(name)
        p.add_argument("--step", choices=STEPS, required=True)
        if name == "select":
            p.add_argument("--suite-id", default=None, help="only runs of this suite")
            continue
        p.add_argument("--variants", type=_csv, default=None, help="comma-separated variant ids")
        p.add_argument("--parts", type=_csv, default=list(PARTS))
        p.add_argument("--split", choices=("all", "heldout", "generated"), default="all")
        p.add_argument(
            "--base", default=None, help="the base variant, instead of the stored winner"
        )
        p.add_argument("--embed", action="store_true", help="embed missing question vectors (paid)")
        if name == "run":
            p.add_argument("--max-usd", default=None, help="hard spend cap (needed for paid parts)")
            p.add_argument(
                "--isolate",
                dest="isolate",
                action="store_true",
                default=None,
                help="each variant in its own process (default for the reranker step)",
            )
            p.add_argument("--no-isolate", dest="isolate", action="store_false")
            p.add_argument("--suite-id", default=None, help=argparse.SUPPRESS)
            p.add_argument("--suite-started", default=None, help=argparse.SUPPRESS)
            p.add_argument("--summary-out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    s = get_settings()

    if args.command == "list":
        for step in STEPS:
            print(f"{step}:")
            for v in variants_of(step, s):
                mark = "*" if v.incumbent else " "
                print(f"  {mark} {v.id:18} {v.label}; settings {dict(v.overrides)}; "
                      f"index {dict(v.index)}; parts {','.join(v.parts)}")  # fmt: skip
        return 0

    from app.db.session import session_scope

    if args.command == "select":
        try:
            with session_scope() as session:
                record = select_step(session, args.step, s, args.suite_id)
        except (LookupError, ValueError) as err:
            print(str(err), file=sys.stderr)
            return EXIT_SETUP
        print(json.dumps(record, indent=2, default=str))
        return 0

    bad = [p for p in args.parts if p not in PARTS]
    if bad:
        parser.error(f"unknown parts {bad}; choose from {list(PARTS)}")
    if args.command == "estimate":
        questions = select_questions(load_question_set(), args.split, None)
        with session_scope() as session:
            try:
                base = base_for(args.step, s, session, args.base)
            except LookupError as err:
                print(f"{err}; estimating on the configured settings", file=sys.stderr)
                base = None
            stats = measured_stats(session)
        resolved = plan_step(args.step, s, base, args.variants)
        lines, cap = estimate_lines(resolved, questions, s, stats or None, args.parts)
        print(
            f"step {args.step}: {len(questions)} questions, base {base.variant.id if base else '-'}"
        )
        print("\n".join(lines))
        print(f"suggested --max-usd for the whole step: {cap} (nothing was sent)")
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())

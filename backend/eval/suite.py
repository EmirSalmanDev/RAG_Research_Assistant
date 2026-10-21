"""`make eval-smoke` and `make eval`: the configured pipeline, end to end, and the report
that CI posts (docs/adr/0012).

    python -m eval.suite smoke --estimate                 # requests, cost, cap (free)
    python -m eval.suite smoke --max-usd 0.30 [--embed]   # PAID: 15 questions, ask first
    python -m eval.suite full --estimate
    python -m eval.suite full --max-usd 1.50 [--embed]    # PAID: every question, ask first
    python -m eval.suite report --runs 31,32 --out DIR    # markdown table + JSON (free)

Both run the app as it is configured (no setting changed: the variant "configured") in
the two parts of the ablation runner (`eval/runner.py`): retrieval (free with cached
question vectors) and answers (a chat turn and the judge per question). Runs are named
`smoke:configured:<part>` and `full:configured:<part>`.

The smoke set is 15 GENERATED questions, never held-out ones: a pull request is judged on
it, and a held-out question that decides whether a change goes in is a held-out question
tuned on. It is picked by a fixed rule, round robin over the question types (fact,
compare, filter, find_papers, unanswerable), each type in the order of the SHA-256 of the
question id, so it is the same 15 for as long as generated.jsonl does not change, and
every type is in it. The full run takes every question, both splits, reported apart.

A paid run needs `--max-usd`, a hard cap enforced by the egress guard (eval/budget.py).
`--report-dir DIR` writes `report.md` (the tables CI posts on the pull request) and
`runs.json` (the runs with every per-question score and the configuration, for the
artifact) at the end, also after a stop. Neither file holds chunk texts: only chunk ids,
the answers and the judge's verdicts.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import EvalItem, EvalResult, EvalRun
from eval.ablations import (
    EXIT_SETUP,
    cli_deps,
    estimate_lines,
    guarded,
    run_variants,
)
from eval.answers import measured_stats
from eval.golden import GoldenError, SetQuestion, load_question_set
from eval.runner import PartResult, Suite
from eval.variants import CONFIGURED, PARTS, resolve

SMOKE_SIZE = 15
TYPE_ORDER = ("fact", "compare", "filter", "find_papers", "unanswerable")


class SmokeSetError(ValueError):
    """Not enough generated questions for the smoke set."""


def smoke_set(questions: Sequence[SetQuestion], size: int = SMOKE_SIZE) -> list[SetQuestion]:
    """The smoke questions: generated only, round robin over the types (see the module doc)."""
    generated = [q for q in questions if q.split == "generated"]
    if len(generated) < size:
        raise SmokeSetError(
            f"the smoke set needs {size} generated questions, there are {len(generated)} "
            "(generate and review them first: python -m eval.make_testset)"
        )

    def order(q: SetQuestion) -> str:
        return hashlib.sha256(q.item.id.encode()).hexdigest()

    types = [*TYPE_ORDER, None]
    queues = {t: sorted((q for q in generated if q.item.type == t), key=order) for t in types}
    chosen: list[SetQuestion] = []
    while len(chosen) < size:
        for t in types:
            if queues[t] and len(chosen) < size:
                chosen.append(queues[t].pop(0))
    return chosen


# --- the report -------------------------------------------------------------------------------

ANSWER_COLUMNS = (
    ("faithfulness", "faithful"), ("response_relevancy", "relevancy"),
    ("context_precision", "ctx prec"), ("context_recall", "ctx recall"),
    ("citation_accuracy", "citations"), ("uncited_sentence_share", "uncited"),
    ("refusal_accuracy", "refusal acc"), ("false_refusal_rate", "false refusal"),
    ("source_coverage", "src coverage"), ("latency_ms_p50", "p50 ms"),
    ("cost_per_answer_usd", "$/answer"),
)  # fmt: skip
RETRIEVAL_COLUMNS = (
    ("recall@8", "Recall@8"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10"),
    ("coverage@8", "coverage@8"), ("retrieve_ms_p50", "p50 ms"),
)  # fmt: skip
TEXT_KEYS = ("hits", "sources")  # per-result lists whose items carry a chunk text


def _cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.1f}" if value >= 10 else f"{value:.3f}"
    return str(value)


def markdown(runs: Sequence[dict[str, Any]]) -> str:
    """The tables for a pull request comment: one per run, a row per split."""
    lines: list[str] = []
    for run in runs:
        m, cfg = run["metrics"], run["config"]
        kind = cfg.get("kind", "?")
        columns = ANSWER_COLUMNS if kind == "answers" else RETRIEVAL_COLUMNS
        sha = (run.get("git_sha") or "?")[:12]
        lines += [
            f"**{run['name']}** (eval_runs {run['id']}, commit `{sha}`): {m.get('status')}, "
            f"spend ${m.get('spend_usd')}",
            "",
            "| split | n | " + " | ".join(label for _, label in columns) + " |",
            "|" + " --- |" * (len(columns) + 2),
        ]
        for split in ("all", "heldout", "generated"):
            if split in m:
                row = m[split]
                cells = " | ".join(_cell(row.get(key)) for key, _ in columns)
                lines.append(f"| {split} | {row.get('n_questions')} | {cells} |")
        lines.append("")
    lines.append(
        "Held-out (golden) questions are reported, never used to choose a setting; the "
        "smoke set holds generated questions only."
    )
    return "\n".join(lines)


def _without_texts(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    for key in TEXT_KEYS:
        if isinstance(out.get(key), list):
            out[key] = [{k: v for k, v in item.items() if k != "text"} for item in out[key]]
    return out


def export_runs(session: Session, run_ids: Sequence[int]) -> list[dict[str, Any]]:
    """The runs with their per-question results, chunk texts left out (an artifact of a
    public repository must not re-publish paper text)."""
    out: list[dict[str, Any]] = []
    for run_id in run_ids:
        run = session.get(EvalRun, run_id)
        if run is None:
            raise LookupError(f"eval_runs {run_id} does not exist")
        rows = session.execute(
            select(EvalItem.question, EvalResult)
            .join(EvalItem, EvalItem.id == EvalResult.item_id)
            .where(EvalResult.run_id == run_id)
            .order_by(EvalResult.id)
        ).all()
        out.append({
            "id": run.id, "name": run.name, "created_at": run.created_at.isoformat(),
            "git_sha": run.git_sha, "config": run.config, "metrics": run.metrics,
            "results": [
                {"question": question,
                 "retrieved_chunk_ids": [str(c) for c in r.retrieved_chunk_ids],
                 "answer": r.answer, "metrics": _without_texts(r.metrics)}
                for question, r in rows
            ],
        })  # fmt: skip
    return out


def write_report(run_ids: Sequence[int], out_dir: Path) -> None:
    from app.db.session import session_scope

    with session_scope() as session:
        runs = export_runs(session, run_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(markdown(runs) + "\n")
    (out_dir / "runs.json").write_text(json.dumps(runs, indent=1, default=str) + "\n")


def suite_run_ids(session: Session, suite_id: str) -> list[int]:
    """The runs of one suite, oldest first (a selection is not a run of the suite)."""
    query = (
        select(EvalRun.id)
        .where(EvalRun.config["suite"]["id"].astext == suite_id)
        .order_by(EvalRun.id)
    )
    return list(session.scalars(query))


# --- command line -----------------------------------------------------------------------------


def _questions(kind: str) -> list[SetQuestion]:
    questions = load_question_set()
    return smoke_set(questions) if kind == "smoke" else questions


def cmd_suite(args: argparse.Namespace) -> int:
    from app.db.session import session_scope

    s = get_settings()
    try:
        questions = _questions(args.command)
    except (SmokeSetError, GoldenError) as err:
        print(str(err), file=sys.stderr)
        return EXIT_SETUP
    resolved = [resolve(CONFIGURED, s)]
    if args.estimate:
        with session_scope() as session:
            stats = measured_stats(session)
        lines, cap = estimate_lines(resolved, questions, s, stats or None, args.parts)
        print(f"{args.command}: {len(questions)} questions")
        print("\n".join(lines))
        print(f"suggested --max-usd: {cap} (nothing was sent)")
        return 0
    paid = args.embed or "answers" in args.parts
    if paid and not args.max_usd:
        print("a paid run needs --max-usd (see --estimate); nothing was run", file=sys.stderr)
        return EXIT_SETUP
    deps = cli_deps(Decimal(args.max_usd) if paid else None, s)
    suite = Suite(args.command, "suite")
    results: list[PartResult] = []
    code = guarded(
        lambda: run_variants(
            resolved, questions, args.parts, deps, suite, embed=args.embed, out=results
        )
    )
    with session_scope() as session:
        run_ids = suite_run_ids(session, suite.id)  # stopped runs too
    for run_id in run_ids:
        print(f"eval_runs {run_id} ({args.command})")
    if args.report_dir and run_ids:
        write_report(run_ids, Path(args.report_dir))
        print(f"report: {Path(args.report_dir) / 'report.md'}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.suite", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "full"):
        p = sub.add_parser(name)
        p.add_argument("--estimate", action="store_true", help="requests and cost only (free)")
        p.add_argument("--max-usd", default=None, help="hard spend cap (needed for paid parts)")
        p.add_argument("--embed", action="store_true", help="embed missing question vectors (paid)")
        p.add_argument(
            "--parts", type=lambda v: [x for x in v.split(",") if x], default=list(PARTS)
        )
        p.add_argument("--report-dir", default=None, help="write report.md and runs.json here")
    report = sub.add_parser("report", help="markdown and JSON of stored runs (free)")
    report.add_argument("--runs", default=None, help="comma-separated eval_runs ids")
    report.add_argument("--suite-id", default=None)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.command == "report":
        from app.db.session import session_scope

        if bool(args.runs) == bool(args.suite_id):
            parser.error("give --runs or --suite-id")
        if args.runs:
            run_ids = [int(v) for v in args.runs.split(",") if v.strip()]
        else:
            with session_scope() as session:
                run_ids = suite_run_ids(session, args.suite_id)
        try:
            write_report(run_ids, Path(args.out))
        except LookupError as err:
            print(str(err), file=sys.stderr)
            return EXIT_SETUP
        print(Path(args.out) / "report.md")
        return 0
    bad = [p for p in args.parts if p not in PARTS]
    if bad:
        parser.error(f"unknown parts {bad}; choose from {list(PARTS)}")
    return cmd_suite(args)


if __name__ == "__main__":
    raise SystemExit(main())

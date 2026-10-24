"use client";

import { useState } from "react";

import { useEvalResults, useEvalRun } from "@/lib/api/queries";
import {
  SPLITS,
  defaultScore,
  filterSplit,
  headlineRows,
  joinCounts,
  joinResults,
  rowDirection,
  scoreKeys,
  scorePair,
  sortRows,
  splitMetrics,
} from "@/lib/evals/compare";
import type { JoinedRow, SortOrder, Split } from "@/lib/evals/compare";
import {
  direction,
  formatDiff,
  formatValue,
  metricDef,
  metricLabel,
} from "@/lib/evals/metrics";
import type { Direction } from "@/lib/evals/metrics";
import { runState, runTitle } from "@/lib/evals/runs";
import type { EvalRunSummary } from "@/lib/evals/runs";
import { formatDate } from "@/lib/format";
import { cn } from "@/lib/utils";

import { RunStatus } from "./run-status";

const SELECT_CLASS =
  "h-9 w-full rounded-md border border-input bg-card px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

const DIRECTION_CLASS: Record<Direction, string> = {
  better: "text-emerald-700 dark:text-emerald-400",
  worse: "text-destructive",
  same: "text-muted-foreground",
};

/** B - A, coloured by whether it is an improvement, with the word for screen readers. */
function Diff({ metric, diff }: { metric: string; diff: number | null }) {
  const dir = direction(metric, diff);
  return (
    <span
      data-testid="diff"
      data-direction={dir ?? "none"}
      className={cn("tabular-nums", dir && DIRECTION_CLASS[dir])}
    >
      {formatDiff(metric, diff)}
      {dir && dir !== "same" && <span className="sr-only"> ({dir})</span>}
    </span>
  );
}

function RunPicker({
  side,
  runs,
  value,
  onPick,
}: {
  side: "a" | "b";
  runs: readonly EvalRunSummary[];
  value: number | null;
  onPick: (side: "a" | "b", runId: number | null) => void;
}) {
  const id = `run-${side}`;
  const run = runs.find((r) => r.id === value) ?? null;
  return (
    <div className="min-w-0 flex-1 space-y-1.5">
      <label htmlFor={id} className="block text-sm font-medium">
        Run {side.toUpperCase()}
      </label>
      <select
        id={id}
        value={value ?? ""}
        onChange={(e) =>
          onPick(side, e.target.value === "" ? null : Number(e.target.value))
        }
        className={SELECT_CLASS}
      >
        <option value="">Choose a run</option>
        {runs.map((r) => {
          const state = runState(r.status);
          return (
            <option key={r.id} value={r.id}>
              {runTitle(r)} · {formatDate(r.created_at)}
              {state !== "complete"
                ? ` · ${state === "unknown" ? "no status" : state}`
                : ""}
            </option>
          );
        })}
      </select>
      {run && (
        <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <RunStatus status={run.status} />
          <span>
            {run.n_questions ?? "?"} questions
            {run.git_sha ? `, commit ${run.git_sha.slice(0, 7)}` : ""}
          </span>
        </p>
      )}
    </div>
  );
}

function SplitFilter({
  split,
  onSplit,
}: {
  split: Split;
  onSplit: (split: Split) => void;
}) {
  return (
    <fieldset className="space-y-2">
      <legend className="text-sm font-medium">Questions</legend>
      <div className="flex flex-wrap gap-1.5">
        {SPLITS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            aria-pressed={split === id}
            onClick={() => onSplit(id)}
            className={cn(
              "rounded-md border px-2.5 py-1 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              split === id
                ? "border-primary bg-primary text-primary-foreground"
                : "bg-card hover:bg-accent",
            )}
          >
            {label}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted-foreground" data-testid="heldout-note">
        Held-out questions are the hand-checked golden set. They are reported
        here and never used to choose settings: the selection between variants
        reads the generated questions only, so the held-out numbers stay an
        honest check.
      </p>
    </fieldset>
  );
}

/** The headline numbers of the chosen split; for one split they come from the run itself. */
function useSplitNumbers(run: EvalRunSummary | null, split: Split) {
  const detail = useEvalRun(run?.id ?? null, split !== "all");
  if (!run)
    return { numbers: null, pending: false, error: null, missing: false };
  if (split === "all")
    return {
      numbers: run.metrics,
      pending: false,
      error: null,
      missing: false,
    };
  const numbers = detail.data
    ? splitMetrics(detail.data.all_metrics, split)
    : null;
  return {
    numbers,
    pending: detail.isPending,
    error: detail.error,
    missing: detail.isSuccess && numbers === null,
  };
}

function HeadlineTable({
  runA,
  runB,
  split,
}: {
  runA: EvalRunSummary | null;
  runB: EvalRunSummary | null;
  split: Split;
}) {
  const a = useSplitNumbers(runA, split);
  const b = useSplitNumbers(runB, split);
  const rows = headlineRows(a.numbers, b.numbers);
  const splitName = SPLITS.find((s) => s.id === split)?.label.toLowerCase();
  return (
    <div className="space-y-2">
      <h3 className="font-medium">Headline metrics</h3>
      {(a.pending || b.pending) && (
        <p role="status" className="text-sm text-muted-foreground">
          Loading the {splitName} numbers&hellip;
        </p>
      )}
      {[a.error, b.error].map(
        (error, i) =>
          error && (
            <p key={i} role="alert" className="text-sm text-destructive">
              {error.message}
            </p>
          ),
      )}
      {[
        { run: runA, missing: a.missing },
        { run: runB, missing: b.missing },
      ].map(
        ({ run, missing }) =>
          missing &&
          run && (
            <p key={run.id} className="text-sm text-muted-foreground">
              Run #{run.id} has no numbers for {splitName} (none of its
              questions are in that split, or it stores only overall numbers).
            </p>
          ),
      )}
      {rows.length > 0 && (
        <div className="overflow-x-auto rounded-md border bg-card">
          <table
            className="w-full min-w-[28rem] border-collapse text-left text-sm"
            data-testid="headline-table"
          >
            <caption className="sr-only">
              Headline metrics of run A and run B, {splitName}
            </caption>
            <thead>
              <tr className="border-b bg-muted/50">
                <th scope="col" className="p-2 font-medium">
                  Metric
                </th>
                {runA && (
                  <th scope="col" className="p-2 text-right font-medium">
                    A #{runA.id}
                  </th>
                )}
                {runB && (
                  <th scope="col" className="p-2 text-right font-medium">
                    B #{runB.id}
                  </th>
                )}
                {runA && runB && (
                  <th scope="col" className="p-2 text-right font-medium">
                    B − A
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const known = metricDef(row.key);
                return (
                  <tr
                    key={row.key}
                    data-metric={row.key}
                    className={cn(
                      "border-b last:border-b-0",
                      !known && "text-muted-foreground",
                    )}
                  >
                    <th scope="row" className="p-2 font-normal">
                      {metricLabel(row.key)}
                      {known?.better === "lower" && (
                        <span className="ml-1 text-xs text-muted-foreground">
                          (lower is better)
                        </span>
                      )}
                    </th>
                    {runA && (
                      <td className="p-2 text-right tabular-nums">
                        {formatValue(row.key, row.a)}
                      </td>
                    )}
                    {runB && (
                      <td className="p-2 text-right tabular-nums">
                        {formatValue(row.key, row.b)}
                      </td>
                    )}
                    {runA && runB && (
                      <td className="p-2 text-right">
                        <Diff metric={row.key} diff={row.diff} />
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const OUTCOME_CLASS: Record<string, string> = {
  answered: "",
  retrieved: "",
  refused: "text-amber-700 dark:text-amber-300",
  error: "text-destructive",
};

function Outcome({ row }: { row: JoinedRow["a"] }) {
  if (!row)
    return (
      <span className="text-xs text-muted-foreground">not in this run</span>
    );
  return (
    <span
      className={cn("text-xs", OUTCOME_CLASS[row.outcome])}
      title={row.refusal ?? undefined}
    >
      {row.outcome}
    </span>
  );
}

function QuestionsTable({
  runA,
  runB,
  split,
  preferredScore,
  onOpen,
  activeKey,
}: {
  runA: EvalRunSummary | null;
  runB: EvalRunSummary | null;
  split: Split;
  preferredScore: string;
  onOpen: (row: JoinedRow) => void;
  activeKey: string | null;
}) {
  // A selection has no per-question results: nothing is asked for it.
  const resultsA = useEvalResults(runA?.id ?? null, runA?.kind !== "selection");
  const resultsB = useEvalResults(runB?.id ?? null, runB?.kind !== "selection");
  const [score, setScore] = useState<string | null>(null);
  const [order, setOrder] = useState<SortOrder>("run");

  const sides = [
    { run: runA, q: resultsA, name: "A" },
    { run: runB, q: resultsB, name: "B" },
  ].filter((s) => s.run !== null);
  const pending = sides.some(
    (s) => s.run?.kind !== "selection" && s.q.isPending,
  );
  const errors = sides.flatMap((s) => (s.q.error ? [s.q.error.message] : []));
  const without = sides.filter(
    (s) =>
      s.run?.kind === "selection" ||
      (s.q.data && s.q.data.results.length === 0),
  );

  const joined = joinResults(
    resultsA.data?.results ?? null,
    resultsB.data?.results ?? null,
  );
  const keys = scoreKeys(joined);
  const shownScore =
    score !== null && keys.includes(score)
      ? score
      : defaultScore(keys, preferredScore);
  const rows = shownScore
    ? sortRows(filterSplit(joined, split), shownScore, order)
    : filterSplit(joined, split);
  const counts = joinCounts(rows);

  return (
    <div className="space-y-3">
      <h3 className="font-medium">Questions</h3>
      {pending && (
        <p role="status" className="text-sm text-muted-foreground">
          Loading the questions&hellip;
        </p>
      )}
      {errors.map((message) => (
        <p key={message} role="alert" className="text-sm text-destructive">
          {message}
        </p>
      ))}
      {without.map((s) => (
        <p
          key={s.name}
          className="text-sm text-muted-foreground"
          data-testid="no-results"
        >
          Run {s.name} (#{s.run?.id}){" "}
          {s.run?.kind === "selection"
            ? "is a selection: it chose between other runs and has no per-question results of its own."
            : "stored no per-question results."}
        </p>
      ))}

      {joined.length > 0 && (
        <>
          <div className="flex flex-wrap items-end gap-3">
            <div className="space-y-1.5">
              <label
                htmlFor="question-score"
                className="block text-sm text-muted-foreground"
              >
                Score
              </label>
              <select
                id="question-score"
                value={shownScore ?? ""}
                onChange={(e) => setScore(e.target.value)}
                className={cn(SELECT_CLASS, "w-56")}
              >
                {keys.map((key) => (
                  <option key={key} value={key}>
                    {metricLabel(key)}
                  </option>
                ))}
              </select>
            </div>
            {runB && runA && (
              <div className="space-y-1.5">
                <label
                  htmlFor="question-order"
                  className="block text-sm text-muted-foreground"
                >
                  Order
                </label>
                <select
                  id="question-order"
                  value={order}
                  onChange={(e) => setOrder(e.target.value as SortOrder)}
                  className={cn(SELECT_CLASS, "w-56")}
                >
                  <option value="run">Run order</option>
                  <option value="worst">Largest step back first</option>
                  <option value="best">Largest gain first</option>
                </select>
              </div>
            )}
            <p
              className="text-sm text-muted-foreground"
              data-testid="join-counts"
            >
              {rows.length} question{rows.length === 1 ? "" : "s"}
              {runA && runB
                ? `: ${counts.both} in both runs, ${counts.onlyA} only in A, ${counts.onlyB} only in B.`
                : "."}{" "}
              Click a question for its chunks, answer and verdicts.
            </p>
          </div>

          <div className="overflow-x-auto rounded-md border bg-card">
            <table
              className="w-full min-w-[44rem] border-collapse text-left text-sm"
              data-testid="questions-table"
            >
              <caption className="sr-only">
                Per-question {shownScore ? metricLabel(shownScore) : "scores"}{" "}
                of the runs
              </caption>
              <thead>
                <tr className="border-b bg-muted/50">
                  <th scope="col" className="p-2 font-medium">
                    Question
                  </th>
                  <th scope="col" className="p-2 font-medium">
                    Split
                  </th>
                  {runA && (
                    <th scope="col" className="p-2 font-medium">
                      Outcome A
                    </th>
                  )}
                  {runB && (
                    <th scope="col" className="p-2 font-medium">
                      Outcome B
                    </th>
                  )}
                  {runA && (
                    <th scope="col" className="p-2 text-right font-medium">
                      A
                    </th>
                  )}
                  {runB && (
                    <th scope="col" className="p-2 text-right font-medium">
                      B
                    </th>
                  )}
                  {runA && runB && (
                    <th scope="col" className="p-2 text-right font-medium">
                      B − A
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => {
                  const pair = shownScore ? scorePair(row, shownScore) : null;
                  return (
                    <tr
                      key={row.key}
                      data-testid="question-row"
                      data-question={row.questionId ?? row.key}
                      data-direction={
                        shownScore
                          ? (rowDirection(row, shownScore) ?? "none")
                          : "none"
                      }
                      className={cn(
                        "border-b align-top last:border-b-0",
                        activeKey === row.key && "bg-accent/60",
                      )}
                    >
                      <td className="max-w-md p-2">
                        <button
                          type="button"
                          onClick={() => onOpen(row)}
                          className="text-left underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        >
                          {row.question}
                        </button>
                        <span className="block text-xs text-muted-foreground">
                          {row.questionId ?? "no question id"}
                          {row.type ? ` · ${row.type}` : ""}
                        </span>
                      </td>
                      <td className="p-2 text-xs">
                        {row.split === "heldout"
                          ? "held-out"
                          : (row.split ?? "—")}
                      </td>
                      {runA && (
                        <td className="p-2">
                          <Outcome row={row.a} />
                        </td>
                      )}
                      {runB && (
                        <td className="p-2">
                          <Outcome row={row.b} />
                        </td>
                      )}
                      {runA && (
                        <td className="p-2 text-right tabular-nums">
                          {shownScore ? formatValue(shownScore, pair?.a) : ""}
                        </td>
                      )}
                      {runB && (
                        <td className="p-2 text-right tabular-nums">
                          {shownScore ? formatValue(shownScore, pair?.b) : ""}
                        </td>
                      )}
                      {runA && runB && (
                        <td className="p-2 text-right">
                          {shownScore && (
                            <Diff
                              metric={shownScore}
                              diff={pair?.diff ?? null}
                            />
                          )}
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {rows.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No question of these runs is in this split.
            </p>
          )}
        </>
      )}
    </div>
  );
}

// Two runs side by side. Run A alone lists its questions; with run B too, every number
// gets the difference B - A, coloured green when it is an improvement (for latency,
// cost, uncited sentences and false refusals, lower is better).
export function RunCompare({
  runs,
  runA,
  runB,
  onPick,
  preferredScore,
  onOpen,
  activeKey,
}: {
  runs: readonly EvalRunSummary[];
  runA: number | null;
  runB: number | null;
  onPick: (side: "a" | "b", runId: number | null) => void;
  preferredScore: string;
  onOpen: (row: JoinedRow) => void;
  activeKey: string | null;
}) {
  const [split, setSplit] = useState<Split>("all");
  const a = runs.find((r) => r.id === runA) ?? null;
  const b = runs.find((r) => r.id === runB) ?? null;
  const differentKinds = a && b && a.kind !== b.kind;

  return (
    <section aria-labelledby="compare-heading" className="space-y-5">
      <h2 id="compare-heading" className="text-lg font-semibold">
        Compare two runs
      </h2>
      <div className="flex flex-col gap-4 sm:flex-row">
        <RunPicker side="a" runs={runs} value={runA} onPick={onPick} />
        <RunPicker side="b" runs={runs} value={runB} onPick={onPick} />
      </div>
      {!a && !b && (
        <p className="text-sm text-muted-foreground">
          Pick a run as A to see its questions, and a second one as B to compare
          them. The A and B buttons in the runs table do the same.
        </p>
      )}
      {differentKinds && (
        <p className="text-sm text-muted-foreground">
          These runs are of different kinds ({a.kind} and {b.kind}), so most of
          their metrics do not line up.
        </p>
      )}
      {(a || b) && (
        <>
          <SplitFilter split={split} onSplit={setSplit} />
          <HeadlineTable runA={a} runB={b} split={split} />
          <QuestionsTable
            runA={a}
            runB={b}
            split={split}
            preferredScore={preferredScore}
            onOpen={onOpen}
            activeKey={activeKey}
          />
        </>
      )}
    </section>
  );
}

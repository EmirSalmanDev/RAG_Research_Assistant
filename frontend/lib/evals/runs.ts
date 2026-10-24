// Evaluation runs as the /evals page shows them: their state (complete, stopped,
// running), the series a run belongs to on the trend chart, and short names. Plain
// TypeScript.

import type { components } from "../api/schema";

export type EvalRunSummary = components["schemas"]["EvalRunSummaryOut"];

// --- state ------------------------------------------------------------------------------------

/**
 * complete: the run finished. stopped: it ended early ("stopped: <reason>", e.g. the
 * spend cap), so its numbers cover only the questions done before. running: it has not
 * finished (or died without writing its end). unknown: no status was stored.
 * Only "complete" is ever presented as a finished result.
 */
export type RunState = "complete" | "stopped" | "running" | "unknown";

export function runState(status: string | null | undefined): RunState {
  const s = (status ?? "").trim().toLowerCase();
  if (s === "complete") return "complete";
  if (s.startsWith("stopped")) return "stopped";
  if (s.startsWith("running")) return "running";
  return "unknown";
}

export const STATE_LABEL: Record<RunState, string> = {
  complete: "complete",
  stopped: "stopped",
  running: "running",
  unknown: "no status",
};

/** Why a run stopped ("budget: ..."), or null. */
export function stopReason(status: string | null | undefined): string | null {
  if (runState(status) !== "stopped") return null;
  const reason = (status ?? "").replace(/^\s*stopped\s*:?\s*/i, "").trim();
  return reason || null;
}

// --- series -----------------------------------------------------------------------------------

/**
 * The line a run belongs to on the trend chart, or null for a run that is never drawn
 * (a selection has no metrics of its own to trend).
 *
 * Runs of the ablation runner and the suites carry a variant: the same variant in the
 * same suite kind is one line ("answers|ablation|a3-hybrid"). The suite kind is part of
 * the key because a smoke suite answers fewer questions than a full one, so their
 * numbers are not points on the same line. Older runs have no variant: their name is
 * the key ("hybrid", "answers:default").
 */
export function seriesKey(run: EvalRunSummary): string | null {
  if (run.kind === "selection") return null;
  if (run.variant) return `${run.kind}|${run.suite ?? "-"}|${run.variant}`;
  return `${run.kind}|name|${run.name ?? `run-${run.id}`}`;
}

/** A short name for a series: "hybrid (RRF) · ablation", "hybrid", "default". */
export function seriesLabel(run: EvalRunSummary): string {
  if (run.variant) {
    const name = run.label || run.variant;
    return run.suite ? `${name} · ${run.suite}` : name;
  }
  const name = run.name ?? `run ${run.id}`;
  return name.replace(/^answers:/, "");
}

/** A run in one line, for the pickers and the panel: "#31 answers · hybrid (RRF) · ablation". */
export function runTitle(run: EvalRunSummary): string {
  if (run.kind === "selection")
    return `#${run.id} selection · ${run.step ?? run.name ?? ""}`.trim();
  return `#${run.id} ${run.kind} · ${seriesLabel(run)}`;
}

/** Oldest first; runs made in the same instant keep their id order. */
export function chronological(
  runs: readonly EvalRunSummary[],
): EvalRunSummary[] {
  return [...runs].sort(
    (a, b) =>
      Date.parse(a.created_at) - Date.parse(b.created_at) || a.id - b.id,
  );
}

export interface Series {
  key: string;
  label: string;
  kind: EvalRunSummary["kind"];
  /** oldest first */
  runs: EvalRunSummary[];
}

/**
 * Every series, in the order of their most recent run (newest first). The order decides
 * the colours (lib/evals/trend.ts), and it is computed from ALL runs, not from the runs
 * of one metric: switching the metric never repaints a line.
 */
export function groupSeries(runs: readonly EvalRunSummary[]): Series[] {
  const byKey = new Map<string, Series>();
  for (const run of chronological(runs)) {
    const key = seriesKey(run);
    if (key === null) continue;
    let series = byKey.get(key);
    if (!series) {
      series = { key, label: seriesLabel(run), kind: run.kind, runs: [] };
      byKey.set(key, series);
    }
    series.runs.push(run);
    series.label = seriesLabel(run); // the newest run's label wins if a label changed
  }
  const newest = (s: Series) => s.runs[s.runs.length - 1]!;
  return [...byKey.values()].sort((a, b) => {
    const x = newest(a);
    const y = newest(b);
    return Date.parse(y.created_at) - Date.parse(x.created_at) || y.id - x.id;
  });
}

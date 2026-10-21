// Two runs side by side: their headline metrics with the difference B - A, and their
// questions joined on the question id. Plain TypeScript.

import type { components } from "../api/schema";
import {
  difference,
  direction,
  metricDef,
  orderKeys,
  toNumber,
} from "./metrics.ts";

export type EvalResultRow = components["schemas"]["EvalResultRowOut"];
export type Split = "all" | "heldout" | "generated";
export const SPLITS: readonly { id: Split; label: string }[] = [
  { id: "all", label: "All questions" },
  { id: "heldout", label: "Held-out (golden)" },
  { id: "generated", label: "Generated" },
];

// --- headline metrics -------------------------------------------------------------------------

export interface MetricRow {
  key: string;
  a: number | null;
  b: number | null;
  /** B - A, null when either run lacks the metric */
  diff: number | null;
}

/** One row per metric either run has: the known metrics first, then the rest. */
export function headlineRows(
  a: Readonly<Record<string, number | null>> | null,
  b: Readonly<Record<string, number | null>> | null,
): MetricRow[] {
  const keys = orderKeys([...Object.keys(a ?? {}), ...Object.keys(b ?? {})]);
  return keys.map((key) => {
    const x = a?.[key] ?? null;
    const y = b?.[key] ?? null;
    return { key, a: x, b: y, diff: difference(x, y) };
  });
}

/**
 * The numbers of one split from a run's stored metrics (GET /evals/runs/{id},
 * all_metrics): {"all": {...}, "heldout": {...}, "generated": {...}}. Null when the run
 * has no such split (an older retrieval run stores flat metrics only; a run without
 * held-out questions has no "heldout").
 */
export function splitMetrics(
  allMetrics: Readonly<Record<string, unknown>> | null | undefined,
  split: Exclude<Split, "all">,
): Record<string, number | null> | null {
  const part = allMetrics?.[split];
  if (typeof part !== "object" || part === null || Array.isArray(part))
    return null;
  const out: Record<string, number | null> = {};
  for (const [key, value] of Object.entries(part)) {
    const n = toNumber(value);
    if (n !== null) out[key] = n;
  }
  return out;
}

// --- per question -----------------------------------------------------------------------------

export interface JoinedRow {
  /** "q:q01" or, for a result without a question id, "item:<item_id>" */
  key: string;
  questionId: string | null;
  question: string;
  split: string | null;
  type: string | null;
  a: EvalResultRow | null;
  b: EvalResultRow | null;
}

export function rowKey(row: EvalResultRow): string {
  return row.question_id ? `q:${row.question_id}` : `item:${row.item_id}`;
}

/**
 * The questions of A and B joined on the question id: A's order first, then the
 * questions only B has. A question only one run answered has the other side null (runs
 * on different question sets, or a run that stopped early).
 */
export function joinResults(
  a: readonly EvalResultRow[] | null | undefined,
  b: readonly EvalResultRow[] | null | undefined,
): JoinedRow[] {
  const rows = new Map<string, JoinedRow>();
  const add = (row: EvalResultRow, side: "a" | "b") => {
    const key = rowKey(row);
    const joined = rows.get(key) ?? {
      key,
      questionId: row.question_id,
      question: row.question,
      split: row.split,
      type: row.type,
      a: null,
      b: null,
    };
    if (joined[side] === null) joined[side] = row; // a repeated id keeps the first row
    joined.split ??= row.split;
    joined.type ??= row.type;
    rows.set(key, joined);
  };
  for (const row of a ?? []) add(row, "a");
  for (const row of b ?? []) add(row, "b");
  return [...rows.values()];
}

/** The rows of one split ("all" keeps every row). */
export function filterSplit(
  rows: readonly JoinedRow[],
  split: Split,
): JoinedRow[] {
  return split === "all" ? [...rows] : rows.filter((r) => r.split === split);
}

export interface JoinCounts {
  both: number;
  onlyA: number;
  onlyB: number;
}

export function joinCounts(rows: readonly JoinedRow[]): JoinCounts {
  return {
    both: rows.filter((r) => r.a && r.b).length,
    onlyA: rows.filter((r) => r.a && !r.b).length,
    onlyB: rows.filter((r) => !r.a && r.b).length,
  };
}

/** Every per-question score key either run has, known metrics first. */
export function scoreKeys(rows: readonly JoinedRow[]): string[] {
  const keys: string[] = [];
  for (const r of rows)
    keys.push(
      ...Object.keys(r.a?.scores ?? {}),
      ...Object.keys(r.b?.scores ?? {}),
    );
  return orderKeys(keys);
}

/** The per-question score shown first: `preferred` (the chart's metric) if it is there. */
export function defaultScore(
  keys: readonly string[],
  preferred: string,
): string | null {
  if (keys.includes(preferred)) return preferred;
  return keys.find((k) => metricDef(k)) ?? keys[0] ?? null;
}

export interface ScorePair {
  a: number | null;
  b: number | null;
  diff: number | null;
}

export function scorePair(row: JoinedRow, key: string): ScorePair {
  const a = row.a?.scores[key] ?? null;
  const b = row.b?.scores[key] ?? null;
  return { a, b, diff: difference(a, b) };
}

export type SortOrder = "run" | "worst" | "best";

/**
 * Sorted by how a score changed. "worst": the largest step in the wrong direction first
 * (down for recall, up for latency); "best" the reverse; "run" keeps the join order.
 * Rows without a difference go last in every order.
 */
export function sortRows(
  rows: readonly JoinedRow[],
  key: string,
  order: SortOrder,
): JoinedRow[] {
  if (order === "run") return [...rows];
  const lowerIsBetter = metricDef(key)?.better === "lower";
  // How much better B is than A: positive = an improvement.
  const gain = (r: JoinedRow) => {
    const d = scorePair(r, key).diff;
    return d === null ? null : lowerIsBetter ? -d : d;
  };
  return rows
    .map((row, i) => ({ row, i, g: gain(row) }))
    .sort((x, y) => {
      if (x.g === null || y.g === null)
        return (x.g === null ? 1 : 0) - (y.g === null ? 1 : 0) || x.i - y.i;
      return (order === "worst" ? x.g - y.g : y.g - x.g) || x.i - y.i;
    })
    .map((x) => x.row);
}

/** The change of a score as better / worse / same (null: no direction known or no pair). */
export function rowDirection(row: JoinedRow, key: string) {
  return direction(key, scorePair(row, key).diff);
}

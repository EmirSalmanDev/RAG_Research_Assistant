// The metric trend chart in plain numbers (components/evals/trend-chart.tsx draws them as
// SVG): which runs have the chosen metric, one line per series, the x order and the
// y scale. Plain TypeScript.
//
// x is the run order (oldest left), not the date: runs come in bursts (a suite writes a
// dozen runs within an hour, then nothing for days), and evenly spaced runs are easier
// to read than a clump. The tick labels and the tooltips carry the dates.

import { METRICS, metricDef } from "./metrics.ts";
import type { MetricDef } from "./metrics.ts";
import { chronological, groupSeries, runState } from "./runs.ts";
import type { EvalRunSummary } from "./runs.ts";

/** Colour slots of the chart; series past these are drawn grey and start hidden. */
export const MAX_COLORED = 8;
export const MAX_X_TICKS = 8;

export interface TrendPoint {
  run: EvalRunSummary;
  /** position in the run order of the visible points, 0 = oldest */
  x: number;
  value: number;
  /** only a complete run is drawn as a solid point on the line */
  complete: boolean;
}

export interface TrendSeries {
  key: string;
  label: string;
  /** 0..MAX_COLORED-1, or null for a grey series */
  slot: number | null;
  visible: boolean;
  /** runs of this series that have the metric (also for a hidden series) */
  count: number;
  /** the visible points, oldest first (empty for a hidden series) */
  points: TrendPoint[];
}

export interface YScale {
  min: number;
  max: number;
  ticks: number[];
  /** 0 at the bottom of the plot, 1 at the top */
  positionOf: (value: number) => number;
}

export interface Trend {
  metric: MetricDef;
  /** every series that has the metric, in colour order */
  series: TrendSeries[];
  /** the runs on the x axis, oldest first (index = x) */
  runs: EvalRunSummary[];
  /** indexes into `runs` that get a tick label */
  ticks: number[];
  y: YScale;
}

/** The value of `metric` in a run, or null when the run does not have it. */
export function metricValue(
  run: EvalRunSummary,
  metric: string,
): number | null {
  const v = run.metrics[metric];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * A colour slot per series, decided per kind over ALL runs (the most recently run series
 * first), so a line keeps its colour whichever metric is shown.
 */
export function colorSlots(
  runs: readonly EvalRunSummary[],
): Map<string, number | null> {
  const slots = new Map<string, number | null>();
  const used = new Map<string, number>(); // kind -> slots handed out
  for (const series of groupSeries(runs)) {
    const n = used.get(series.kind) ?? 0;
    slots.set(series.key, n < MAX_COLORED ? n : null);
    used.set(series.kind, n + 1);
  }
  return slots;
}

/** A series is shown unless toggled: coloured ones start shown, grey ones hidden. */
export function isVisible(slot: number | null, toggled: boolean): boolean {
  return (slot !== null) !== toggled;
}

/** How many runs have each metric (for the picker: a metric without runs is disabled). */
export function metricCounts(
  runs: readonly EvalRunSummary[],
): Map<string, number> {
  const counts = new Map<string, number>();
  for (const m of METRICS)
    counts.set(
      m.key,
      runs.filter((r) => r.kind === m.group && metricValue(r, m.key) !== null)
        .length,
    );
  return counts;
}

/** The metric shown first: the first one (retrieval, then answers) that some run has. */
export function defaultMetric(runs: readonly EvalRunSummary[]): string {
  const counts = metricCounts(runs);
  return (
    METRICS.find((m) => (counts.get(m.key) ?? 0) > 0)?.key ?? METRICS[0]!.key
  );
}

function niceStep(raw: number): number {
  const exp = Math.floor(Math.log10(raw));
  const f = raw / 10 ** exp;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nice * 10 ** exp;
}

// toPrecision: 3 * 0.1 is 0.30000000000000004, and a tick label should say 0.3
const clean = (v: number) => Number(v.toPrecision(12));

/**
 * The y scale. Shares (recall, faithfulness, ...) always span 0..1, so lines of
 * different metrics or runs are never stretched to look like large changes. Latency and
 * cost get their own scale from 0 to a round number above the largest value.
 */
export function yScale(metric: MetricDef, values: readonly number[]): YScale {
  const finite = values.filter(Number.isFinite);
  let lo = Math.min(0, ...finite);
  let hi = Math.max(0, ...finite);
  let ticks: number[];
  if (metric.unit === "share" && lo >= 0 && hi <= 1) {
    lo = 0;
    hi = 1;
    ticks = [0, 0.25, 0.5, 0.75, 1];
  } else {
    if (metric.unit === "share") hi = Math.max(1, hi); // a share outside 0..1: widen, never cut
    if (hi <= lo) hi = lo + 1; // all zero: any scale, but not a zero-height one
    const step = niceStep((hi - lo) / 4);
    lo = clean(Math.floor(lo / step) * step);
    hi = clean(Math.ceil(hi / step) * step);
    ticks = [];
    for (let i = 0; lo + i * step <= hi + step / 1e6; i++)
      ticks.push(clean(lo + i * step));
  }
  const min = lo;
  const max = hi;
  return { min, max, ticks, positionOf: (v) => (v - min) / (max - min) };
}

/** At most `max` evenly spread indexes of `count` runs, always the first and the last. */
export function xTicks(count: number, max = MAX_X_TICKS): number[] {
  if (count <= 0) return [];
  if (count <= max) return Array.from({ length: count }, (_, i) => i);
  const out = new Set<number>();
  for (let i = 0; i < max; i++)
    out.add(Math.round((i * (count - 1)) / (max - 1)));
  return [...out];
}

/**
 * The chart for one metric. Only runs of the metric's kind (retrieval or answers) with a
 * value for it are drawn; selection runs never are. `toggled` holds the series keys the
 * reader switched from their default (see isVisible).
 */
export function buildTrend(
  runs: readonly EvalRunSummary[],
  metricKey: string,
  toggled: ReadonlySet<string> = new Set(),
): Trend | null {
  const metric = metricDef(metricKey);
  if (!metric) return null;
  const slots = colorSlots(runs);
  const groups = groupSeries(runs).filter((s) => s.kind === metric.group);

  const series: TrendSeries[] = [];
  const shown: EvalRunSummary[] = [];
  for (const s of groups) {
    const withValue = s.runs.filter((r) => metricValue(r, metric.key) !== null);
    if (withValue.length === 0) continue;
    const slot = slots.get(s.key) ?? null;
    const visible = isVisible(slot, toggled.has(s.key));
    series.push({
      key: s.key,
      label: s.label,
      slot,
      visible,
      count: withValue.length,
      points: [],
    });
    if (visible) shown.push(...withValue);
  }
  // Colour order: coloured slots first, then grey ones (each in their newest-first order).
  series.sort((a, b) => (a.slot ?? MAX_COLORED) - (b.slot ?? MAX_COLORED));

  const order = chronological(shown);
  const xOf = new Map(order.map((r, i) => [r.id, i]));
  for (const s of series) {
    if (!s.visible) continue;
    const group = groups.find((g) => g.key === s.key)!;
    s.points = group.runs
      .filter((r) => metricValue(r, metric.key) !== null)
      .map((run) => ({
        run,
        x: xOf.get(run.id)!,
        value: metricValue(run, metric.key)!,
        complete: runState(run.status) === "complete",
      }));
  }
  return {
    metric,
    series,
    runs: order,
    ticks: xTicks(order.length),
    y: yScale(
      metric,
      series.flatMap((s) => s.points.map((p) => p.value)),
    ),
  };
}

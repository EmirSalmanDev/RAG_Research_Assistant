import { strict as assert } from "node:assert";
import { test } from "node:test";

import { metricDef } from "./metrics.ts";
import type { EvalRunSummary } from "./runs.ts";
import {
  MAX_COLORED,
  buildTrend,
  colorSlots,
  defaultMetric,
  isVisible,
  metricCounts,
  xTicks,
  yScale,
} from "./trend.ts";

function run(id: number, extra: Partial<EvalRunSummary> = {}): EvalRunSummary {
  return {
    id,
    name: null,
    kind: "retrieval",
    suite: null,
    suite_id: null,
    step: null,
    variant: null,
    label: null,
    created_at: `2026-10-${String(id).padStart(2, "0")}T12:00:00+00:00`,
    git_sha: null,
    status: "complete",
    n_questions: 20,
    splits: [],
    metrics: {},
    ...extra,
  };
}

const RUNS = [
  // newest first, as GET /evals/runs returns them
  run(9, {
    kind: "selection",
    name: "selection:retrieval",
    metrics: { winner: 1 },
  }),
  run(8, {
    kind: "answers",
    name: "answers:default",
    metrics: { faithfulness: 0.8 },
  }),
  run(7, { name: "hybrid", metrics: { "recall@8": 0.95, mrr: 0.8 } }),
  run(6, {
    name: "dense",
    status: "stopped: budget",
    metrics: { "recall@8": 0.7 },
  }),
  run(5, { name: "dense", metrics: { "recall@8": 0.85 } }),
  run(4, { name: "hybrid", metrics: { mrr: 0.7 } }), // no recall@8: skipped for it
  run(3, { name: "hybrid", metrics: { "recall@8": 0.9 } }),
];

test("one line per series, points in run order, runs without the metric skipped", () => {
  const trend = buildTrend(RUNS, "recall@8")!;
  assert.deepEqual(
    trend.series.map((s) => [
      s.label,
      s.points.map((p) => [p.run.id, p.x, p.value]),
    ]),
    [
      [
        "hybrid",
        [
          [3, 0, 0.9],
          [7, 3, 0.95],
        ],
      ],
      [
        "dense",
        [
          [5, 1, 0.85],
          [6, 2, 0.7],
        ],
      ],
    ],
  );
  assert.deepEqual(
    trend.runs.map((r) => r.id),
    [3, 5, 6, 7],
  );
  // the stopped run is a point, but never a complete one
  const stopped = trend.series[1]!.points.find((p) => p.run.id === 6)!;
  assert.equal(stopped.complete, false);
  assert.equal(trend.series[0]!.points[0]!.complete, true);
});

test("an answer metric draws only answer runs; selections are never drawn", () => {
  const trend = buildTrend(RUNS, "faithfulness")!;
  assert.deepEqual(
    trend.series.map((s) => s.label),
    ["default"],
  );
  assert.equal(buildTrend(RUNS, "winner"), null); // not a chart metric
  assert.equal(metricCounts(RUNS).get("recall@8"), 4);
  assert.equal(metricCounts(RUNS).get("faithfulness"), 1);
  assert.equal(metricCounts(RUNS).get("context_recall"), 0);
  assert.equal(defaultMetric(RUNS), "recall@8");
  assert.equal(defaultMetric([]), "recall@8");
});

test("colours follow the series, not the metric: switching keeps every line's slot", () => {
  const recall = buildTrend(RUNS, "recall@8")!;
  const mrr = buildTrend(RUNS, "mrr")!;
  const slot = (t: typeof recall, label: string) =>
    t.series.find((s) => s.label === label)?.slot;
  assert.equal(slot(recall, "hybrid"), slot(mrr, "hybrid"));
  // dense has no mrr: the hybrid line does not take over its colour
  assert.equal(mrr.series.length, 1);
  assert.notEqual(slot(recall, "dense"), slot(recall, "hybrid"));
});

test("past eight series the rest are grey and start hidden; a toggle shows them", () => {
  const many = Array.from({ length: 10 }, (_, i) =>
    run(i + 1, { name: `mode-${i + 1}`, metrics: { "recall@8": i / 10 } }),
  );
  const slots = colorSlots(many);
  assert.equal(
    [...slots.values()].filter((s) => s !== null).length,
    MAX_COLORED,
  );
  const trend = buildTrend(many, "recall@8")!;
  assert.equal(trend.series.filter((s) => s.visible).length, MAX_COLORED);
  const grey = trend.series.filter((s) => s.slot === null);
  assert.equal(grey.length, 2);
  assert.ok(
    grey.every((s) => !s.visible && s.points.length === 0 && s.count === 1),
  );
  const shown = buildTrend(many, "recall@8", new Set([grey[0]!.key]))!;
  assert.equal(shown.series.filter((s) => s.visible).length, MAX_COLORED + 1);
  assert.equal(isVisible(0, false), true);
  assert.equal(isVisible(0, true), false);
  assert.equal(isVisible(null, false), false);
  assert.equal(isVisible(null, true), true);
});

test("shares always span 0..1; latency and cost get their own round scale from 0", () => {
  const share = yScale(metricDef("recall@8")!, [0.62, 0.71]);
  assert.deepEqual([share.min, share.max], [0, 1]);
  assert.deepEqual(share.ticks, [0, 0.25, 0.5, 0.75, 1]);
  assert.equal(share.positionOf(0.5), 0.5);

  const latency = yScale(metricDef("latency_ms_p50")!, [2100, 3400]);
  assert.equal(latency.min, 0);
  assert.equal(latency.max, 4000);
  assert.deepEqual(latency.ticks, [0, 1000, 2000, 3000, 4000]);

  const cost = yScale(metricDef("cost_per_answer_usd")!, [0.0031, 0.0042]);
  assert.equal(cost.min, 0);
  assert.ok(cost.max >= 0.0042 && cost.max <= 0.006);
  assert.ok(cost.ticks.every((t) => Number(t.toPrecision(12)) === t)); // no float noise

  // nothing to draw, or all zero: still a scale with height
  const empty = yScale(metricDef("latency_ms_p50")!, []);
  assert.ok(empty.max > empty.min);
  // a share outside 0..1 widens the scale instead of being cut off
  const odd = yScale(metricDef("response_relevancy")!, [-0.1, 0.9]);
  assert.ok(odd.min <= -0.1 && odd.max >= 1);
});

test("x ticks: every run when few, at most eight with the first and the last when many", () => {
  assert.deepEqual(xTicks(0), []);
  assert.deepEqual(xTicks(3), [0, 1, 2]);
  const ticks = xTicks(40);
  assert.ok(ticks.length <= 8);
  assert.equal(ticks[0], 0);
  assert.equal(ticks[ticks.length - 1], 39);
});

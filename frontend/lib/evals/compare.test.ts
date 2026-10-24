import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
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
} from "./compare.ts";
import type { EvalResultRow } from "./compare.ts";
import { panelTargets, pickTarget } from "./panel.ts";

function row(
  itemId: number,
  questionId: string | null,
  split: string | null,
  scores: Record<string, number | null>,
  extra: Partial<EvalResultRow> = {},
): EvalResultRow {
  return {
    item_id: itemId,
    question_id: questionId,
    question: `Invented question ${questionId ?? itemId}?`,
    split,
    type: "fact",
    outcome: "answered",
    refusal: null,
    scores,
    ...extra,
  };
}

const A = [
  row(1, "q01", "heldout", { faithfulness: 0.5, latency_ms: 2000 }),
  row(2, "gen-001", "generated", { faithfulness: 1, latency_ms: 1500 }),
  row(3, "gen-002", "generated", { faithfulness: 0.8 }),
];
const B = [
  row(2, "gen-001", "generated", { faithfulness: 0.6, latency_ms: 1000 }),
  row(1, "q01", "heldout", { faithfulness: 0.75, latency_ms: 2500 }),
  row(
    4,
    "gen-003",
    "generated",
    { faithfulness: 1 },
    { outcome: "refused", refusal: "insufficient" },
  ),
];

test("join on question id: A's order, then questions only B has", () => {
  const rows = joinResults(A, B);
  assert.deepEqual(
    rows.map((r) => [r.questionId, r.a?.item_id ?? null, r.b?.item_id ?? null]),
    [
      ["q01", 1, 1],
      ["gen-001", 2, 2],
      ["gen-002", 3, null], // only in A
      ["gen-003", null, 4], // only in B
    ],
  );
  assert.deepEqual(joinCounts(rows), { both: 2, onlyA: 1, onlyB: 1 });
  // one run alone: every row has one side
  assert.deepEqual(joinCounts(joinResults(A, null)), {
    both: 0,
    onlyA: 3,
    onlyB: 0,
  });
});

test("a result without a question id is joined on its item id", () => {
  const rows = joinResults(
    [row(7, null, null, { mrr: 1 })],
    [row(7, null, null, { mrr: 0.5 })],
  );
  assert.equal(rows.length, 1);
  assert.equal(rows[0]!.key, "item:7");
  assert.equal(scorePair(rows[0]!, "mrr").diff, -0.5);
});

test("differences per question, null when one side is missing", () => {
  const rows = joinResults(A, B);
  const q01 = scorePair(rows[0]!, "faithfulness");
  assert.deepEqual([q01.a, q01.b], [0.5, 0.75]);
  assert.equal(q01.diff, 0.25);
  assert.equal(rowDirection(rows[0]!, "faithfulness"), "better");
  assert.equal(rowDirection(rows[0]!, "latency_ms"), null); // per-question latency: no known direction
  assert.equal(scorePair(rows[2]!, "faithfulness").diff, null); // only in A
  assert.equal(scorePair(rows[2]!, "faithfulness").a, 0.8);
});

test("the split filter keeps one split; 'all' keeps everything", () => {
  const rows = joinResults(A, B);
  assert.deepEqual(
    filterSplit(rows, "heldout").map((r) => r.questionId),
    ["q01"],
  );
  assert.deepEqual(
    filterSplit(rows, "generated").map((r) => r.questionId),
    ["gen-001", "gen-002", "gen-003"],
  );
  assert.equal(filterSplit(rows, "all").length, 4);
});

test("score keys: known metrics first; the default follows the chart's metric", () => {
  const rows = joinResults(A, B);
  assert.deepEqual(scoreKeys(rows), ["faithfulness", "latency_ms"]);
  assert.equal(defaultScore(scoreKeys(rows), "faithfulness"), "faithfulness");
  assert.equal(defaultScore(scoreKeys(rows), "recall@8"), "faithfulness");
  assert.equal(defaultScore(["latency_ms"], "recall@8"), "latency_ms");
  assert.equal(defaultScore([], "recall@8"), null);
});

test("sorting by change: worst first, best first, rows without a pair last", () => {
  const rows = joinResults(A, B);
  // faithfulness: q01 +0.25, gen-001 -0.4, the other two have no pair
  assert.deepEqual(
    sortRows(rows, "faithfulness", "worst").map((r) => r.questionId),
    ["gen-001", "q01", "gen-002", "gen-003"],
  );
  assert.deepEqual(
    sortRows(rows, "faithfulness", "best").map((r) => r.questionId),
    ["q01", "gen-001", "gen-002", "gen-003"],
  );
  assert.deepEqual(
    sortRows(rows, "faithfulness", "run").map((r) => r.questionId),
    rows.map((r) => r.questionId),
  );
});

test("headline rows: both runs' metrics side by side with B - A", () => {
  const out = headlineRows(
    { "recall@8": 0.8, n_questions: 20 },
    { "recall@8": 0.9, mrr: 0.7, n_questions: 20 },
  );
  assert.deepEqual(
    out.map((r) => [
      r.key,
      r.a,
      r.b,
      r.diff === null ? null : Number(r.diff.toFixed(6)),
    ]),
    [
      ["recall@8", 0.8, 0.9, 0.1],
      ["mrr", null, 0.7, null],
      ["n_questions", 20, 20, 0],
    ],
  );
  assert.deepEqual(headlineRows(null, null), []);
});

test("split metrics come from the run's stored splits; missing split is null", () => {
  const all = {
    status: "complete",
    all: { faithfulness: 0.75, n_questions: 2 },
    generated: { faithfulness: 0.7, cost_per_answer_usd: "0.004", flag: true },
  };
  assert.deepEqual(splitMetrics(all, "generated"), {
    faithfulness: 0.7,
    cost_per_answer_usd: 0.004,
  });
  assert.equal(splitMetrics(all, "heldout"), null);
  assert.equal(splitMetrics({ "recall@8": 0.9 }, "heldout"), null); // flat (older) metrics
  assert.equal(splitMetrics(null, "generated"), null);
});

test("the panel opens a question on A, or on B when only B has it", () => {
  const rows = joinResults(A, B);
  const both = panelTargets(rows[0]!, 10, 11);
  assert.deepEqual(both, [
    { side: "a", runId: 10, itemId: 1 },
    { side: "b", runId: 11, itemId: 1 },
  ]);
  assert.equal(pickTarget(both, null)?.side, "a");
  assert.equal(pickTarget(both, "b")?.side, "b");
  const onlyB = panelTargets(rows[3]!, 10, 11);
  assert.equal(pickTarget(onlyB, "a")?.side, "b");
  assert.equal(pickTarget([], "a"), null);
});

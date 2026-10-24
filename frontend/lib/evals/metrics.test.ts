import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  METRICS,
  difference,
  direction,
  formatDiff,
  formatValue,
  orderKeys,
  toNumber,
  unitOf,
} from "./metrics.ts";

test("the chart offers the retrieval and the answer metrics", () => {
  const byGroup = (g: string) =>
    METRICS.filter((m) => m.group === g).map((m) => m.key);
  assert.deepEqual(byGroup("retrieval"), [
    "recall@8",
    "mrr",
    "ndcg@10",
    "coverage@8",
  ]);
  assert.deepEqual(byGroup("answers"), [
    "faithfulness",
    "response_relevancy",
    "context_precision",
    "context_recall",
    "citation_accuracy",
    "uncited_sentence_share",
    "refusal_accuracy",
    "false_refusal_rate",
    "source_coverage",
    "latency_ms_p50",
    "cost_per_answer_usd",
  ]);
});

test("units: known metrics by definition, other keys by their name", () => {
  assert.equal(unitOf("recall@8"), "share");
  assert.equal(unitOf("latency_ms_p50"), "ms");
  assert.equal(unitOf("cost_per_answer_usd"), "usd");
  assert.equal(unitOf("cost_usd"), "usd");
  assert.equal(unitOf("latency_ms"), "ms");
  assert.equal(unitOf("retrieve_ms"), "ms");
  assert.equal(unitOf("n_questions"), "count");
  assert.equal(unitOf("source_tokens"), "count");
  assert.equal(unitOf("something_else"), "number");
});

test("values and differences are formatted by unit; a missing one is a dash", () => {
  assert.equal(formatValue("recall@8", 0.9123), "0.912");
  assert.equal(formatValue("latency_ms_p50", 2100.4), "2,100 ms");
  assert.equal(formatValue("cost_per_answer_usd", 0.0041), "$0.004100");
  assert.equal(formatValue("n_questions", 20), "20");
  assert.equal(formatValue("recall@8", null), "—");
  assert.equal(formatDiff("recall@8", 0.05), "+0.050");
  assert.equal(formatDiff("recall@8", -0.02), "−0.020");
  assert.equal(formatDiff("recall@8", 1e-12), "±0.000");
  assert.equal(formatDiff("latency_ms_p50", -300), "−300 ms");
  assert.equal(formatDiff("recall@8", null), "—");
});

test("direction: up is better for recall, down is better for latency and false refusals", () => {
  assert.equal(direction("recall@8", 0.1), "better");
  assert.equal(direction("recall@8", -0.1), "worse");
  assert.equal(direction("latency_ms_p50", -10), "better");
  assert.equal(direction("false_refusal_rate", 0.1), "worse");
  assert.equal(direction("uncited_sentence_share", -0.1), "better");
  assert.equal(direction("recall@8", 0), "same");
  assert.equal(direction("n_questions", 3), null); // no known direction
  assert.equal(direction("recall@8", null), null);
});

test("difference is B - A, null when either is missing", () => {
  assert.ok(Math.abs(difference(0.5, 0.75)! - 0.25) < 1e-12);
  assert.equal(difference(null, 0.75), null);
  assert.equal(difference(0.5, undefined), null);
});

test("orderKeys: known metrics in their order, the rest alphabetically, no duplicates", () => {
  assert.deepEqual(
    orderKeys(["n_questions", "mrr", "errors", "recall@8", "mrr"]),
    ["recall@8", "mrr", "errors", "n_questions"],
  );
});

test("toNumber reads numbers and numeric strings like the API, nothing else", () => {
  assert.equal(toNumber(0.5), 0.5);
  assert.equal(toNumber("0.004"), 0.004);
  assert.equal(toNumber(""), null);
  assert.equal(toNumber("abc"), null);
  assert.equal(toNumber(true), null);
  assert.equal(toNumber(null), null);
  assert.equal(toNumber({}), null);
  assert.equal(toNumber(Number.NaN), null);
});

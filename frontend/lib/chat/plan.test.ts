import assert from "node:assert/strict";
import { test } from "node:test";

import type { PlanMeta } from "./events.ts";
import { planChips } from "./plan.ts";

const plan = (over: Partial<PlanMeta>): PlanMeta => ({
  intent: "qa",
  rephrasings: [],
  arxiv_ids: [],
  categories: [],
  published_from: null,
  published_to: null,
  dropped: [],
  ...over,
});

test("a plain question with no filters has no chips", () => {
  assert.deepEqual(planChips(plan({})), []);
});

test("the mode, papers and categories become chips; qa is not a mode worth showing", () => {
  assert.deepEqual(
    planChips(
      plan({
        intent: "find_papers",
        arxiv_ids: ["2601.00001"],
        categories: ["cs.CL"],
      }),
    ),
    ["Mode: find papers", "Paper 2601.00001", "cs.CL"],
  );
});

test("the planner's full timestamps are shown as days (shapes seen from the real route)", () => {
  assert.deepEqual(
    planChips(
      plan({
        published_from: "2026-01-01T00:00:00+00:00",
        published_to: "2026-10-12T23:59:59.999999+00:00",
      }),
    ),
    ["From Jan 1, 2026", "Until Oct 12, 2026"],
  );
});

test("a plain date still works, and an unreadable one is shown as it came", () => {
  assert.deepEqual(
    planChips(plan({ published_from: "2026-09-01", published_to: "soon" })),
    ["From Sep 1, 2026", "Until soon"],
  );
});

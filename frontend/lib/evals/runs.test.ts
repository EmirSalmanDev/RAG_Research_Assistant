import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  chronological,
  groupSeries,
  runState,
  runTitle,
  seriesKey,
  seriesLabel,
  stopReason,
} from "./runs.ts";
import type { EvalRunSummary } from "./runs.ts";

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
    git_sha: "abc1234",
    status: "complete",
    n_questions: 20,
    splits: [],
    metrics: {},
    ...extra,
  };
}

test("status: only 'complete' is complete; stopped and running are told apart", () => {
  assert.equal(runState("complete"), "complete");
  assert.equal(
    runState("stopped: budget: the cap of $1.00 is reached"),
    "stopped",
  );
  assert.equal(runState("stopped"), "stopped");
  assert.equal(runState("running"), "running");
  assert.equal(runState(" Running "), "running");
  assert.equal(runState(null), "unknown");
  assert.equal(runState(undefined), "unknown");
  assert.equal(runState("completed?"), "unknown"); // never guessed to be complete
  assert.equal(
    stopReason("stopped: budget: cap reached"),
    "budget: cap reached",
  );
  assert.equal(stopReason("stopped"), null);
  assert.equal(stopReason("complete"), null);
});

test("series keys: variant runs by kind, suite and variant; older runs by name", () => {
  const ablation = run(1, {
    kind: "answers",
    name: "ablation:a3-hybrid:answers",
    suite: "ablation",
    variant: "a3-hybrid",
    label: "hybrid (RRF)",
  });
  const smoke = run(2, {
    kind: "answers",
    name: "smoke:a3-hybrid:answers",
    suite: "smoke",
    variant: "a3-hybrid",
  });
  const retrievalPart = run(3, {
    name: "ablation:a3-hybrid:retrieval",
    suite: "ablation",
    variant: "a3-hybrid",
  });
  assert.equal(seriesKey(ablation), "answers|ablation|a3-hybrid");
  // the same variant in a smoke suite (fewer questions) is a line of its own
  assert.notEqual(seriesKey(smoke), seriesKey(ablation));
  // and its retrieval part is another kind
  assert.equal(seriesKey(retrievalPart), "retrieval|ablation|a3-hybrid");
  assert.equal(seriesKey(run(4, { name: "hybrid" })), "retrieval|name|hybrid");
  assert.equal(
    seriesKey(run(5, { kind: "answers", name: "answers:default" })),
    "answers|name|answers:default",
  );
  assert.equal(
    seriesKey(run(6, { kind: "selection", name: "selection:retrieval" })),
    null,
  );

  assert.equal(seriesLabel(ablation), "hybrid (RRF) · ablation");
  assert.equal(seriesLabel(smoke), "a3-hybrid · smoke");
  assert.equal(
    seriesLabel(run(5, { kind: "answers", name: "answers:default" })),
    "default",
  );
  assert.equal(runTitle(ablation), "#1 answers · hybrid (RRF) · ablation");
  assert.equal(
    runTitle(
      run(6, {
        kind: "selection",
        name: "selection:retrieval",
        step: "retrieval",
      }),
    ),
    "#6 selection · retrieval",
  );
});

test("chronological: oldest first, same instant by id", () => {
  const same = "2026-10-05T12:00:00+00:00";
  const ids = chronological([
    run(9),
    run(3),
    run(7, { created_at: same }),
    run(5, { created_at: same }),
  ]).map((r) => r.id);
  assert.deepEqual(ids, [3, 5, 7, 9]);
});

test("groupSeries: runs grouped oldest first, series by their newest run, selections left out", () => {
  const runs = [
    run(10, { name: "hybrid" }),
    run(9, { kind: "selection", name: "selection:retrieval" }),
    run(8, { name: "dense" }),
    run(2, { name: "hybrid" }),
    run(1, { name: "bm25" }),
  ];
  const series = groupSeries(runs);
  assert.deepEqual(
    series.map((s) => [s.label, s.runs.map((r) => r.id)]),
    [
      ["hybrid", [2, 10]],
      ["dense", [8]],
      ["bm25", [1]],
    ],
  );
});

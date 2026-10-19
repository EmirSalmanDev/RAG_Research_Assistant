import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  ALL_DIMENSIONS,
  MAX_PAPERS,
  MIN_PAPERS,
  compareProblem,
  compareRequest,
  dimensionLabel,
  removePaper,
  toggleDimension,
  togglePaper,
} from "./selection.ts";
import type { PickedPaper } from "./selection.ts";

const paper = (n: number): PickedPaper => ({
  arxiv_id: `2601.0000${n}`,
  title: `Invented paper ${n}`,
  has_fulltext: n % 2 === 0,
});

test("picking adds, picking again removes", () => {
  const one = togglePaper([], paper(1));
  assert.deepEqual(one, { selected: [paper(1)], refused: null });
  const two = togglePaper(one.selected, paper(2));
  assert.deepEqual(
    two.selected.map((p) => p.arxiv_id),
    ["2601.00001", "2601.00002"],
  );
  assert.deepEqual(togglePaper(two.selected, paper(1)).selected, [paper(2)]);
});

test("never more than five papers: the sixth click is refused and changes nothing", () => {
  let selected: PickedPaper[] = [];
  for (let n = 1; n <= MAX_PAPERS; n++) {
    selected = togglePaper(selected, paper(n)).selected;
  }
  assert.equal(selected.length, MAX_PAPERS);
  const sixth = togglePaper(selected, paper(6));
  assert.equal(sixth.refused, "full");
  assert.deepEqual(sixth.selected, selected);
  // a picked paper can still be removed when the list is full
  assert.equal(togglePaper(selected, paper(3)).selected.length, MAX_PAPERS - 1);
});

test("Compare is allowed for 2 to 5 papers and at least one dimension", () => {
  const dims = ALL_DIMENSIONS;
  assert.match(
    compareProblem([], dims) ?? "",
    /at least 2 papers \(0 picked\)/,
  );
  assert.match(compareProblem([paper(1)], dims) ?? "", /\(1 picked\)/);
  assert.equal(compareProblem([paper(1), paper(2)], dims), null);
  const five = [1, 2, 3, 4, 5].map(paper);
  assert.equal(compareProblem(five, dims), null);
  assert.match(compareProblem([...five, paper(6)], dims) ?? "", /at most 5/);
  assert.match(compareProblem([paper(1), paper(2)], []) ?? "", /one dimension/);
  assert.equal(MIN_PAPERS, 2);
});

test("dimensions toggle and keep the table's order", () => {
  assert.deepEqual(ALL_DIMENSIONS, [
    "task",
    "method",
    "data",
    "metrics",
    "compute",
    "limitations",
  ]);
  assert.deepEqual(toggleDimension(["task", "data"], "method"), [
    "task",
    "method",
    "data",
  ]);
  assert.deepEqual(toggleDimension(["task", "data"], "task"), ["data"]);
  assert.equal(dimensionLabel("limitations"), "Limitations");
  assert.equal(dimensionLabel("unknown"), "unknown");
});

test("the request carries the ids in the order picked", () => {
  assert.deepEqual(compareRequest([paper(2), paper(1)], ["method", "data"]), {
    arxiv_ids: ["2601.00002", "2601.00001"],
    dimensions: ["method", "data"],
  });
  assert.deepEqual(removePaper([paper(1), paper(2)], "2601.00001"), [paper(2)]);
});

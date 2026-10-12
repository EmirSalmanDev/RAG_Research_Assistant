import { strict as assert } from "node:assert";
import { test } from "node:test";

import { groupByPaper } from "./group.ts";

const hit = (arxiv_id: string, section: string) => ({
  arxiv_id,
  title: `Paper ${arxiv_id}`,
  published_at: "2026-09-01T00:00:00+00:00",
  section,
});

test("groups passages by paper, ordered by each paper's best passage", () => {
  const hits = [
    hit("B", "b1"),
    hit("A", "a1"),
    hit("B", "b2"),
    hit("C", "c1"),
    hit("A", "a2"),
  ];
  const groups = groupByPaper(hits);
  assert.deepEqual(
    groups.map((g) => g.arxivId),
    ["B", "A", "C"],
  );
  assert.deepEqual(
    groups.map((g) => g.hits.map((h) => h.section)),
    [["b1", "b2"], ["a1", "a2"], ["c1"]],
  );
  assert.equal(groups[0]?.title, "Paper B");
});

test("no hits, no groups; the input is not modified", () => {
  assert.deepEqual(groupByPaper([]), []);
  const hits = [hit("A", "a1")];
  groupByPaper(hits);
  assert.equal(hits.length, 1);
});

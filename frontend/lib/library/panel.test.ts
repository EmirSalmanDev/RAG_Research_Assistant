import { strict as assert } from "node:assert";
import { test } from "node:test";

import { REVIEW } from "./fixtures.ts";
import { referenceSources, referenceToSource } from "./panel.ts";

test("a reference becomes a source of the existing side panel", () => {
  const source = referenceToSource(REVIEW.references[0]!);
  assert.deepEqual(source, {
    n: 1,
    chunk_id: "00000000-0000-4000-8000-0000000000a1",
    arxiv_id: "2601.00001",
    title: "Invented | Paper One",
    section_path: "Analysis > Summary",
    chunk_type: "analysis",
    published_at: "2020-03-04T00:00:00+00:00",
    abs_url: "https://arxiv.org/abs/2601.00001",
  });
});

test("an uploaded document has no link; the sources come in number order", () => {
  const sources = referenceSources([...REVIEW.references].reverse());
  assert.deepEqual(
    sources.map((s) => s.n),
    [1, 2, 3],
  );
  assert.equal(sources[2]!.abs_url, null);
  assert.equal(sources[1]!.chunk_type, "abstract");
});

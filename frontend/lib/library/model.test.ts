import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  ITEM_EVIDENCE_LABEL,
  REFERENCE_EVIDENCE_LABEL,
  STATUSES,
  paperCount,
  sortCollections,
  statusLabel,
} from "./model.ts";

test("the four reading states, in order, with labels", () => {
  assert.deepEqual(
    STATUSES.map((s) => s.id),
    ["to_read", "reading", "read", "archived"],
  );
  assert.equal(statusLabel("to_read"), "To read");
  assert.equal(statusLabel("something-else"), "something-else");
});

test("evidence labels use the words of chat and compare", () => {
  assert.equal(ITEM_EVIDENCE_LABEL.full_text, "full text");
  assert.equal(ITEM_EVIDENCE_LABEL.abstract_only, "abstract only");
  assert.equal(ITEM_EVIDENCE_LABEL.uploaded, "uploaded");
  assert.equal(REFERENCE_EVIDENCE_LABEL.analysis, "generated analysis");
  assert.equal(REFERENCE_EVIDENCE_LABEL.abstract, "abstract only");
});

test("collections sort by name without regard to case; counts are worded", () => {
  const c = (name: string) => ({
    name,
    description: null,
    paper_count: 0,
    created_at: null,
  });
  assert.deepEqual(
    sortCollections([c("thesis"), c("Saved"), c("Alpha")]).map((x) => x.name),
    ["Alpha", "Saved", "thesis"],
  );
  assert.equal(paperCount(1), "1 paper");
  assert.equal(paperCount(0), "0 papers");
  assert.equal(paperCount(2), "2 papers");
});

import { strict as assert } from "node:assert";
import { test } from "node:test";

import { RELATED_TABS, tabAfterKey } from "./tabs.ts";

test("the related tabs are similar, cites and cited by, in that order", () => {
  assert.deepEqual([...RELATED_TABS], ["similar", "cites", "cited_by"]);
});

test("arrow keys wrap around, Home and End jump, other keys do nothing", () => {
  assert.equal(tabAfterKey("similar", "ArrowRight"), "cites");
  assert.equal(tabAfterKey("cited_by", "ArrowRight"), "similar");
  assert.equal(tabAfterKey("similar", "ArrowLeft"), "cited_by");
  assert.equal(tabAfterKey("cites", "Home"), "similar");
  assert.equal(tabAfterKey("similar", "End"), "cited_by");
  assert.equal(tabAfterKey("cites", "Enter"), null);
});

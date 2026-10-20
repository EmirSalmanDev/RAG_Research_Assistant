import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  FOCUS_MAX_CHARS,
  MAX_REVIEW_PAPERS,
  filenameFromDisposition,
  nameProblem,
  referenceProblem,
  reviewProblem,
  reviewRequest,
} from "./forms.ts";

test("a collection name must be 1 to 100 characters after trimming", () => {
  assert.equal(nameProblem("Thesis"), null);
  assert.equal(nameProblem("  spaced   out  "), null);
  assert.match(nameProblem("   ") ?? "", /name/);
  assert.match(nameProblem("x".repeat(101)) ?? "", /100/);
  assert.equal(nameProblem("x".repeat(100)), null);
});

test("the add-a-paper box takes a short non-empty text", () => {
  assert.equal(referenceProblem("1706.03762"), null);
  assert.equal(referenceProblem("https://arxiv.org/abs/1706.03762v7"), null);
  assert.match(referenceProblem("  ") ?? "", /arXiv id/);
  assert.match(referenceProblem("x".repeat(301)) ?? "", /too long/);
});

test("Generate review is off until a collection with two papers is picked", () => {
  assert.match(reviewProblem(null, 0, "") ?? "", /Pick a collection/);
  assert.match(
    reviewProblem("Thesis", 1, "") ?? "",
    /at least 2 papers \(Thesis has 1\)/,
  );
  assert.match(reviewProblem("Thesis", 0, "") ?? "", /has 0/);
  assert.equal(reviewProblem("Thesis", 2, ""), null);
  assert.equal(reviewProblem("Thesis", 40, "adapters"), null);
  assert.match(
    reviewProblem("Thesis", 5, "x".repeat(FOCUS_MAX_CHARS + 1)) ?? "",
    /200/,
  );
  assert.equal(reviewProblem("Thesis", 5, "x".repeat(FOCUS_MAX_CHARS)), null);
});

test("the review request cleans the focus and bounds max_papers", () => {
  assert.deepEqual(reviewRequest("T", "  small \n adapters  "), {
    collection: "T",
    focus: "small adapters",
    max_papers: 30,
  });
  assert.equal(reviewRequest("T", "   ").focus, null);
  assert.equal(reviewRequest("T", "", 500).max_papers, MAX_REVIEW_PAPERS);
  assert.equal(reviewRequest("T", "", 0).max_papers, 1);
  assert.equal(reviewRequest("T", "", 7.9).max_papers, 7);
});

test("the file name comes from Content-Disposition, else the fallback", () => {
  assert.equal(
    filenameFromDisposition('attachment; filename="my-thesis.bib"', "x.bib"),
    "my-thesis.bib",
  );
  assert.equal(filenameFromDisposition(null, "x.bib"), "x.bib");
  assert.equal(filenameFromDisposition("attachment", "x.bib"), "x.bib");
  assert.equal(
    filenameFromDisposition('attachment; filename="../evil.bib"', "x.bib"),
    "x.bib",
    "a path in the name is never used",
  );
});

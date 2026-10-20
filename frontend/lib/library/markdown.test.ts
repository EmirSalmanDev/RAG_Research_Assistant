import { strict as assert } from "node:assert";
import { test } from "node:test";

import { REVIEW } from "./fixtures.ts";
import {
  EVIDENCE_NOTE,
  citedNumbers,
  reviewFileName,
  reviewToMarkdown,
} from "./markdown.ts";
import type { ReviewOut } from "./model.ts";

test("the draft keeps its [n] markers and lists only the cited references", () => {
  const md = reviewToMarkdown(REVIEW);
  assert.match(md, /^# Related work: Thesis ch\. 2: adapters\n/);
  assert.ok(md.includes("Focus: small adapters"));
  assert.ok(md.includes(`_${EVIDENCE_NOTE}_`));
  assert.ok(md.includes("## Adapters"));
  assert.ok(
    md.includes(
      "Several lines of work follow. Example (2020) trains small adapters.[1] Sample (2021) compares two methods.[1][3]",
    ),
  );
  const refs = md.slice(md.indexOf("## References"));
  assert.ok(
    refs.includes(
      "[1] `example2020invented`: Invented | Paper One (2020), generated analysis. https://arxiv.org/abs/2601.00001",
    ),
  );
  assert.ok(
    refs.includes(
      "[3] `anon2022uploaded`: An Uploaded Paper (2022), abstract only. uploaded document, no arXiv page",
    ),
  );
  assert.ok(
    !refs.includes("sample2021unused"),
    "an uncited paper is not a reference",
  );
  assert.ok(!md.includes(".pdf"), "the export never links a PDF");
  assert.ok(md.includes("1 sentence removed for lack of a valid citation"));
  assert.ok(md.includes("review_write_v1"));
});

test("the same review gives the same text", () => {
  assert.equal(
    reviewToMarkdown(REVIEW),
    reviewToMarkdown(structuredClone(REVIEW)),
  );
});

test("cited numbers are collected once, ascending", () => {
  assert.deepEqual(citedNumbers(REVIEW), [1, 3]);
});

test("a draft with no sentence says so and has no references", () => {
  const empty: ReviewOut = {
    ...REVIEW,
    focus: null,
    sections: [],
    counts: { ...REVIEW.counts, dropped_uncited: 3 },
  };
  const md = reviewToMarkdown(empty);
  assert.ok(md.includes("No sentence survived the checks."));
  assert.ok(md.includes("No reference is cited."));
  assert.ok(!md.includes("Focus:"));
  assert.ok(md.includes("3 sentences removed"));
});

test("a link that is not an arXiv abstract page is never exported", () => {
  const odd: ReviewOut = {
    ...REVIEW,
    references: REVIEW.references.map((r) =>
      r.n === 1 ? { ...r, abs_url: "https://evil.example/abs/1" } : r,
    ),
  };
  const md = reviewToMarkdown(odd);
  assert.ok(!md.includes("evil.example"));
});

test("the file name is safe for any file system", () => {
  assert.equal(reviewFileName(REVIEW), "related-work-thesis-ch-2-adapters.md");
  assert.equal(
    reviewFileName({ ...REVIEW, collection: "Tez çalışması" }),
    "related-work-tez-calismasi.md",
  );
  assert.equal(
    reviewFileName({ ...REVIEW, collection: "日本語" }),
    "related-work-library.md",
  );
});

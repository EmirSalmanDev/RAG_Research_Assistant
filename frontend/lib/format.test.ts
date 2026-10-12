import { strict as assert } from "node:assert";
import { test } from "node:test";

import { excerpt, formatDate, fulltextLabel } from "./format.ts";

test("formatDate shows a UTC calendar date and leaves non-dates alone", () => {
  assert.equal(formatDate("2026-09-01T23:30:00+00:00"), "Sep 1, 2026");
  assert.equal(formatDate("not a date"), "not a date");
});

test("excerpt keeps a short text whole, with whitespace collapsed", () => {
  assert.deepEqual(excerpt("  A short\n\n passage. "), {
    text: "A short passage.",
    truncated: false,
  });
});

test("excerpt cuts a long text at a word boundary and says so", () => {
  const text = "word ".repeat(100);
  const result = excerpt(text, 50);
  assert.equal(result.truncated, true);
  assert.ok(result.text.endsWith("…"));
  assert.ok(result.text.length <= 51);
  assert.ok(!result.text.slice(0, -1).endsWith(" "));
  assert.match(result.text, /^(word )+word…$/);
});

test("excerpt still cuts a text with no spaces", () => {
  const result = excerpt("x".repeat(500), 40);
  assert.equal(result.text, `${"x".repeat(40)}…`);
  assert.equal(result.truncated, true);
});

test("fulltextLabel explains every status and passes unknown ones through", () => {
  assert.equal(fulltextLabel("indexed"), "Full text indexed");
  assert.equal(fulltextLabel("none"), "Abstract only");
  assert.equal(fulltextLabel("something-new"), "something-new");
});

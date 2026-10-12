import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  MAX_QUERY_CHARS,
  parseSearchParams,
  toSearchString,
  validateSearch,
} from "./params.ts";

test("an empty or missing query is idle, not an error", () => {
  assert.deepEqual(parseSearchParams(""), { params: null, problems: [] });
  assert.deepEqual(parseSearchParams("q=%20%20"), {
    params: null,
    problems: [],
  });
  assert.deepEqual(parseSearchParams("category=cs.CL"), {
    params: null,
    problems: [],
  });
});

test("reads the question, repeated categories and the dates", () => {
  const { params, problems } = parseSearchParams(
    "q=low-rank+adapters&category=cs.CL&category=cs.LG&from=2026-09-01&to=2026-09-30&junk=1",
  );
  assert.deepEqual(problems, []);
  assert.deepEqual(params, {
    q: "low-rank adapters",
    categories: ["cs.CL", "cs.LG"],
    from: "2026-09-01",
    to: "2026-09-30",
  });
});

test("the question is trimmed, duplicate categories dropped, empty dates become null", () => {
  const { params } = parseSearchParams(
    "q=%20adapters%20&category=cs.CL&category=cs.CL&from=&to=",
  );
  assert.deepEqual(params, {
    q: "adapters",
    categories: ["cs.CL"],
    from: null,
    to: null,
  });
});

test("toSearchString and parseSearchParams are inverses", () => {
  const original = {
    q: "What is RAG? (a & b)",
    categories: ["cs.IR", "stat.ML"],
    from: "2026-01-02",
    to: null,
  };
  const text = toSearchString(original);
  assert.equal(
    text,
    "q=What+is+RAG%3F+%28a+%26+b%29&category=cs.IR&category=stat.ML&from=2026-01-02",
  );
  assert.deepEqual(parseSearchParams(text).params, original);
  assert.equal(
    toSearchString({ q: "x", categories: [], from: null, to: null }),
    "q=x",
  );
});

test("dates must be real days, in order", () => {
  for (const [from, to, why] of [
    ["2026-13-01", null, /start date is not a valid date/],
    ["2026-02-30", null, /start date is not a valid date/],
    ["09/01/2026", null, /start date is not a valid date/],
    [null, "tomorrow", /end date is not a valid date/],
    ["2026-10-01", "2026-09-01", /start date is after the end date/],
  ] as const) {
    const result = validateSearch({ q: "x", categories: [], from, to });
    assert.equal(result.params, null);
    assert.match(result.problems.join(" "), why);
  }
  assert.notEqual(
    validateSearch({
      q: "x",
      categories: [],
      from: "2026-09-01",
      to: "2026-09-01",
    }).params,
    null,
  );
  assert.notEqual(
    validateSearch({ q: "x", categories: [], from: "2024-02-29", to: null })
      .params,
    null,
  );
});

test("a too-long question, a bad category and too many categories are refused", () => {
  assert.match(
    validateSearch({
      q: "x".repeat(MAX_QUERY_CHARS + 1),
      categories: [],
      from: null,
      to: null,
    }).problems.join(),
    /longer than 1000/,
  );
  assert.match(
    validateSearch({
      q: "x",
      categories: ["cs.CL; drop"],
      from: null,
      to: null,
    }).problems.join(),
    /not a valid arXiv category/,
  );
  const many = Array.from(
    { length: 9 },
    (_, i) => `cs.X${String.fromCharCode(65 + i)}`,
  );
  assert.match(
    validateSearch({
      q: "x",
      categories: many,
      from: null,
      to: null,
    }).problems.join(),
    /at most 8/,
  );
  assert.notEqual(
    validateSearch({
      q: "x".repeat(MAX_QUERY_CHARS),
      categories: ["cs.CL"],
      from: null,
      to: null,
    }).params,
    null,
  );
});

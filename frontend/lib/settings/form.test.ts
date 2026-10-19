import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  formatUsd,
  isCategory,
  parseInterest,
  parseSettings,
  splitList,
} from "./form.ts";

const interest = {
  name: "  Agents   and tools ",
  description: "  About agents. ",
  keywords: "agent, tool use\nplanning; Agent",
  categories: "cs.AI, cs.CL",
  isActive: true,
};

test("splitList trims, drops blanks and repeats, and accepts commas, semicolons and lines", () => {
  assert.deepEqual(splitList("a, b\nc;d,, ,A, a "), ["a", "b", "c", "d"]);
  assert.deepEqual(splitList("  "), []);
  assert.deepEqual(splitList("kv   cache"), ["kv cache"]);
});

test("isCategory matches the API's rule", () => {
  for (const good of [
    "cs.CL",
    "stat.ML",
    "hep-th",
    "astro-ph.GA",
    "physics.comp-ph",
    "cs",
  ])
    assert.equal(isCategory(good), true, good);
  for (const bad of [
    "",
    "CS.CL",
    "cs.",
    "cs.CL.X",
    "cs CL",
    "cs/CL",
    "cs.C1",
    "../x",
    "cs." + "a".repeat(40),
  ])
    assert.equal(isCategory(bad), false, bad);
});

test("a good interest becomes a request body", () => {
  const result = parseInterest(interest);
  assert.ok(result.ok);
  assert.deepEqual(result.body, {
    name: "Agents and tools",
    description: "About agents.",
    keywords: ["agent", "tool use", "planning"],
    categories: ["cs.AI", "cs.CL"],
    is_active: true,
  });
});

test("an empty description is sent as null", () => {
  const result = parseInterest({ ...interest, description: "  " });
  assert.ok(result.ok && result.body.description === null);
});

test("interest problems are listed, all at once, in words", () => {
  const result = parseInterest({
    ...interest,
    name: " ",
    categories: "cs.CL, not a category",
    keywords: "x".repeat(61),
  });
  assert.ok(!result.ok);
  assert.equal(result.problems.length, 3);
  assert.match(result.problems[0], /name/);
  assert.ok(result.problems.some((p) => /longer than 60/.test(p)));
  assert.ok(result.problems.some((p) => /not an arXiv category/.test(p)));
  const many = parseInterest({
    ...interest,
    keywords: Array.from({ length: 31 }, (_, i) => `k${i}`).join(","),
  });
  assert.ok(!many.ok && /at most 30 keywords/.test(many.problems[0]));
});

const settings = {
  digestSize: "20",
  fulltextTopN: "20",
  categories: "",
  budgetCap: "",
};

test("settings become a request body; empty categories and budget mean default and none", () => {
  const result = parseSettings(settings);
  assert.ok(result.ok);
  assert.deepEqual(result.body, {
    digest_size: 20,
    fulltext_top_n: 20,
    categories: [],
    budget_cap_usd: null,
  });
  const full = parseSettings({
    ...settings,
    categories: "cs.CV, stat.ML",
    budgetCap: "$12.50",
    fulltextTopN: "0",
  });
  assert.ok(full.ok);
  assert.deepEqual(full.body, {
    digest_size: 20,
    fulltext_top_n: 0,
    categories: ["cs.CV", "stat.ML"],
    budget_cap_usd: 12.5,
  });
});

test("settings problems", () => {
  for (const [field, value] of [
    ["digestSize", "0"],
    ["digestSize", "101"],
    ["digestSize", "ten"],
    ["digestSize", "1.5"],
    ["digestSize", ""],
    ["fulltextTopN", "-1"],
    ["fulltextTopN", "101"],
    ["categories", "cs.CL, nonsense here"],
    ["budgetCap", "-5"],
    ["budgetCap", "lots"],
    ["budgetCap", "1.234"],
    ["budgetCap", "100001"],
  ] as const) {
    const result = parseSettings({ ...settings, [field]: value });
    assert.ok(!result.ok, `${field}=${value}`);
  }
  assert.ok(parseSettings({ ...settings, budgetCap: "0" }).ok); // a cap of zero is allowed
});

test("formatUsd", () => {
  assert.equal(formatUsd(0.045026), "$0.0450");
  assert.equal(formatUsd(0), "$0.0000");
  assert.equal(formatUsd(123.456), "$123.46");
});

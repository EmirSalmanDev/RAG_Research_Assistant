import { strict as assert } from "node:assert";
import { test } from "node:test";

import { cell, judgeSections, judgeView } from "./judge.ts";

test("a list of flat verdicts becomes a table with yes / no cells", () => {
  const view = judgeView([
    { statement: "It does.", supported: true, reason: "said in [1]" },
    { statement: "It also flies.", supported: false },
  ]);
  assert.equal(view.kind, "table");
  if (view.kind !== "table") return;
  assert.deepEqual(view.columns, ["statement", "supported", "reason"]);
  assert.deepEqual(view.rows[0], [
    { text: "It does." },
    { text: "yes", verdict: "yes" },
    { text: "said in [1]" },
  ]);
  assert.deepEqual(view.rows[1]![2], { text: "" }); // a key one row lacks
});

test("an object of a list and a count becomes fields; plain values a list", () => {
  const view = judgeView({
    items: [{ sentence: 0, n: 1, supported: true }],
    invalid_citations: 0,
  });
  assert.equal(view.kind, "fields");
  if (view.kind !== "fields") return;
  assert.deepEqual(
    view.fields.map((f) => [f.key, f.view.kind]),
    [
      ["items", "table"],
      ["invalid_citations", "text"],
    ],
  );
  assert.deepEqual(judgeView(["What is X?", "How does X work?"]), {
    kind: "list",
    items: [{ text: "What is X?" }, { text: "How does X work?" }],
  });
});

test("unknown or deep shapes fall back to JSON; empty ones say so", () => {
  const nested = judgeView({ a: { b: { c: 1 } } });
  assert.equal(nested.kind, "fields");
  if (nested.kind === "fields")
    assert.equal(nested.fields[0]!.view.kind, "json");
  const mixed = judgeView([{ a: [1, 2] }, "x"]);
  assert.equal(mixed.kind, "json");
  if (mixed.kind === "json") assert.match(mixed.text, /"a"/);
  assert.deepEqual(judgeView([]), { kind: "text", text: "(none)" });
  assert.deepEqual(judgeView({}), { kind: "text", text: "(none)" });
  assert.deepEqual(judgeView(null), { kind: "text", text: "—" });
  assert.deepEqual(judgeView(3), { kind: "text", text: "3" });
  assert.deepEqual(cell(0.123456), { text: "0.1235" });
});

test("sections: the answer metrics first, then the rest by name", () => {
  const sections = judgeSections({
    sentences: 3,
    uncited_sentences: 1,
    citation_accuracy: { items: [], invalid_citations: 0 },
    faithfulness: [{ statement: "s", supported: true }],
  });
  assert.deepEqual(
    sections.map((s) => s.key),
    ["faithfulness", "citation_accuracy", "sentences", "uncited_sentences"],
  );
  assert.deepEqual(judgeSections(null), []);
});

import { strict as assert } from "node:assert";
import { test } from "node:test";

import { splitCitations } from "./citations.ts";
import type { Segment } from "./citations.ts";
import { ABSTRACT_SOURCE, SECTION_SOURCE } from "./fixtures.ts";

const SOURCES = [ABSTRACT_SOURCE, SECTION_SOURCE]; // [1] is an abstract, [2] a section

const kinds = (segments: Segment[]) =>
  segments.map((s) => (s.kind === "citation" ? `#${s.n}` : s.text));

test("a [n] with a matching source becomes a citation carrying that source", () => {
  const segments = splitCitations("Adapters are small [1].", SOURCES);
  assert.deepEqual(kinds(segments), ["Adapters are small ", "#1", "."]);
  const chip = segments[1];
  assert.ok(chip?.kind === "citation");
  // the chip can show the abstract badge from the source's chunk_type
  assert.equal(chip.source.chunk_type, "abstract");
  assert.equal(chip.source.abs_url, "https://arxiv.org/abs/2609.00001");
});

test("a [n] with no matching source stays plain text, exactly as written", () => {
  const segments = splitCitations("Unsupported claim [9].", SOURCES);
  assert.deepEqual(segments, [
    { kind: "text", text: "Unsupported claim [9]." },
  ]);
  assert.deepEqual(splitCitations("Odd [0] and [03].", SOURCES), [
    { kind: "text", text: "Odd [0] and [03]." },
  ]);
});

test("a group [1, 2] gives one chip per number; unknown numbers in a group stay text", () => {
  assert.deepEqual(kinds(splitCitations("Both [1, 2] agree.", SOURCES)), [
    "Both ",
    "#1",
    " ",
    "#2",
    " agree.",
  ]);
  assert.deepEqual(kinds(splitCitations("Mixed [2; 7].", SOURCES)), [
    "Mixed ",
    "#2",
    " [7].",
  ]);
  assert.deepEqual(splitCitations("None [8, 9].", SOURCES), [
    { kind: "text", text: "None [8, 9]." },
  ]);
});

test("leading zeros and the same source twice are fine", () => {
  assert.deepEqual(kinds(splitCitations("A [01] B [1].", SOURCES)), [
    "A ",
    "#1",
    " B ",
    "#1",
    ".",
  ]);
});

test("things that look similar but are not citations are left alone", () => {
  const text = "Array a[1] and [x] and [] and [1 2] and (1).";
  assert.deepEqual(kinds(splitCitations(text, [])), [text]);
  // a[1] does match the citation pattern, as it does in the backend: with a source 1 it is a chip
  assert.deepEqual(kinds(splitCitations("a[1]", SOURCES)), ["a", "#1"]);
});

test("while streaming, an unfinished marker at the end is held back", () => {
  const streaming = { streaming: true };
  assert.deepEqual(kinds(splitCitations("Adapters [", SOURCES, streaming)), [
    "Adapters ",
  ]);
  assert.deepEqual(kinds(splitCitations("Adapters [1", SOURCES, streaming)), [
    "Adapters ",
  ]);
  assert.deepEqual(kinds(splitCitations("Adapters [1, ", SOURCES, streaming)), [
    "Adapters ",
  ]);
  assert.deepEqual(kinds(splitCitations("Adapters [1]", SOURCES, streaming)), [
    "Adapters ",
    "#1",
  ]);
  // not streaming (a finished or stopped answer): the text is shown as it is
  assert.deepEqual(kinds(splitCitations("Adapters [1", SOURCES)), [
    "Adapters [1",
  ]);
});

test("no sources yet (before the sources event): everything is text", () => {
  assert.deepEqual(splitCitations("Claim [1].", []), [
    { kind: "text", text: "Claim [1]." },
  ]);
});

test("newlines and spacing are kept, so the answer can be shown pre-wrapped", () => {
  const segments = splitCitations("- one [1]\n- two [2]\n", SOURCES);
  assert.deepEqual(kinds(segments), ["- one ", "#1", "\n- two ", "#2", "\n"]);
});

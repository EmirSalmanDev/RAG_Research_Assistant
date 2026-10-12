import { strict as assert } from "node:assert";
import { test } from "node:test";

import { panelReducer, shouldDismiss } from "./panel.ts";
import type { PanelState } from "./panel.ts";
import { ABSTRACT_SOURCE, SECTION_SOURCE } from "./fixtures.ts";

const ALL = [ABSTRACT_SOURCE, SECTION_SOURCE];
const OTHER_ANSWER = [
  { ...SECTION_SOURCE, chunk_id: "00000000-0000-4000-8000-0000000000ff", n: 1 },
];

test("clicking a chip opens the panel on that source", () => {
  const state = panelReducer(null, {
    type: "open",
    source: ABSTRACT_SOURCE,
    all: ALL,
  });
  assert.deepEqual(state, { source: ABSTRACT_SOURCE, all: ALL });
});

test("clicking another chip while the panel is open switches it (it does not close)", () => {
  const open = panelReducer(null, {
    type: "open",
    source: ABSTRACT_SOURCE,
    all: ALL,
  });
  const switched = panelReducer(open, {
    type: "open",
    source: SECTION_SOURCE,
    all: ALL,
  });
  assert.equal(switched?.source.chunk_id, SECTION_SOURCE.chunk_id);
  // a chip of a different answer replaces the list of other sources as well
  const other = panelReducer(switched, {
    type: "open",
    source: OTHER_ANSWER[0]!,
    all: OTHER_ANSWER,
  });
  assert.deepEqual(other?.all, OTHER_ANSWER);
});

test("picking from the panel's own list changes the source, only within the same answer", () => {
  const open: PanelState = { source: ABSTRACT_SOURCE, all: ALL };
  assert.equal(
    panelReducer(open, { type: "select", source: SECTION_SOURCE })?.source
      .chunk_id,
    SECTION_SOURCE.chunk_id,
  );
  assert.equal(
    panelReducer(open, { type: "select", source: OTHER_ANSWER[0]! }),
    open,
  );
  assert.equal(
    panelReducer(null, { type: "select", source: SECTION_SOURCE }),
    null,
  );
});

test("close empties the panel", () => {
  const open: PanelState = { source: ABSTRACT_SOURCE, all: ALL };
  assert.equal(panelReducer(open, { type: "close" }), null);
  assert.equal(panelReducer(null, { type: "close" }), null);
});

test("Escape and the close button close the panel; interacting outside does not", () => {
  assert.equal(shouldDismiss("escape"), true);
  assert.equal(shouldDismiss("close-button"), true);
  assert.equal(shouldDismiss("outside"), false);
});

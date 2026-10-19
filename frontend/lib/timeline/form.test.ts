import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  DEFAULT_KEY_PAPERS,
  DEFAULT_MAX_PAPERS,
  describeNotes,
  validateTimeline,
} from "./form.ts";
import type { TimelineForm, TimelineOut } from "./form.ts";

const FORM: TimelineForm = {
  topic: "  speculative   decoding ",
  from: "",
  to: "",
  categories: [],
  maxPapers: DEFAULT_MAX_PAPERS,
  keyPapers: DEFAULT_KEY_PAPERS,
};

test("a valid form gives a trimmed topic and empty dates as null", () => {
  const { body, problems } = validateTimeline(FORM);
  assert.deepEqual(problems, []);
  assert.deepEqual(body, {
    topic: "speculative decoding",
    date_from: null,
    date_to: null,
    categories: [],
    max_papers: 30,
    key_papers: 8,
  });
});

test("every broken rule is reported and nothing is sent", () => {
  for (const [form, text] of [
    [{ ...FORM, topic: "ab" }, /at least 3/],
    [{ ...FORM, topic: "x".repeat(201) }, /longer than 200/],
    [{ ...FORM, from: "2026-10-02", to: "2026-10-01" }, /after the end date/],
    [{ ...FORM, maxPapers: 51 }, /1 to 50/],
    [{ ...FORM, maxPapers: 0 }, /1 to 50/],
    [{ ...FORM, keyPapers: 16 }, /1 to 15/],
    [{ ...FORM, keyPapers: 2.5 }, /1 to 15/],
  ] as const) {
    const { body, problems } = validateTimeline(form);
    assert.equal(body, null);
    assert.ok(
      problems.some((p) => text.test(p)),
      String(text),
    );
  }
});

const OUT = (notes: Partial<TimelineOut["notes"]>): TimelineOut => ({
  topic: "t",
  range: { first: null, last: null },
  points: [],
  notes: {
    papers: 12,
    key_papers: 8,
    known_citation_counts: 3,
    narrow_range: true,
    range_days: 9,
    fallback_lines: 0,
    lines_call_failed: false,
    ...notes,
  },
  prompt_version: "timeline_v1",
  tokens_in: 0,
  tokens_out: 0,
  cost_usd: 0,
});

test("the notes say how thin the citations and the time range are", () => {
  const lines = describeNotes(OUT({}));
  assert.equal(lines[0], "12 papers, 8 marked as key.");
  assert.match(lines[1]!, /known for 3 of 12 papers.*count as zero/);
  assert.match(lines[2]!, /only 9 days.*by day/);
  assert.equal(lines.length, 3);
});

test("failed or replaced lines are named; no papers gives no notes", () => {
  assert.match(
    describeNotes(OUT({ lines_call_failed: true })).at(-1)!,
    /call .* failed.*first sentence/,
  );
  assert.match(
    describeNotes(OUT({ fallback_lines: 2 })).at(-1)!,
    /2 lines are the first sentence/,
  );
  assert.deepEqual(describeNotes(OUT({ papers: 0 })), []);
  assert.match(
    describeNotes(
      OUT({ narrow_range: false, range_days: 400, known_citation_counts: 12 }),
    ).join(" "),
    /all 12 papers.*400 days/,
  );
});

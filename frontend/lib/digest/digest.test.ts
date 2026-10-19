import { strict as assert } from "node:assert";
import { test } from "node:test";

import type { DigestEntryOut, DigestOut } from "../api/endpoints.ts";
import {
  authorLine,
  formatDay,
  parseDay,
  statusAfter,
  statusCounts,
  statusLine,
  withEntry,
  withStatus,
} from "./digest.ts";

const entry = (
  id: string,
  status: DigestEntryOut["status"] = "new",
): DigestEntryOut => ({
  arxiv_id: id,
  rank: 1,
  score: 0.5,
  status,
  title: `Invented ${id}`,
  authors: ["Ada Example"],
  author_count: 1,
  categories: ["cs.CL"],
  published_at: "2026-10-18T12:00:00+00:00",
  hf_upvotes: null,
  trending: false,
  best_interest: null,
  summary: "Invented.",
  summary_source: "abstract",
  abs_url: `https://arxiv.org/abs/${id}`,
});

const digest = (entries: DigestEntryOut[]): DigestOut => ({
  requested_date: "2026-10-19",
  digest_date: "2026-10-19",
  is_fallback: false,
  has_interests: true,
  recent_dates: ["2026-10-19"],
  entries,
});

test("each action leads to one status", () => {
  assert.equal(statusAfter("save"), "saved");
  assert.equal(statusAfter("dismiss"), "dismissed");
  assert.equal(statusAfter("undo"), "new");
});

test("withStatus changes one entry, keeps order, and does not change its input", () => {
  const before = digest([entry("a"), entry("b"), entry("c")]);
  const after = withStatus(before, "b", "saved");
  assert.deepEqual(
    after.entries.map((e) => [e.arxiv_id, e.status]),
    [
      ["a", "new"],
      ["b", "saved"],
      ["c", "new"],
    ],
  );
  assert.equal(before.entries[1].status, "new");
  assert.equal(
    withStatus(before, "zzz", "saved").entries.every((e) => e.status === "new"),
    true,
  );
});

test("withEntry swaps in the server's version", () => {
  const before = digest([entry("a"), entry("b")]);
  const server = { ...entry("b", "dismissed"), title: "From the server" };
  const after = withEntry(before, server);
  assert.equal(after.entries[1].title, "From the server");
  assert.equal(after.entries[0].title, "Invented a");
});

test("status counts and lines", () => {
  const counts = statusCounts([
    entry("a"),
    entry("b", "saved"),
    entry("c", "saved"),
    entry("d", "dismissed"),
  ]);
  assert.deepEqual(counts, { new: 1, saved: 2, dismissed: 1 });
  assert.equal(statusLine("new"), null);
  assert.match(statusLine("saved") ?? "", /Saved/);
  assert.match(statusLine("dismissed") ?? "", /not come back/);
});

test("parseDay accepts only real calendar days", () => {
  assert.equal(parseDay("2026-10-19"), "2026-10-19");
  assert.equal(parseDay("2024-02-29"), "2024-02-29");
  for (const bad of [
    "2026-02-30",
    "2026-13-01",
    "2026-1-1",
    "yesterday",
    "",
    null,
    undefined,
    "2026-10-19T00:00",
  ]) {
    assert.equal(parseDay(bad), null, String(bad));
  }
});

test("formatDay reads the day as a calendar day, whatever the time zone", () => {
  assert.equal(formatDay("2026-10-19"), "Mon, Oct 19, 2026");
  assert.equal(formatDay("2026-01-01"), "Thu, Jan 1, 2026");
  assert.equal(formatDay("not a day"), "not a day");
});

test("authorLine", () => {
  assert.equal(authorLine(["A", "B"], 2), "A, B");
  assert.equal(authorLine(["A", "B", "C"], 7), "A, B, C and 4 more");
  assert.equal(authorLine([], 0), "");
});

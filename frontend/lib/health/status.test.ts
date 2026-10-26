import { strict as assert } from "node:assert";
import { test } from "node:test";

import type { StatusOut } from "../api/endpoints.ts";
import { formatAge, paperRows, summarize } from "./status.ts";

const NOW = new Date("2026-10-26T12:00:00Z");

function status(over: Partial<StatusOut> = {}): StatusOut {
  return {
    stores: { postgres: "up", qdrant: "up", redis: "up" },
    database: {
      papers_total: 10,
      papers_by_fulltext_status: { none: 8, indexed: 2 },
      last_runs: [
        {
          source: "embed-paper",
          started_at: "2026-10-26T07:00:00Z",
          finished_at: "2026-10-26T07:01:00Z",
          ok: true,
          error: null,
          skipped: null,
          abandoned: false,
          note: null,
        },
      ],
      budget: {
        cap_usd: 5,
        spent_usd: 1,
        paused: false,
        month_start: "2026-10-01T00:00:00Z",
      },
    },
    queue_length: 0,
    qdrant_points: 100,
    errors: {},
    ...over,
  };
}

test("a healthy system has no problems", () => {
  assert.deepEqual(summarize(status()), { level: "ok", problems: [] });
});

test("a store that is down makes the system down, and is named", () => {
  const s = summarize(
    status({ stores: { postgres: "up", qdrant: "down", redis: "up" } }),
  );
  assert.equal(s.level, "down");
  assert.deepEqual(s.problems, ["Qdrant is not reachable."]);
});

test("an unreadable database is down, and its reason is not repeated", () => {
  const s = summarize(
    status({ database: null, errors: { database: "no answer within 3 s" } }),
  );
  assert.equal(s.level, "down");
  assert.deepEqual(s.problems, ["The database numbers could not be read."]);
});

test("a paused budget, a failed step and failed papers are warnings", () => {
  const base = status();
  const database = {
    ...base.database!,
    papers_by_fulltext_status: { indexed: 1, failed: 2 },
    last_runs: [
      {
        source: "index-abstracts",
        started_at: "2026-10-26T07:00:00Z",
        finished_at: "2026-10-26T07:00:05Z",
        ok: false,
        error: "RuntimeError: boom",
        skipped: null,
        abandoned: false,
        note: null,
      },
      {
        // still running: not finished, so not a failure
        source: "select-fulltext",
        started_at: "2026-10-26T07:00:00Z",
        finished_at: null,
        ok: false,
        error: null,
        skipped: null,
        abandoned: false,
        note: null,
      },
    ],
    budget: { ...base.database!.budget, paused: true, spent_usd: 5.2 },
  };
  const s = summarize(status({ database, errors: { queue_length: "gone" } }));
  assert.equal(s.level, "warning");
  assert.equal(s.problems.length, 4);
  assert.match(s.problems[0]!, /budget cap is reached/);
  assert.match(
    s.problems[1]!,
    /A step failed in its last run: index-abstracts/,
  );
  assert.match(s.problems[2]!, /2 papers could not get full text/);
  assert.match(s.problems[3]!, /queue_length could not be read: gone/);
});

test("formatAge", () => {
  assert.equal(formatAge("2026-10-26T11:59:40Z", NOW), "just now");
  assert.equal(formatAge("2026-10-26T11:55:00Z", NOW), "5 min ago");
  assert.equal(formatAge("2026-10-26T09:00:00Z", NOW), "3 h ago");
  assert.equal(formatAge("2026-10-23T12:00:00Z", NOW), "3 d ago");
  assert.equal(formatAge("not a date", NOW), "not a date");
});

test("paperRows are in a fixed order and an unknown status comes last", () => {
  const rows = paperRows({ failed: 1, weird: 4, none: 8, indexed: 2 });
  assert.deepEqual(
    rows.map((r) => [r.status, r.count]),
    [
      ["indexed", 2],
      ["none", 8],
      ["failed", 1],
      ["weird", 4],
    ],
  );
  assert.equal(rows[1]!.label, "Abstract only");
  assert.equal(rows[3]!.label, "weird");
});

test("a run that never finished is reported as abandoned, a running one is not", () => {
  const base = status();
  const started = "2026-10-04T06:56:00Z";
  const database = {
    ...base.database!,
    last_runs: [
      {
        source: "arxiv-daily",
        started_at: started,
        finished_at: null,
        ok: false,
        error: "abandoned: started 533 h ago and never finished",
        skipped: null,
        abandoned: true,
        note: null,
      },
      {
        source: "embed-paper",
        started_at: "2026-10-26T11:00:00Z",
        finished_at: null,
        ok: false,
        error: null,
        skipped: null,
        abandoned: false,
        note: null,
      },
    ],
  };
  const s = summarize(status({ database }));
  assert.equal(s.level, "warning");
  assert.deepEqual(s.problems, [
    "A step never finished (abandoned, the worker stopped during the run): arxiv-daily.",
  ]);
});

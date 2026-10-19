import { strict as assert } from "node:assert";
import { test } from "node:test";

import { cellCounts, toPanelSource } from "./panel.ts";

test("a comparison source becomes a panel source with the same fields", () => {
  const out = toPanelSource({
    n: 3,
    chunk_id: "00000000-0000-4000-8000-000000000001",
    arxiv_id: "2601.00001",
    chunk_type: "section",
    section_path: null,
    title: "Invented",
    published_at: "2026-01-05T00:00:00Z",
    abs_url: "https://arxiv.org/abs/2601.00001",
  });
  assert.deepEqual(out, {
    n: 3,
    chunk_id: "00000000-0000-4000-8000-000000000001",
    arxiv_id: "2601.00001",
    title: "Invented",
    section_path: null,
    chunk_type: "section",
    published_at: "2026-01-05T00:00:00Z",
    abs_url: "https://arxiv.org/abs/2601.00001",
  });
});

test("cell counts split cited from not stated", () => {
  const cell = (status: "cited" | "not_stated") => ({
    dimension: "task" as const,
    text: "x",
    status,
    citations: [],
  });
  const row = (cells: ReturnType<typeof cell>[]) => ({
    arxiv_id: "a",
    title: "t",
    has_fulltext: true,
    abs_url: "https://arxiv.org/abs/a",
    cells,
  });
  const counts = cellCounts({
    dimensions: ["task"],
    rows: [row([cell("cited"), cell("not_stated")]), row([cell("cited")])],
    sources: [],
    notes: {
      invalid_citations_dropped: 0,
      uncited_cells_converted: 0,
      missing_cells: 0,
      truncated_cells: 0,
      cut_sources: 0,
      top_k_used: 3,
    },
    tokens_in: 0,
    tokens_out: 0,
    cost_usd: 0,
    prompt_version: "compare_v1",
  });
  assert.deepEqual(counts, { cited: 2, notStated: 1, total: 3 });
});

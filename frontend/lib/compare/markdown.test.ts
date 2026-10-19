import { strict as assert } from "node:assert";
import { test } from "node:test";

import { compareToMarkdown, markdownFileName } from "./markdown.ts";
import type { CompareOut } from "./markdown.ts";

const CHUNK_A = "00000000-0000-4000-8000-00000000000a";
const CHUNK_B = "00000000-0000-4000-8000-00000000000b";

const RESULT: CompareOut = {
  dimensions: ["task", "data"],
  rows: [
    {
      arxiv_id: "2601.00001",
      title: "Invented | Paper One",
      has_fulltext: true,
      abs_url: "https://arxiv.org/abs/2601.00001",
      cells: [
        {
          dimension: "task",
          text: "Trains small adapters.\nSecond line.",
          status: "cited",
          citations: [
            {
              n: 1,
              chunk_id: CHUNK_A,
              chunk_type: "section",
              section_path: "3 Method",
            },
          ],
        },
        {
          dimension: "data",
          text: "Not stated in the sources.",
          status: "not_stated",
          citations: [],
        },
      ],
    },
    {
      arxiv_id: "2601.00002",
      title: "Invented Paper Two",
      has_fulltext: false,
      abs_url: "https://arxiv.org/abs/2601.00002",
      cells: [
        {
          dimension: "task",
          text: "Studies adapters.",
          status: "cited",
          citations: [
            {
              n: 2,
              chunk_id: CHUNK_B,
              chunk_type: "abstract",
              section_path: null,
            },
            {
              n: 3,
              chunk_id: CHUNK_B,
              chunk_type: "abstract",
              section_path: null,
            },
          ],
        },
        {
          dimension: "data",
          text: "Not stated in the sources.",
          status: "not_stated",
          citations: [],
        },
      ],
    },
  ],
  sources: [
    {
      n: 1,
      chunk_id: CHUNK_A,
      arxiv_id: "2601.00001",
      chunk_type: "section",
      section_path: "3 Method",
      title: "Invented | Paper One",
      published_at: "2026-01-05T00:00:00Z",
      abs_url: "https://arxiv.org/abs/2601.00001",
    },
    {
      n: 2,
      chunk_id: CHUNK_B,
      arxiv_id: "2601.00002",
      chunk_type: "abstract",
      section_path: null,
      title: "Invented Paper Two",
      published_at: "2026-01-06T00:00:00Z",
      abs_url: "https://arxiv.org/abs/2601.00002",
    },
    {
      n: 3,
      chunk_id: CHUNK_B,
      arxiv_id: "2601.00002",
      chunk_type: "abstract",
      section_path: null,
      title: "Invented Paper Two",
      published_at: "2026-01-06T00:00:00Z",
      abs_url: "javascript:alert(1)", // never turned into a link
    },
    {
      n: 4,
      chunk_id: CHUNK_B,
      arxiv_id: "2601.00002",
      chunk_type: "abstract",
      section_path: null,
      title: "Not cited, so not listed",
      published_at: "2026-01-06T00:00:00Z",
      abs_url: "https://arxiv.org/abs/2601.00002",
    },
  ],
  notes: {
    invalid_citations_dropped: 0,
    uncited_cells_converted: 0,
    missing_cells: 0,
    truncated_cells: 0,
    cut_sources: 0,
    top_k_used: 3,
  },
  tokens_in: 1000,
  tokens_out: 100,
  cost_usd: 0.000225,
  prompt_version: "compare_v1",
};

test("a table with the papers as columns, markers in cells and untouched not-stated cells", () => {
  const md = compareToMarkdown(RESULT);
  const lines = md.split("\n");
  assert.equal(lines[0], "# Paper comparison");
  assert.equal(
    lines[2],
    "| Dimension | Invented \\| Paper One (arXiv:2601.00001) | Invented Paper Two (arXiv:2601.00002, abstract only) |",
  );
  assert.equal(lines[3], "| --- | --- | --- |");
  assert.equal(
    lines[4],
    "| Task | Trains small adapters. Second line. [1] | Studies adapters. [2][3] |",
  );
  assert.equal(
    lines[5],
    "| Data | _Not stated in the sources._ | _Not stated in the sources._ |",
  );
});

test("the sources list has only cited sources, links abstract pages and drops unsafe links", () => {
  const md = compareToMarkdown(RESULT);
  assert.ok(
    md.includes(
      "1. Invented | Paper One (arXiv:2601.00001), 3 Method. https://arxiv.org/abs/2601.00001",
    ),
  );
  assert.ok(
    md.includes(
      "2. Invented Paper Two (arXiv:2601.00002), abstract. https://arxiv.org/abs/2601.00002",
    ),
  );
  assert.ok(
    md.includes("3. Invented Paper Two (arXiv:2601.00002), abstract\n"),
  );
  assert.ok(!md.includes("javascript:"));
  assert.ok(!md.includes("Not cited, so not listed"));
  assert.ok(!md.includes(".pdf"));
  assert.ok(
    md.includes("Abstract only (no full text indexed): arXiv:2601.00002."),
  );
  assert.ok(md.includes("Prompt compare_v1."));
});

test("the export is deterministic and the file name is safe", () => {
  assert.equal(compareToMarkdown(RESULT), compareToMarkdown(RESULT));
  assert.equal(markdownFileName(RESULT), "comparison-2601.00001_2601.00002.md");
  const old = {
    ...RESULT,
    rows: [{ ...RESULT.rows[0]!, arxiv_id: "hep-th/9901001" }],
  };
  assert.equal(markdownFileName(old), "comparison-hep-th-9901001.md");
});

test("a table where nothing is cited says so", () => {
  const empty: CompareOut = {
    ...RESULT,
    rows: RESULT.rows.map((row) => ({
      ...row,
      cells: row.cells.map((c) => ({
        ...c,
        text: "Not stated in the sources.",
        status: "not_stated" as const,
        citations: [],
      })),
    })),
  };
  assert.ok(
    compareToMarkdown(empty).includes("No cell is supported by a source."),
  );
});

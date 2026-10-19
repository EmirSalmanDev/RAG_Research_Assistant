// A comparison source in the shape the sources side panel (the chat's) expects.
// Plain TypeScript.

import type { Source } from "../chat/events.ts";
import type { components } from "../api/schema";

type SourceOut = components["schemas"]["SourceOut"];

export function toPanelSource(source: SourceOut): Source {
  return {
    n: source.n,
    chunk_id: source.chunk_id,
    arxiv_id: source.arxiv_id,
    title: source.title,
    section_path: source.section_path ?? null,
    chunk_type: source.chunk_type as Source["chunk_type"],
    published_at: source.published_at,
    abs_url: source.abs_url,
  };
}

/** Counts for the line under the table: how many cells carry a claim with a source. */
export function cellCounts(result: components["schemas"]["CompareOut"]): {
  cited: number;
  notStated: number;
  total: number;
} {
  const cells = result.rows.flatMap((row) => row.cells);
  const cited = cells.filter((c) => c.status === "cited").length;
  return { cited, notStated: cells.length - cited, total: cells.length };
}

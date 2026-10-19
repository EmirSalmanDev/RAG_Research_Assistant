"use client";

import Link from "next/link";

import { CitationChip } from "@/components/chat/citation-chip";
import type { CompareOut } from "@/lib/compare/markdown";
import { toPanelSource } from "@/lib/compare/panel";
import { dimensionLabel } from "@/lib/compare/selection";
import type { Source } from "@/lib/chat/events";
import { paperPath } from "@/lib/links";
import { cn } from "@/lib/utils";

// Columns are the papers, rows the dimensions. A cell either shows a claim WITH its
// citation chips, or says that the sources do not state it: the API never returns text
// for a cell without a valid citation (backend/app/features/compare.py), and this view
// never invents any. A chip opens the same sources panel as in the chat.
export function CompareTable({
  result,
  openSource,
  activeChunkId,
}: {
  result: CompareOut;
  openSource: (source: Source, all: readonly Source[]) => void;
  activeChunkId: string | null;
}) {
  const sources = new Map(result.sources.map((s) => [s.n, toPanelSource(s)]));
  return (
    <div className="overflow-x-auto rounded-md border bg-card">
      <table className="w-full min-w-[40rem] border-collapse text-left text-sm">
        <caption className="sr-only">
          Comparison of {result.rows.length} papers by{" "}
          {result.dimensions.map(dimensionLabel).join(", ")}
        </caption>
        <thead>
          <tr className="border-b bg-muted/50 align-top">
            <th scope="col" className="w-28 p-3 font-medium">
              Dimension
            </th>
            {result.rows.map((row) => (
              <th
                key={row.arxiv_id}
                scope="col"
                className="min-w-48 space-y-1 p-3 align-top font-normal"
              >
                <Link
                  href={paperPath(row.arxiv_id)}
                  className="block font-semibold leading-snug underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {row.title}
                </Link>
                <span className="block text-xs text-muted-foreground">
                  arXiv {row.arxiv_id}
                </span>
                <span
                  className={cn(
                    "inline-block rounded border px-1.5 py-0.5 text-xs",
                    row.has_fulltext
                      ? "border-primary/40"
                      : "border-dashed text-muted-foreground",
                  )}
                >
                  {row.has_fulltext ? "full text" : "abstract only"}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {result.dimensions.map((dimension, i) => (
            <tr key={dimension} className="border-b last:border-b-0 align-top">
              <th scope="row" className="p-3 font-medium">
                {dimensionLabel(dimension)}
              </th>
              {result.rows.map((row) => {
                const cell = row.cells[i];
                if (!cell || cell.status === "not_stated") {
                  return (
                    <td
                      key={row.arxiv_id}
                      data-status="not_stated"
                      className="p-3 italic text-muted-foreground"
                    >
                      Not stated in the sources.
                    </td>
                  );
                }
                const own = result.sources
                  .filter((s) => s.arxiv_id === row.arxiv_id)
                  .map(toPanelSource);
                return (
                  <td
                    key={row.arxiv_id}
                    data-status="cited"
                    className="space-y-1 p-3 leading-6"
                  >
                    <span>{cell.text}</span>{" "}
                    <span className="whitespace-nowrap">
                      {cell.citations.map((c) => {
                        const source = sources.get(c.n);
                        return source ? (
                          <CitationChip
                            key={c.n}
                            source={source}
                            active={activeChunkId === source.chunk_id}
                            onOpen={(s) => openSource(s, own)}
                          />
                        ) : null;
                      })}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

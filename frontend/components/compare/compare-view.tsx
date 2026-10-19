"use client";

import { useMutation } from "@tanstack/react-query";
import { useReducer, useState } from "react";

import { SourcePanel } from "@/components/chat/source-panel";
import { Button } from "@/components/ui/button";
import { postCompare } from "@/lib/api/queries";
import type { CompareOut } from "@/lib/api/queries";
import type { Source } from "@/lib/chat/events";
import { panelReducer } from "@/lib/chat/panel";
import { compareToMarkdown, markdownFileName } from "@/lib/compare/markdown";
import { cellCounts } from "@/lib/compare/panel";
import {
  ALL_DIMENSIONS,
  DIMENSIONS,
  compareProblem,
  compareRequest,
  removePaper,
  toggleDimension,
  togglePaper,
} from "@/lib/compare/selection";
import type { Dimension, PickedPaper } from "@/lib/compare/selection";
import { cn } from "@/lib/utils";

import { CompareTable } from "./compare-table";
import { PaperPicker } from "./paper-picker";

function download(name: string, text: string) {
  const url = URL.createObjectURL(
    new Blob([text], { type: "text/markdown;charset=utf-8" }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function ExportBar({ result }: { result: CompareOut }) {
  const [copied, setCopied] = useState<"yes" | "failed" | null>(null);
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        type="button"
        variant="outline"
        onClick={() =>
          download(markdownFileName(result), compareToMarkdown(result))
        }
      >
        Download as markdown
      </Button>
      <Button
        type="button"
        variant="outline"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(compareToMarkdown(result));
            setCopied("yes");
          } catch {
            setCopied("failed"); // the clipboard can be blocked; the download still works
          }
        }}
      >
        Copy markdown
      </Button>
      <span role="status" className="text-sm text-muted-foreground">
        {copied === "yes" && "Copied to the clipboard."}
        {copied === "failed" && "Could not copy. Use the download instead."}
      </span>
    </div>
  );
}

function Result({
  result,
  onOpen,
  activeChunkId,
}: {
  result: CompareOut;
  onOpen: (source: Source, all: readonly Source[]) => void;
  activeChunkId: string | null;
}) {
  const counts = cellCounts(result);
  const abstractOnly = result.rows.filter((r) => !r.has_fulltext);
  const { notes } = result;
  return (
    <section aria-label="Comparison" className="space-y-4">
      <CompareTable
        result={result}
        openSource={onOpen}
        activeChunkId={activeChunkId}
      />
      <div className="space-y-1 text-sm text-muted-foreground">
        <p data-testid="cell-counts">
          {counts.cited} of {counts.total} cells have a claim with a source.{" "}
          {counts.notStated} say &ldquo;Not stated in the sources.&rdquo;
        </p>
        {abstractOnly.length > 0 && (
          <p>
            {abstractOnly.length === 1 ? "One paper has" : "Some papers have"}{" "}
            only an abstract indexed, so cells about data, compute or metrics
            are often not stated. Their citations are marked &ldquo;abstract
            only&rdquo;.
          </p>
        )}
        {(notes.invalid_citations_dropped > 0 ||
          notes.uncited_cells_converted > 0) && (
          <p>
            The model cited {notes.invalid_citations_dropped} source number
            {notes.invalid_citations_dropped === 1 ? "" : "s"} that do not
            belong to the paper; {notes.uncited_cells_converted} cell
            {notes.uncited_cells_converted === 1 ? " was" : "s were"} left
            unstated for lack of a valid source.
          </p>
        )}
        <p>
          One model call ({result.prompt_version}): {result.tokens_in} tokens
          in, {result.tokens_out} out, about ${result.cost_usd.toFixed(4)}.
        </p>
      </div>
      <ExportBar result={result} />
    </section>
  );
}

export function CompareView() {
  const [selected, setSelected] = useState<PickedPaper[]>([]);
  const [dimensions, setDimensions] = useState<Dimension[]>([
    ...ALL_DIMENSIONS,
  ]);
  const [notice, setNotice] = useState<string | null>(null);
  const [panel, dispatchPanel] = useReducer(panelReducer, null);

  // The ONLY paid call on this page, started by the Compare button below.
  const compare = useMutation({
    mutationFn: () => postCompare(compareRequest(selected, dimensions)),
    onMutate: () => dispatchPanel({ type: "close" }),
  });
  const problem = compareProblem(selected, dimensions);

  return (
    <div className="mx-auto w-full max-w-6xl space-y-8 px-4 py-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">
          Compare papers
        </h1>
        <p className="text-sm text-muted-foreground">
          Pick 2 to 5 papers and the dimensions to compare. Every claim in the
          table has a citation that opens the passage it rests on. Nothing is
          sent until you press Compare.
        </p>
      </div>

      <PaperPicker
        selected={selected}
        notice={notice}
        onToggle={(paper) => {
          const next = togglePaper(selected, paper);
          setSelected(next.selected);
          setNotice(
            next.refused === "full"
              ? "You can compare at most 5 papers. Remove one first."
              : null,
          );
        }}
        onRemove={(id) => {
          setSelected(removePaper(selected, id));
          setNotice(null);
        }}
      />

      <fieldset className="space-y-2">
        <legend className="text-sm font-medium">Dimensions</legend>
        <div className="flex flex-wrap gap-1.5">
          {DIMENSIONS.map(({ id, label }) => {
            const on = dimensions.includes(id);
            return (
              <button
                key={id}
                type="button"
                aria-pressed={on}
                onClick={() => setDimensions(toggleDimension(dimensions, id))}
                className={cn(
                  "rounded-md border px-2.5 py-1 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  on
                    ? "border-primary bg-primary text-primary-foreground"
                    : "bg-card hover:bg-accent",
                )}
              >
                {label}
              </button>
            );
          })}
        </div>
      </fieldset>

      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type="button"
            disabled={problem !== null || compare.isPending}
            onClick={() => compare.mutate()}
          >
            Compare
          </Button>
          <p className="text-sm text-muted-foreground">
            {problem ??
              "Compare makes one embeddings request and one model call (a fraction of a cent)."}
          </p>
        </div>
        {compare.isPending && (
          <p role="status" className="text-sm text-muted-foreground">
            Comparing&hellip; this takes a few seconds.
          </p>
        )}
        {compare.isError && (
          <p
            role="alert"
            className="rounded-md border border-destructive/40 p-3 text-sm text-destructive"
          >
            {compare.error.message}
          </p>
        )}
      </div>

      {compare.data && (
        <Result
          result={compare.data}
          activeChunkId={panel?.source.chunk_id ?? null}
          onOpen={(source, all) => dispatchPanel({ type: "open", source, all })}
        />
      )}
      {!compare.data && !compare.isPending && !compare.isError && (
        <p className="text-sm text-muted-foreground">
          The table appears here. A paper with only its abstract indexed is
          marked &ldquo;abstract only&rdquo;; many of its cells will say
          &ldquo;Not stated in the sources&rdquo;.
        </p>
      )}

      <SourcePanel
        source={panel?.source ?? null}
        sources={panel?.all ?? []}
        onSelect={(source) => dispatchPanel({ type: "select", source })}
        onClose={() => dispatchPanel({ type: "close" })}
      />
    </div>
  );
}

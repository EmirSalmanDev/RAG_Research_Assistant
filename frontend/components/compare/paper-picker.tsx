"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { Input } from "@/components/ui/input";
import { fetchPapers } from "@/lib/api/queries";
import type { PaperListItemOut } from "@/lib/api/queries";
import { MAX_PAPERS } from "@/lib/compare/selection";
import type { PickedPaper } from "@/lib/compare/selection";
import { formatDate } from "@/lib/format";
import { cn } from "@/lib/utils";

const SEARCH_DELAY_MS = 250;
const MIN_SEARCH_CHARS = 2;

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return debounced;
}

function pick(paper: PaperListItemOut): PickedPaper {
  return {
    arxiv_id: paper.arxiv_id,
    title: paper.title,
    has_fulltext: paper.has_fulltext,
  };
}

function PaperRow({
  paper,
  picked,
  onToggle,
}: {
  paper: PaperListItemOut;
  picked: boolean;
  onToggle: (paper: PickedPaper) => void;
}) {
  return (
    <li>
      <button
        type="button"
        aria-pressed={picked}
        onClick={() => onToggle(pick(paper))}
        className={cn(
          "flex w-full items-start gap-3 rounded-md border px-3 py-2 text-left text-sm transition-colors",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          picked
            ? "border-primary bg-accent text-accent-foreground"
            : "bg-card hover:bg-accent",
        )}
      >
        <span
          aria-hidden
          className={cn(
            "mt-0.5 flex size-4 shrink-0 items-center justify-center rounded border text-[10px]",
            picked && "border-primary bg-primary text-primary-foreground",
          )}
        >
          {picked ? "✓" : ""}
        </span>
        <span className="flex-1 space-y-0.5">
          <span className="block font-medium leading-snug">{paper.title}</span>
          <span className="block text-xs text-muted-foreground">
            arXiv {paper.arxiv_id}, {formatDate(paper.published_at)}
          </span>
        </span>
        <span
          className={cn(
            "shrink-0 rounded border px-1.5 py-0.5 text-xs",
            paper.has_fulltext
              ? "border-primary/40 text-foreground"
              : "border-dashed text-muted-foreground",
          )}
        >
          {paper.has_fulltext ? "full text" : "abstract only"}
        </span>
      </button>
    </li>
  );
}

function PaperList({
  papers,
  selected,
  onToggle,
}: {
  papers: readonly PaperListItemOut[];
  selected: readonly PickedPaper[];
  onToggle: (paper: PickedPaper) => void;
}) {
  return (
    <ul className="space-y-1.5">
      {papers.map((paper) => (
        <PaperRow
          key={paper.arxiv_id}
          paper={paper}
          picked={selected.some((p) => p.arxiv_id === paper.arxiv_id)}
          onToggle={onToggle}
        />
      ))}
    </ul>
  );
}

// Choose 2 to 5 papers: a quick list of the papers that have full text, and a title or
// arXiv-id search. Both lookups read Postgres only (GET /papers): typing here never makes
// a paid call.
export function PaperPicker({
  selected,
  onToggle,
  onRemove,
  notice,
}: {
  selected: readonly PickedPaper[];
  onToggle: (paper: PickedPaper) => void;
  onRemove: (arxivId: string) => void;
  notice: string | null;
}) {
  const [text, setText] = useState("");
  const q = useDebounced(text.trim(), SEARCH_DELAY_MS);
  const searching = q.length >= MIN_SEARCH_CHARS;

  const quick = useQuery({
    queryKey: ["papers", "fulltext"],
    queryFn: ({ signal }) => fetchPapers({ q: "", hasFulltext: true }, signal),
  });
  const found = useQuery({
    queryKey: ["papers", "search", q],
    queryFn: ({ signal }) => fetchPapers({ q }, signal),
    enabled: searching,
  });

  return (
    <section aria-label="Choose papers" className="space-y-4">
      <div className="space-y-2">
        <p className="text-sm font-medium">
          Papers to compare ({selected.length} of {MAX_PAPERS})
        </p>
        {selected.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Pick 2 to 5 papers below.
          </p>
        ) : (
          <ul aria-label="Picked papers" className="flex flex-wrap gap-2">
            {selected.map((paper) => (
              <li
                key={paper.arxiv_id}
                className="flex items-center gap-1 rounded-md border bg-card py-1 pl-2.5 pr-1 text-sm"
              >
                <span className="max-w-[16rem] truncate" title={paper.title}>
                  {paper.title}
                </span>
                {!paper.has_fulltext && (
                  <span className="rounded border border-dashed px-1 text-xs text-muted-foreground">
                    abstract only
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => onRemove(paper.arxiv_id)}
                  aria-label={`Remove ${paper.title}`}
                  className="rounded px-1.5 text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
        {notice && (
          <p role="alert" className="text-sm text-destructive">
            {notice}
          </p>
        )}
      </div>

      <div className="space-y-2">
        <label htmlFor="paper-search" className="text-sm font-medium">
          Find a paper by title or arXiv id
        </label>
        <Input
          id="paper-search"
          value={text}
          onChange={(e) => setText(e.target.value)}
          maxLength={200}
          placeholder="attention, 1706.03762, ..."
          className="bg-card"
          autoComplete="off"
        />
        {searching && found.isPending && (
          <p role="status" className="text-sm text-muted-foreground">
            Searching&hellip;
          </p>
        )}
        {found.isError && (
          <p role="alert" className="text-sm text-destructive">
            {found.error.message}
          </p>
        )}
        {searching && found.data && (
          <div className="space-y-2" aria-label="Search results" role="group">
            {found.data.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No paper matches &ldquo;{q}&rdquo;.
              </p>
            ) : (
              <PaperList
                papers={found.data}
                selected={selected}
                onToggle={onToggle}
              />
            )}
          </div>
        )}
      </div>

      <div className="space-y-2">
        <p className="text-sm font-medium">Papers with full text</p>
        {quick.isPending && (
          <p role="status" className="text-sm text-muted-foreground">
            Loading&hellip;
          </p>
        )}
        {quick.isError && (
          <p role="alert" className="text-sm text-destructive">
            {quick.error.message}
          </p>
        )}
        {quick.data && quick.data.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No paper has full text indexed yet. Abstract-only papers can still
            be compared, with fewer cells filled.
          </p>
        )}
        {quick.data && quick.data.length > 0 && (
          <PaperList
            papers={quick.data}
            selected={selected}
            onToggle={onToggle}
          />
        )}
      </div>
    </section>
  );
}

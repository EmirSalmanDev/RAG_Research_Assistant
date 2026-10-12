"use client";

import type { Source } from "@/lib/chat/events";
import { cn } from "@/lib/utils";

// A footnote-style marker for [n]. A dashed border means the source is only the paper's
// abstract (weaker evidence than a passage from the full text), and says so in words.
export function CitationChip({
  source,
  active,
  onOpen,
}: {
  source: Source;
  active?: boolean;
  onOpen: (source: Source) => void;
}) {
  const abstractOnly = source.chunk_type === "abstract";
  return (
    <button
      type="button"
      onClick={() => onOpen(source)}
      aria-label={`Source ${source.n}: ${source.title}${abstractOnly ? " (abstract only)" : ""}`}
      aria-haspopup="dialog"
      className={cn(
        "mx-0.5 inline-flex items-baseline gap-1 rounded border px-1.5 font-sans text-xs font-medium leading-5 align-baseline transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        abstractOnly ? "border-dashed border-primary/60" : "border-primary/40",
        active
          ? "bg-primary text-primary-foreground"
          : "bg-accent text-accent-foreground hover:bg-primary hover:text-primary-foreground",
      )}
    >
      <span>{source.n}</span>
      {abstractOnly && (
        <span className="text-[10px] font-normal">abstract</span>
      )}
    </button>
  );
}

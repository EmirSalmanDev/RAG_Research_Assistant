"use client";

import { useId, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { excerpt } from "@/lib/format";

// One passage (a chunk of a paper) in a list: its place in the paper, a badge when it is
// only the abstract, a short excerpt, and a button that expands it to the full text.
// Used by the search results and by the paper page.
export function Passage({
  sectionPath,
  chunkType,
  text,
}: {
  sectionPath: string | null;
  chunkType: string;
  text: string;
}) {
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  const short = excerpt(text, 260);

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        {chunkType === "abstract" && (
          <Badge variant="outline" className="border-dashed font-normal">
            abstract
          </Badge>
        )}
        {chunkType === "analysis" && (
          <Badge variant="secondary" className="font-normal">
            generated analysis
          </Badge>
        )}
        {sectionPath && (
          <span className="text-muted-foreground">{sectionPath}</span>
        )}
      </div>
      <p id={bodyId} className="whitespace-pre-wrap font-serif leading-7">
        {open || !short.truncated ? text : short.text}
      </p>
      {short.truncated && (
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={() => setOpen(!open)}
          className="rounded text-sm text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {open ? "Show less" : "Show the full passage"}
        </button>
      )}
    </div>
  );
}

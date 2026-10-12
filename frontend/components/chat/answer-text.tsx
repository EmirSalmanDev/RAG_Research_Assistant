"use client";

import { useMemo } from "react";

import { splitCitations } from "@/lib/chat/citations";
import type { Source } from "@/lib/chat/events";

import { CitationChip } from "./citation-chip";

// The answer as reading text, with every [n] that has a source turned into a chip.
// whitespace-pre-wrap keeps the model's line breaks and list layout without a markdown
// library; a number with no source stays as written (see lib/chat/citations.ts).
export function AnswerText({
  text,
  sources,
  streaming,
  activeChunkId,
  onOpen,
}: {
  text: string;
  sources: readonly Source[];
  streaming: boolean;
  activeChunkId?: string;
  onOpen: (source: Source) => void;
}) {
  const segments = useMemo(
    () => splitCitations(text, sources, { streaming }),
    [text, sources, streaming],
  );
  return (
    <div className="whitespace-pre-wrap font-serif text-[1.0625rem] leading-8">
      {segments.map((segment, i) =>
        segment.kind === "text" ? (
          <span key={i}>{segment.text}</span>
        ) : (
          <CitationChip
            key={i}
            source={segment.source}
            active={segment.source.chunk_id === activeChunkId}
            onOpen={onOpen}
          />
        ),
      )}
      {streaming && (
        <span
          aria-hidden="true"
          className="ml-0.5 inline-block h-4 w-0.5 translate-y-0.5 animate-pulse bg-foreground/60"
        />
      )}
    </div>
  );
}

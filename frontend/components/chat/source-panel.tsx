"use client";

import { useQuery } from "@tanstack/react-query";

import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { fetchChunk } from "@/lib/api/queries";
import type { Source } from "@/lib/chat/events";
import { cn } from "@/lib/utils";

const ARXIV_ABS_PREFIX = "https://arxiv.org/abs/";

const KIND_LABEL: Record<Source["chunk_type"], string> = {
  abstract: "Abstract only",
  section: "Full-text passage",
  analysis: "Generated analysis",
};

function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString("en", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
}

// The cited chunk, in full, next to the answer. The chunk text is not in the chat
// stream, so it is fetched when a chip is opened (and cached by TanStack Query).
export function SourcePanel({
  source,
  sources,
  onSelect,
  onClose,
}: {
  source: Source | null;
  sources: readonly Source[];
  onSelect: (source: Source) => void;
  onClose: () => void;
}) {
  const chunk = useQuery({
    queryKey: ["chunk", source?.chunk_id],
    queryFn: () => fetchChunk(source?.chunk_id ?? ""),
    enabled: source !== null,
  });
  // The link comes from the API; only a real arXiv abstract page is ever linked.
  const absUrl = source?.abs_url.startsWith(ARXIV_ABS_PREFIX)
    ? source.abs_url
    : null;

  return (
    <Sheet open={source !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-0 overflow-y-auto p-0 sm:max-w-lg"
      >
        {source && (
          <>
            <SheetHeader className="space-y-1 border-b p-6 pr-12 text-left">
              <SheetTitle className="font-serif text-xl leading-snug">
                {source.title}
              </SheetTitle>
              <SheetDescription>
                arXiv {source.arxiv_id}, {formatDate(source.published_at)}
              </SheetDescription>
            </SheetHeader>

            <div className="flex-1 space-y-6 p-6">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <Badge
                    variant={
                      source.chunk_type === "abstract" ? "outline" : "secondary"
                    }
                  >
                    {KIND_LABEL[source.chunk_type]}
                  </Badge>
                  {source.section_path && (
                    <span className="text-muted-foreground">
                      {source.section_path}
                    </span>
                  )}
                </div>
                {source.chunk_type === "abstract" && (
                  <p className="text-sm text-muted-foreground">
                    Only this paper&rsquo;s abstract is indexed, so the claim
                    rests on the summary, not on the paper&rsquo;s own evidence.
                  </p>
                )}
              </div>

              <figure className="border-l-2 border-primary/50 pl-4">
                {chunk.isPending && (
                  <p role="status" className="text-sm text-muted-foreground">
                    Loading the passage&hellip;
                  </p>
                )}
                {chunk.isError && (
                  <p role="alert" className="text-sm text-destructive">
                    {chunk.error.message}
                  </p>
                )}
                {chunk.data && (
                  <blockquote
                    data-testid="cited-passage"
                    className="whitespace-pre-wrap font-serif text-base leading-7"
                  >
                    {chunk.data.text}
                  </blockquote>
                )}
              </figure>

              {absUrl && (
                <a
                  href={absUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center rounded-md border px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  Read the abstract on arXiv
                </a>
              )}

              {sources.length > 1 && (
                <nav
                  aria-label="Other sources in this answer"
                  className="space-y-2 border-t pt-4"
                >
                  <p className="text-sm font-medium">
                    Other sources in this answer
                  </p>
                  <ul className="space-y-1">
                    {sources
                      .filter((s) => s.chunk_id !== source.chunk_id)
                      .map((s) => (
                        <li key={s.chunk_id}>
                          <button
                            type="button"
                            onClick={() => onSelect(s)}
                            className={cn(
                              "flex w-full items-baseline gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors hover:bg-accent",
                              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                            )}
                          >
                            <span className="w-5 shrink-0 font-medium">
                              {s.n}
                            </span>
                            <span className="flex-1">{s.title}</span>
                            {s.chunk_type === "abstract" && (
                              <span className="shrink-0 text-xs text-muted-foreground">
                                abstract
                              </span>
                            )}
                          </button>
                        </li>
                      ))}
                  </ul>
                </nav>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

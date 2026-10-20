"use client";

import { useMutation } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { postReview } from "@/lib/api/queries";
import type { CollectionOut } from "@/lib/api/queries";
import type { Source } from "@/lib/chat/events";
import {
  FOCUS_MAX_CHARS,
  reviewProblem,
  reviewRequest,
} from "@/lib/library/forms";
import { paperCount } from "@/lib/library/model";

import { DraftView } from "./draft-view";

// The literature review builder. PAID: Generate review makes one model call per group of
// eight papers above eight, plus one writing call. It reads the stored analyses and
// abstracts only (no search, no embeddings). Choosing a collection or typing a focus sends
// nothing; only the button does, and it is off below two papers.
export function ReviewPanel({
  collections,
  initial,
  activeChunkId,
  onOpen,
  onNewDraft,
}: {
  collections: readonly CollectionOut[];
  initial: string | null;
  activeChunkId: string | null;
  onOpen: (source: Source, all: readonly Source[]) => void;
  onNewDraft: () => void;
}) {
  const [chosen, setChosen] = useState<string | null>(null);
  const [focus, setFocus] = useState("");
  const collection =
    chosen !== null && collections.some((c) => c.name === chosen)
      ? chosen
      : initial;
  const picked = collections.find((c) => c.name === collection) ?? null;
  const count = picked?.paper_count ?? 0;
  const problem = reviewProblem(collection, count, focus);
  const calls = count > 8 ? Math.ceil(Math.min(count, 30) / 8) + 1 : 1;

  const review = useMutation({
    mutationFn: () => postReview(reviewRequest(collection ?? "", focus)),
    onMutate: onNewDraft,
  });

  return (
    <section aria-label="Literature review" className="space-y-4">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold tracking-tight">
          Literature review
        </h2>
        <p className="text-sm text-muted-foreground">
          Draft a related-work section from the papers of a collection. Every
          sentence about a paper carries a citation that opens the text it rests
          on. Nothing is sent until you press Generate review.
        </p>
      </div>
      <div className="grid gap-3 sm:grid-cols-[14rem_1fr]">
        <div className="space-y-1">
          <label htmlFor="review-collection" className="text-sm font-medium">
            Collection
          </label>
          <select
            id="review-collection"
            value={collection ?? ""}
            onChange={(event) => setChosen(event.target.value)}
            className="h-9 w-full rounded-md border border-input bg-background px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {collection === null && <option value="">Pick one&hellip;</option>}
            {collections.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({paperCount(c.paper_count)})
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-1">
          <label htmlFor="review-focus" className="text-sm font-medium">
            Focus (optional)
          </label>
          <Input
            id="review-focus"
            value={focus}
            maxLength={FOCUS_MAX_CHARS + 20}
            placeholder="for example: attention and word embeddings"
            onChange={(event) => setFocus(event.target.value)}
          />
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button
          type="button"
          disabled={problem !== null || review.isPending}
          onClick={() => review.mutate()}
        >
          Generate review
        </Button>
        <p className="text-sm text-muted-foreground">
          {problem ??
            `Makes ${calls} model call${calls === 1 ? "" : "s"} (a fraction of a cent each).`}
        </p>
      </div>
      {review.isPending && (
        <p role="status" className="text-sm text-muted-foreground">
          Writing the draft&hellip; this takes a few seconds.
        </p>
      )}
      {review.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-3 text-sm text-destructive"
        >
          {review.error.message}
        </p>
      )}
      {review.data && (
        <DraftView
          review={review.data}
          activeChunkId={activeChunkId}
          onOpen={onOpen}
        />
      )}
      {!review.data && !review.isPending && !review.isError && (
        <p className="text-sm text-muted-foreground">
          The draft appears here, built from generated analyses and abstracts,
          not from the papers&rsquo; full text.
        </p>
      )}
    </section>
  );
}

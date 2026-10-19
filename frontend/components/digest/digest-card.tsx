"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { actOnEntry } from "@/lib/api/queries";
import type { DigestOut, EntryAction, DigestEntryOut } from "@/lib/api/queries";
import {
  authorLine,
  statusAfter,
  statusLine,
  withEntry,
  withStatus,
} from "@/lib/digest/digest";
import { formatDate } from "@/lib/format";
import { arxivAbsUrl, paperPath } from "@/lib/links";
import { cn } from "@/lib/utils";

const LINK =
  "text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

// One paper of the digest. Save, Dismiss and Undo update the card at once (the cached
// digest is changed before the server answers) and put it back if the request fails.
// Each card has its own mutation, so a slow request on one never blocks another.
export function DigestCard({
  entry,
  digestKey,
  digestDate,
}: {
  entry: DigestEntryOut;
  digestKey: readonly unknown[];
  digestDate: string;
}) {
  const queryClient = useQueryClient();
  const act = useMutation({
    mutationFn: (action: EntryAction) =>
      actOnEntry(action, digestDate, entry.arxiv_id),
    onMutate: async (action) => {
      await queryClient.cancelQueries({ queryKey: digestKey });
      const before = queryClient.getQueryData<DigestOut>(digestKey);
      if (before) {
        queryClient.setQueryData<DigestOut>(
          digestKey,
          withStatus(before, entry.arxiv_id, statusAfter(action)),
        );
      }
      return { before };
    },
    onError: (_error, _action, context) => {
      if (context?.before) queryClient.setQueryData(digestKey, context.before);
    },
    onSuccess: (updated) => {
      queryClient.setQueryData<DigestOut>(digestKey, (current) =>
        current ? withEntry(current, updated) : current,
      );
    },
  });

  const abs = arxivAbsUrl(entry.abs_url);
  const line = statusLine(entry.status);
  const busy = act.isPending;
  return (
    <li>
      <article
        aria-labelledby={`title-${entry.arxiv_id}`}
        data-status={entry.status}
        className={cn(
          "rounded-xl border bg-card p-5 text-card-foreground shadow transition-opacity",
          entry.status === "dismissed" && "opacity-60",
          entry.status === "saved" && "border-primary/60",
        )}
      >
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span
            className="font-medium"
            title={`Relevance ${entry.score.toFixed(3)}`}
          >
            #{entry.rank}
          </span>
          <span>{formatDate(entry.published_at)}</span>
          {entry.categories.map((category) => (
            <Badge key={category} variant="outline" className="font-normal">
              {category}
            </Badge>
          ))}
          {entry.trending && (
            <Badge
              variant="secondary"
              title={`${entry.hf_upvotes} upvotes on Hugging Face`}
            >
              Trending
            </Badge>
          )}
          {entry.best_interest && (
            <Badge variant="outline" className="font-normal">
              Matches: {entry.best_interest.name}
            </Badge>
          )}
        </div>
        <h2
          id={`title-${entry.arxiv_id}`}
          className="mt-2 font-serif text-xl font-semibold leading-snug"
        >
          <Link href={paperPath(entry.arxiv_id)} className="hover:underline">
            {entry.title}
          </Link>
        </h2>
        {entry.authors.length > 0 && (
          <p className="mt-1 text-sm text-muted-foreground">
            {authorLine(entry.authors, entry.author_count)}
          </p>
        )}
        <p className="mt-3 text-sm leading-relaxed">{entry.summary}</p>
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex gap-2">
            {entry.status === "new" ? (
              <>
                <Button
                  size="sm"
                  disabled={busy}
                  onClick={() => act.mutate("save")}
                  aria-label={`Save: ${entry.title}`}
                >
                  Save
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={busy}
                  onClick={() => act.mutate("dismiss")}
                  aria-label={`Dismiss: ${entry.title}`}
                >
                  Dismiss
                </Button>
              </>
            ) : (
              <Button
                size="sm"
                variant="secondary"
                disabled={busy}
                onClick={() => act.mutate("undo")}
                aria-label={`Undo: ${entry.title}`}
              >
                Undo
              </Button>
            )}
          </div>
          <Link
            href={paperPath(entry.arxiv_id)}
            className={cn("text-sm", LINK)}
          >
            Open paper page
          </Link>
          {abs && (
            <a
              href={abs}
              target="_blank"
              rel="noopener noreferrer"
              className={cn("text-sm", LINK)}
            >
              arXiv {entry.arxiv_id}
            </a>
          )}
        </div>
        {line && (
          <p role="status" className="mt-3 text-sm text-muted-foreground">
            {line}
          </p>
        )}
        {act.isError && (
          <p role="alert" className="mt-3 text-sm text-destructive">
            That did not work and the card was put back: {act.error.message}
          </p>
        )}
      </article>
    </li>
  );
}

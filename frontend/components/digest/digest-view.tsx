"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";

import { DigestCard } from "@/components/digest/digest-card";
import { ErrorState } from "@/components/ui/error-state";
import { LoadingState } from "@/components/ui/loading-state";
import { fetchDigest } from "@/lib/api/queries";
import { formatDay, parseDay, statusCounts } from "@/lib/digest/digest";

// The home page: today's digest, ranked against the interests. Reading it only asks the
// API for stored rows (GET /digest never builds a digest and never calls OpenAI); the
// 08:00 job or `make digest` builds it. `?date=YYYY-MM-DD` picks an earlier day.
export function DigestView() {
  const params = useSearchParams();
  const router = useRouter();
  const date = parseDay(params.get("date")) ?? undefined;
  const digestKey = ["digest", date ?? "latest"] as const;
  const digest = useQuery({
    queryKey: digestKey,
    queryFn: () => fetchDigest(date),
  });

  if (digest.isPending) {
    return (
      <LoadingState
        label="Loading the digest…"
        className="mx-auto w-full max-w-3xl px-4 py-8"
      />
    );
  }
  if (digest.isError) {
    return (
      <div className="mx-auto w-full max-w-3xl space-y-3 px-4 py-8">
        <h1 className="text-2xl font-semibold tracking-tight">Digest</h1>
        <ErrorState
          what="The digest"
          error={digest.error}
          onRetry={() => void digest.refetch()}
        />
      </div>
    );
  }

  const d = digest.data;
  const counts = statusCounts(d.entries);
  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8">
      <header className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight">
          {d.digest_date ? formatDay(d.digest_date) : "Daily digest"}
        </h1>
        {d.digest_date && (
          <p className="text-sm text-muted-foreground">
            {d.entries.length} papers ranked against your interests
            {counts.saved > 0 ? `, ${counts.saved} saved` : ""}
            {counts.dismissed > 0 ? `, ${counts.dismissed} dismissed` : ""}.
          </p>
        )}
        {d.is_fallback && (
          <p role="status" className="text-sm text-muted-foreground">
            There is no digest for {formatDay(d.requested_date)}; this is the
            latest one.
          </p>
        )}
        {d.recent_dates.length > 1 && (
          <div className="flex items-center gap-2 text-sm">
            <label htmlFor="digest-day" className="text-muted-foreground">
              Earlier digests
            </label>
            <select
              id="digest-day"
              value={d.digest_date ?? ""}
              onChange={(e) => router.push(`/?date=${e.target.value}`)}
              className="h-8 rounded-md border border-input bg-background px-2 text-sm"
            >
              {d.recent_dates.map((day) => (
                <option key={day} value={day}>
                  {formatDay(day)}
                </option>
              ))}
            </select>
          </div>
        )}
      </header>

      {!d.has_interests && (
        <div
          role="note"
          className="rounded-lg border border-dashed p-4 text-sm"
        >
          <p className="font-medium">Add interests to rank this list.</p>
          <p className="mt-1 text-muted-foreground">
            Without interests the order only reflects Hugging Face upvotes.{" "}
            <Link
              href="/settings"
              className="text-primary underline-offset-4 hover:underline"
            >
              Add an interest
            </Link>
            .
          </p>
        </div>
      )}

      {d.digest_date === null ? (
        <div className="rounded-lg border p-6">
          <h2 className="font-semibold">No digest yet</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            The digest is built every morning at 08:00 from the papers of the
            last week, or by hand with <code>make digest</code>. When the daily
            job has not run (the stack is stopped), nothing is built.
          </p>
        </div>
      ) : d.entries.length === 0 ? (
        <div className="rounded-lg border p-6">
          <h2 className="font-semibold">Nothing new today</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            No new papers were left to rank: everything recent is in an earlier
            digest or was dismissed.
          </p>
        </div>
      ) : (
        <ol aria-label="Ranked papers" className="space-y-4">
          {d.entries.map((entry) => (
            <DigestCard
              key={entry.arxiv_id}
              entry={entry}
              digestKey={digestKey}
              digestDate={d.digest_date as string}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

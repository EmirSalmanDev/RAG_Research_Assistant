// Pure logic for the digest page: optimistic Save / Dismiss / Undo, the status line of a
// card, and dates. Plain TypeScript (type-only imports), so Node's test runner covers it.

import type {
  DigestEntryOut,
  DigestOut,
  EntryAction,
} from "../api/endpoints.ts";

export type EntryStatus = DigestEntryOut["status"];

/** The status an action leads to. */
export function statusAfter(action: EntryAction): EntryStatus {
  return action === "save"
    ? "saved"
    : action === "dismiss"
      ? "dismissed"
      : "new";
}

/**
 * The digest with one entry's status changed, as the page shows it before the server has
 * answered. The entry list and order are kept; an unknown id changes nothing.
 */
export function withStatus(
  digest: DigestOut,
  arxivId: string,
  status: EntryStatus,
): DigestOut {
  return {
    ...digest,
    entries: digest.entries.map((entry) =>
      entry.arxiv_id === arxivId ? { ...entry, status } : entry,
    ),
  };
}

/** The digest with one entry replaced by the server's version of it. */
export function withEntry(
  digest: DigestOut,
  updated: DigestEntryOut,
): DigestOut {
  return {
    ...digest,
    entries: digest.entries.map((entry) =>
      entry.arxiv_id === updated.arxiv_id ? updated : entry,
    ),
  };
}

export function statusCounts(
  entries: DigestEntryOut[],
): Record<EntryStatus, number> {
  const counts: Record<EntryStatus, number> = {
    new: 0,
    saved: 0,
    dismissed: 0,
  };
  for (const entry of entries) counts[entry.status] += 1;
  return counts;
}

/** The one-line status under a card's title; null while the entry is new. */
export function statusLine(status: EntryStatus): string | null {
  if (status === "saved") return "Saved to your library.";
  if (status === "dismissed") return "Dismissed. It will not come back.";
  return null;
}

const ISO_DAY = /^(\d{4})-(\d{2})-(\d{2})$/;

/** A YYYY-MM-DD date as a real calendar day, or null for anything else. */
export function parseDay(value: string | null | undefined): string | null {
  const m = ISO_DAY.exec(value ?? "");
  if (!m) return null;
  const [year, month, day] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year &&
    date.getUTCMonth() === month - 1 &&
    date.getUTCDate() === day
    ? value!
    : null;
}

/** "Mon, Oct 19, 2026" for a YYYY-MM-DD day; the input itself if it is not one. The day
 * is read as a calendar day (UTC), so no time zone shifts it. */
export function formatDay(day: string): string {
  if (parseDay(day) === null) return day;
  return new Date(`${day}T00:00:00Z`).toLocaleDateString("en", {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

/** "A, B, C" or "A, B and 4 more" from the card's first three authors and the total. */
export function authorLine(authors: string[], total: number): string {
  const more = total - authors.length;
  return more > 0
    ? `${authors.join(", ")} and ${more} more`
    : authors.join(", ");
}

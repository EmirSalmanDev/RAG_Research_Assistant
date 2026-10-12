// Small display helpers. Plain TypeScript.

/** "Sep 1, 2026" for an ISO timestamp; the input itself if it is not a date. */
export function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString("en", {
        year: "numeric",
        month: "short",
        day: "numeric",
        timeZone: "UTC",
      });
}

/**
 * The start of a passage for a list: at most `max` characters, cut at a word boundary,
 * with an ellipsis. Whitespace runs are collapsed. `truncated` says whether anything was cut.
 */
export function excerpt(
  text: string,
  max = 240,
): { text: string; truncated: boolean } {
  const flat = text.replace(/\s+/g, " ").trim();
  if (flat.length <= max) return { text: flat, truncated: false };
  const cut = flat.slice(0, max);
  const space = cut.lastIndexOf(" ");
  // Cut at the last space unless that would throw away most of the excerpt.
  const base = space > max * 0.6 ? cut.slice(0, space) : cut;
  return { text: `${base.replace(/[\s.,;:!?-]+$/, "")}…`, truncated: true };
}

const FULLTEXT_LABELS: Record<string, string> = {
  indexed: "Full text indexed",
  fetched: "Full text fetched, not indexed yet",
  requested: "Full text requested",
  none: "Abstract only",
  unavailable: "Full text not available",
  failed: "Full text could not be processed",
};

/** What a paper's `fulltext_status` means for the reader. Unknown values are shown as is. */
export function fulltextLabel(status: string): string {
  return FULLTEXT_LABELS[status] ?? status;
}

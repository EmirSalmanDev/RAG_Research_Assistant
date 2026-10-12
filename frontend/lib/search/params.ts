// The /search page keeps its query in the URL (?q=...&category=cs.CL&from=2026-09-01), so a
// search can be bookmarked and the back button works. This module reads and writes that
// string and checks it against the limits of GET /search. Plain TypeScript.

/** What the API's /search accepts for `q` (backend/app/api/search.py: max_length=1000). */
export const MAX_QUERY_CHARS = 1000;
/** Passages per search. One request, no paging. */
export const RESULT_LIMIT = 20;
const MAX_CATEGORIES = 8;

/**
 * Categories offered as filters: the backend's default ARXIV_CATEGORIES. The API takes any
 * category, but only these are fetched unless the setting was changed.
 */
export const KNOWN_CATEGORIES = ["cs.CL", "cs.LG", "cs.AI", "cs.IR"] as const;

export interface SearchParams {
  q: string;
  categories: string[];
  /** Inclusive start day, YYYY-MM-DD, or null. */
  from: string | null;
  /** Inclusive end day, YYYY-MM-DD, or null. */
  to: string | null;
}

export interface ParsedSearch {
  /** null when there is nothing (valid) to search for yet. */
  params: SearchParams | null;
  /** Reasons the URL was not accepted, in plain words. Empty when it was. */
  problems: string[];
}

const CATEGORY = /^[A-Za-z][A-Za-z-]*(\.[A-Za-z-]+)?$/;

function isRealDay(value: string): boolean {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!m) return false;
  const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
  const date = new Date(Date.UTC(y, mo - 1, d));
  return (
    date.getUTCFullYear() === y &&
    date.getUTCMonth() === mo - 1 &&
    date.getUTCDate() === d
  );
}

/** Check a search the reader typed (or a URL they opened). */
export function validateSearch(input: {
  q: string;
  categories: readonly string[];
  from: string | null;
  to: string | null;
}): ParsedSearch {
  const problems: string[] = [];
  const q = input.q.trim();
  if (q.length > MAX_QUERY_CHARS)
    problems.push(`The question is longer than ${MAX_QUERY_CHARS} characters.`);

  const categories = [
    ...new Set(input.categories.map((c) => c.trim()).filter(Boolean)),
  ];
  if (categories.some((c) => !CATEGORY.test(c)))
    problems.push("A category is not a valid arXiv category.");
  if (categories.length > MAX_CATEGORIES)
    problems.push(`Pick at most ${MAX_CATEGORIES} categories.`);

  const from = input.from?.trim() || null;
  const to = input.to?.trim() || null;
  if (from && !isRealDay(from))
    problems.push("The start date is not a valid date.");
  if (to && !isRealDay(to)) problems.push("The end date is not a valid date.");
  if (from && to && isRealDay(from) && isRealDay(to) && from > to) {
    problems.push("The start date is after the end date.");
  }

  if (problems.length > 0) return { params: null, problems };
  if (q === "") return { params: null, problems: [] };
  return { params: { q, categories, from, to }, problems: [] };
}

/** Read the page's query string. Unknown parameters are ignored. */
export function parseSearchParams(
  search: string | URLSearchParams,
): ParsedSearch {
  const sp = typeof search === "string" ? new URLSearchParams(search) : search;
  return validateSearch({
    q: sp.get("q") ?? "",
    categories: sp.getAll("category"),
    from: sp.get("from"),
    to: sp.get("to"),
  });
}

/** The query string for a search (no leading "?"). Empty filters are left out. */
export function toSearchString(params: SearchParams): string {
  const sp = new URLSearchParams();
  sp.set("q", params.q);
  for (const category of params.categories) sp.append("category", category);
  if (params.from) sp.set("from", params.from);
  if (params.to) sp.set("to", params.to);
  return sp.toString();
}

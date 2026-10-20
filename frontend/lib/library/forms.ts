// The rules the library's inputs follow before anything is sent, and when the paid button
// may be pressed. Plain TypeScript. Typing never starts a request: the only paid actions are
// the Generate review and Upload buttons, and they are off unless these say the input is valid.

import type { components } from "../api/schema";

/** The API's limits (backend/app/features/library.py and review.py). */
export const NAME_MAX_CHARS = 100;
export const DESCRIPTION_MAX_CHARS = 500;
export const NOTES_MAX_CHARS = 10_000;
export const REFERENCE_MAX_CHARS = 300;
export const FOCUS_MAX_CHARS = 200;
/** A review needs at least two papers. */
export const MIN_REVIEW_PAPERS = 2;
export const DEFAULT_REVIEW_PAPERS = 30;
export const MAX_REVIEW_PAPERS = 60;

export type ReviewRequest = components["schemas"]["ReviewRequest"];

/** Why a collection name is not acceptable (empty or too long); null when it is fine. */
export function nameProblem(raw: string): string | null {
  const name = raw.trim().replace(/\s+/g, " ");
  if (name === "") return "Give the collection a name.";
  if (name.length > NAME_MAX_CHARS)
    return `A name has at most ${NAME_MAX_CHARS} characters.`;
  return null;
}

/** Why the add-a-paper box is not ready (it takes an arXiv id or an arxiv.org link). */
export function referenceProblem(raw: string): string | null {
  const text = raw.trim();
  if (text === "") return "Type an arXiv id or an arxiv.org link.";
  if (text.length > REFERENCE_MAX_CHARS)
    return "That is too long for an arXiv id.";
  return null;
}

/** Why Generate review is off; null when it may be pressed. */
export function reviewProblem(
  collection: string | null,
  paperCount: number,
  focus: string,
): string | null {
  if (!collection) return "Pick a collection.";
  if (paperCount < MIN_REVIEW_PAPERS)
    return `A review needs at least ${MIN_REVIEW_PAPERS} papers (${collection} has ${paperCount}).`;
  if (focus.trim().length > FOCUS_MAX_CHARS)
    return `The focus has at most ${FOCUS_MAX_CHARS} characters.`;
  return null;
}

/** The body of POST /library/review: an empty focus is left out. */
export function reviewRequest(
  collection: string,
  focus: string,
  maxPapers: number = DEFAULT_REVIEW_PAPERS,
): ReviewRequest {
  const cleaned = focus.trim().replace(/\s+/g, " ");
  return {
    collection,
    focus: cleaned === "" ? null : cleaned,
    max_papers: Math.min(Math.max(Math.trunc(maxPapers), 1), MAX_REVIEW_PAPERS),
  };
}

/** The name in "Content-Disposition: attachment; filename=\"x.bib\"", or the fallback. */
export function filenameFromDisposition(
  header: string | null,
  fallback: string,
): string {
  const match = header ? /filename="([^"\\/]+)"/.exec(header) : null;
  return match?.[1] ?? fallback;
}

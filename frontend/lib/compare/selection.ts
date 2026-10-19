// Which papers and dimensions are picked on /compare, and when the Compare button may be
// pressed. Plain TypeScript. Nothing here calls the API: a comparison costs money, so it
// starts only from the button, and the button is off unless this says the choice is valid.

import type { components } from "../api/schema";

export type Dimension =
  components["schemas"]["CompareOut"]["dimensions"][number];

/** The API takes 2 to 5 papers (backend/app/features/compare.py). */
export const MIN_PAPERS = 2;
export const MAX_PAPERS = 5;

/** In the order the table shows them; the labels are the table's row headings. */
export const DIMENSIONS: readonly { id: Dimension; label: string }[] = [
  { id: "task", label: "Task" },
  { id: "method", label: "Method" },
  { id: "data", label: "Data" },
  { id: "metrics", label: "Metrics" },
  { id: "compute", label: "Compute" },
  { id: "limitations", label: "Limitations" },
];

export const ALL_DIMENSIONS: readonly Dimension[] = DIMENSIONS.map((d) => d.id);

export function dimensionLabel(id: string): string {
  return DIMENSIONS.find((d) => d.id === id)?.label ?? id;
}

/** What the picker keeps about a picked paper. */
export interface PickedPaper {
  arxiv_id: string;
  title: string;
  has_fulltext: boolean;
}

export interface ToggleResult {
  selected: PickedPaper[];
  /** Set when the click was refused: the list is already full. */
  refused: "full" | null;
}

/** Add the paper, or remove it if it is picked. Never more than MAX_PAPERS. */
export function togglePaper(
  selected: readonly PickedPaper[],
  paper: PickedPaper,
): ToggleResult {
  if (selected.some((p) => p.arxiv_id === paper.arxiv_id)) {
    return {
      selected: selected.filter((p) => p.arxiv_id !== paper.arxiv_id),
      refused: null,
    };
  }
  if (selected.length >= MAX_PAPERS) {
    return { selected: [...selected], refused: "full" };
  }
  return { selected: [...selected, paper], refused: null };
}

export function removePaper(
  selected: readonly PickedPaper[],
  arxivId: string,
): PickedPaper[] {
  return selected.filter((p) => p.arxiv_id !== arxivId);
}

/** Why Compare is off, in words for the reader; null when it may be pressed. */
export function compareProblem(
  selected: readonly PickedPaper[],
  dimensions: readonly Dimension[],
): string | null {
  if (selected.length < MIN_PAPERS)
    return `Pick at least ${MIN_PAPERS} papers (${selected.length} picked).`;
  if (selected.length > MAX_PAPERS) return `Pick at most ${MAX_PAPERS} papers.`;
  if (dimensions.length === 0) return "Pick at least one dimension.";
  return null;
}

/** Add or remove a dimension; the result keeps the table's order. */
export function toggleDimension(
  current: readonly Dimension[],
  id: Dimension,
): Dimension[] {
  const next = current.includes(id)
    ? current.filter((d) => d !== id)
    : [...current, id];
  return ALL_DIMENSIONS.filter((d) => next.includes(d));
}

/** The request body for POST /compare. */
export function compareRequest(
  selected: readonly PickedPaper[],
  dimensions: readonly Dimension[],
): { arxiv_ids: string[]; dimensions: Dimension[] } {
  return {
    arxiv_ids: selected.map((p) => p.arxiv_id),
    dimensions: [...dimensions],
  };
}

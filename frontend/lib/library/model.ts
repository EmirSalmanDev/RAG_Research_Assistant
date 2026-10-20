// What the library shows, taken from the API's types: statuses, evidence labels and the order
// of things. Plain TypeScript.

import type { components } from "../api/schema";

export type CollectionOut = components["schemas"]["CollectionOut"];
export type ItemOut = components["schemas"]["ItemOut"];
export type LibraryStatus = ItemOut["status"];
export type ItemEvidence = ItemOut["evidence"];
export type ReviewOut = components["schemas"]["ReviewOut"];
export type ReferenceOut = ReviewOut["references"][number];
export type ReferenceEvidence = ReferenceOut["evidence"];
export type UploadOut = components["schemas"]["UploadOut"];

/** The reading states, in the order a paper moves through them. */
export const STATUSES: readonly { id: LibraryStatus; label: string }[] = [
  { id: "to_read", label: "To read" },
  { id: "reading", label: "Reading" },
  { id: "read", label: "Read" },
  { id: "archived", label: "Archived" },
];

export function statusLabel(id: string): string {
  return STATUSES.find((s) => s.id === id)?.label ?? id;
}

/** The badge of an item in the table: how much of the paper is indexed. */
export const ITEM_EVIDENCE_LABEL: Record<ItemEvidence, string> = {
  full_text: "full text",
  abstract_only: "abstract only",
  uploaded: "uploaded",
};

/**
 * The label of a citation in a review. The same words as in chat and in the comparison
 * table: a review rests on generated analyses and abstracts, never on full text.
 */
export const REFERENCE_EVIDENCE_LABEL: Record<ReferenceEvidence, string> = {
  analysis: "generated analysis",
  abstract: "abstract only",
};

/** Collections by name, case-insensitively, as the server lists them. */
export function sortCollections(
  collections: readonly CollectionOut[],
): CollectionOut[] {
  return [...collections].sort((a, b) =>
    a.name.localeCompare(b.name, "en", { sensitivity: "base" }),
  );
}

/** "1 paper", "2 papers". */
export function paperCount(n: number): string {
  return `${n} ${n === 1 ? "paper" : "papers"}`;
}

// Export of a related-work draft as markdown, in the browser (no server round trip, no
// cost). The draft keeps its [n] markers; a References list maps each cited number to the
// BibTeX key (the same key as in the collection's .bib file) and to the paper's arXiv
// ABSTRACT page. The app never re-serves a PDF, so neither does the export. Plain TypeScript.

import { arxivAbsUrl } from "../links.ts";
import type { ReferenceOut, ReviewOut } from "./model.ts";
import { REFERENCE_EVIDENCE_LABEL } from "./model.ts";

/** What the draft rests on, in one sentence. Shown on the page and put into the export. */
export const EVIDENCE_NOTE =
  "Built from model-generated analyses and abstracts of the papers, not from their full text. A citation that opens does not prove it supports the sentence: check each claim against the paper.";

function oneLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function markers(numbers: readonly number[]): string {
  return numbers.map((n) => `[${n}]`).join("");
}

/** The numbers cited anywhere in the draft, ascending. */
export function citedNumbers(review: ReviewOut): number[] {
  const seen = new Set<number>();
  for (const section of review.sections)
    for (const paragraph of section.paragraphs)
      for (const sentence of paragraph.sentences)
        for (const n of sentence.source_numbers) seen.add(n);
  return [...seen].sort((a, b) => a - b);
}

function referenceLine(ref: ReferenceOut): string {
  const url = arxivAbsUrl(ref.abs_url);
  const where = url ?? "uploaded document, no arXiv page";
  const evidence = REFERENCE_EVIDENCE_LABEL[ref.evidence];
  return `[${ref.n}] \`${ref.bibtex_key}\`: ${oneLine(ref.title)} (${ref.year}), ${evidence}. ${where}`;
}

/** The draft as markdown. Deterministic: the same review gives the same text. */
export function reviewToMarkdown(review: ReviewOut): string {
  const lines = [`# Related work: ${oneLine(review.collection)}`, ""];
  if (review.focus) lines.push(`Focus: ${oneLine(review.focus)}`, "");
  lines.push(`_${EVIDENCE_NOTE}_`, "");
  for (const section of review.sections) {
    lines.push(`## ${oneLine(section.heading)}`, "");
    for (const paragraph of section.paragraphs) {
      lines.push(
        paragraph.sentences
          .map((s) => `${oneLine(s.text)}${markers(s.source_numbers)}`)
          .join(" "),
        "",
      );
    }
  }
  if (review.sections.length === 0)
    lines.push("No sentence survived the checks.", "");
  const cited = new Set(citedNumbers(review));
  const refs = review.references
    .filter((r) => cited.has(r.n))
    .sort((a, b) => a.n - b.n);
  lines.push("## References", "");
  if (refs.length === 0) lines.push("No reference is cited.");
  for (const ref of refs) lines.push(referenceLine(ref));
  lines.push(
    "",
    `Written by a language model (${review.prompt_versions.write}) from ${review.counts.papers} papers; ${review.counts.dropped_uncited} sentence${review.counts.dropped_uncited === 1 ? "" : "s"} removed for lack of a valid citation. The BibTeX keys match the collection's .bib file.`,
    "",
  );
  return lines.join("\n");
}

/** A file name for the download, safe for any file system. */
export function reviewFileName(review: ReviewOut): string {
  const slug = review.collection
    .replace(/ı/g, "i") // the dotless i does not decompose
    .normalize("NFKD")
    .replace(/[^\x00-\x7f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return `related-work-${slug || "library"}.md`;
}

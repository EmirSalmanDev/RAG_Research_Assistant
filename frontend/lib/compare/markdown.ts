// Export of a comparison as markdown, in the browser (no server round trip, no cost).
// A table with the papers as columns and the dimensions as rows, [n] markers in the cells,
// and a Sources list that links each cited source to its arXiv ABSTRACT page. The app never
// re-serves a PDF, so neither does the export. Plain TypeScript.

import type { components } from "../api/schema";
import { arxivAbsUrl } from "../links.ts";
import { dimensionLabel } from "./selection.ts";

export type CompareOut = components["schemas"]["CompareOut"];
type Source = CompareOut["sources"][number];

/** A table cell: no pipes or line breaks that would end the cell early. */
function cell(text: string): string {
  return text.replace(/\s+/g, " ").replace(/\|/g, "\\|").trim();
}

function markers(numbers: readonly number[]): string {
  return numbers.map((n) => `[${n}]`).join("");
}

function sourceLine(source: Source): string {
  const where =
    source.chunk_type === "abstract"
      ? "abstract"
      : (source.section_path ?? "full text");
  const url = arxivAbsUrl(source.abs_url);
  const title = source.title.replace(/\s+/g, " ").trim();
  const text = `${source.n}. ${title} (arXiv:${source.arxiv_id}), ${where}`;
  return url ? `${text}. ${url}` : text;
}

/** The comparison as markdown. Deterministic: the same result gives the same text. */
export function compareToMarkdown(result: CompareOut): string {
  const header = [
    "Dimension",
    ...result.rows.map(
      (row) =>
        `${cell(row.title)} (arXiv:${row.arxiv_id}${row.has_fulltext ? "" : ", abstract only"})`,
    ),
  ];
  const lines = [
    "# Paper comparison",
    "",
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
  ];
  result.dimensions.forEach((dimension, i) => {
    const cells = result.rows.map((row) => {
      const c = row.cells[i];
      if (!c || c.status === "not_stated")
        return "_Not stated in the sources._";
      return `${cell(c.text)} ${markers(c.citations.map((x) => x.n))}`.trim();
    });
    lines.push(`| ${[dimensionLabel(dimension), ...cells].join(" | ")} |`);
  });

  const cited = new Set(
    result.rows.flatMap((row) =>
      row.cells.flatMap((c) => c.citations.map((x) => x.n)),
    ),
  );
  const sources = result.sources
    .filter((s) => cited.has(s.n))
    .sort((a, b) => a.n - b.n);
  lines.push("", "## Sources", "");
  if (sources.length === 0) lines.push("No cell is supported by a source.");
  for (const source of sources) lines.push(sourceLine(source));
  const abstractOnly = result.rows.filter((r) => !r.has_fulltext);
  if (abstractOnly.length > 0) {
    lines.push(
      "",
      `Abstract only (no full text indexed): ${abstractOnly.map((r) => `arXiv:${r.arxiv_id}`).join(", ")}.`,
    );
  }
  lines.push(
    "",
    `Cells are filled by a language model from the numbered sources only; a cell without a source says "Not stated in the sources". Prompt ${result.prompt_version}.`,
    "",
  );
  return lines.join("\n");
}

/** A file name for the download: the paper ids, safe for any file system. */
export function markdownFileName(result: CompareOut): string {
  const ids = result.rows.map((r) =>
    r.arxiv_id.replace(/[^0-9A-Za-z.-]/g, "-"),
  );
  return `comparison-${ids.join("_")}.md`;
}

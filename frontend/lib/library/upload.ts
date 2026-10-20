// The PDF upload card's checks and wording. Plain TypeScript. An upload is paid (embeddings),
// so the button is off unless these pass; the size is checked here before a byte is sent,
// and again by the server.

/** The server's default limit (UPLOAD_MAX_BYTES); the server has the final say. */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
export const TITLE_MAX_CHARS = 300;
/** What the card says about the cost: embeddings only, about this much per 10 pages. */
export const UPLOAD_COST_NOTE = "about $0.0002 per 10 pages of embeddings";

export interface PickedFile {
  name: string;
  size: number;
  type: string;
}

/** "812 B", "1.4 MB". */
export function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10 * 1024 ? 1 : 0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** A starting title from the file name: "my_paper-v2.pdf" -> "my paper v2". */
export function titleFromFileName(name: string): string {
  return name
    .replace(/\.pdf$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Why Upload is off; null when the file and the title are fine. */
export function uploadProblem(
  file: PickedFile | null,
  title: string,
  maxBytes: number = MAX_UPLOAD_BYTES,
): string | null {
  if (!file) return "Choose a PDF file.";
  const looksLikePdf =
    file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  if (!looksLikePdf) return "Only PDF files can be uploaded.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > maxBytes)
    return `That file is ${formatBytes(file.size)}; the limit is ${formatBytes(maxBytes)}.`;
  const clean = title.trim();
  if (clean === "") return "The title is required.";
  if (clean.length > TITLE_MAX_CHARS)
    return `The title has at most ${TITLE_MAX_CHARS} characters.`;
  return null;
}

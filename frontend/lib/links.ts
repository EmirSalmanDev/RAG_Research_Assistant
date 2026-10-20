// Links come from the API and, behind it, from third-party data (arXiv, Hugging Face,
// GitHub), so none is rendered as a link without a check. Plain TypeScript.

const ARXIV_ORIGIN = "https://arxiv.org";

/**
 * An arXiv abstract page URL, or null. Only https://arxiv.org/abs/... qualifies: not a PDF
 * (the app links to arXiv and never re-serves papers), not another host that merely starts
 * with the same text, not a URL with credentials.
 */
export function arxivAbsUrl(url: string | null | undefined): string | null {
  if (!url?.startsWith(`${ARXIV_ORIGIN}/abs/`)) return null;
  try {
    const parsed = new URL(url);
    if (parsed.origin !== ARXIV_ORIGIN || parsed.username || parsed.password)
      return null;
    if (!parsed.pathname.startsWith("/abs/") || parsed.pathname === "/abs/")
      return null;
  } catch {
    return null;
  }
  return url;
}

/** An http(s) URL for a project or code link, or null (never javascript:, data:, ...). */
export function safeHttpUrl(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:" && parsed.protocol !== "http:")
      return null;
    if (parsed.username || parsed.password) return null;
    return parsed.href;
  } catch {
    return null;
  }
}

/**
 * A link to a code or model page, or null: only https URLs on the given site (exact
 * origin, no credentials) qualify. Used for GitHub repos and Hugging Face artifacts.
 */
export function siteUrl(
  url: string | null | undefined,
  origin: "https://github.com" | "https://huggingface.co",
): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (parsed.origin !== origin || parsed.username || parsed.password)
      return null;
    if (parsed.pathname === "/" || parsed.pathname === "") return null;
    return parsed.href;
  } catch {
    return null;
  }
}

/** The id of an uploaded document (a PDF that is not on arXiv): it has no arXiv page. */
export function isLocalId(arxivId: string): boolean {
  return arxivId.startsWith("local-");
}

/** The page of one paper in this app. */
export function paperPath(arxivId: string): string {
  return `/paper/${encodeURIComponent(arxivId)}`;
}

/** A dynamic route segment, decoded once; a malformed escape is kept as written. */
export function decodeRouteParam(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

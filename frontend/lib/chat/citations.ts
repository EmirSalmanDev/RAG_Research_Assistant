// Turns answer text into plain text and citation markers.
//
// The model cites as [3], or several at once as [1, 2] / [1; 2]; this is the same pattern
// the backend uses to find them (backend/app/rag/citations.py). A number that is one of
// the answer's sources becomes a citation segment (a chip in the UI). A number with no
// source behind it stays plain text, so the reader is never offered a link to nothing.
//
// Plain TypeScript; the only import is a type, so Node's test runner runs it as is.

import type { Source } from "./events.ts";

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; n: number; source: Source };

const CITATION = /\[(\d+(?:\s*[,;]\s*\d+)*)\]/g;

// While the answer is still streaming, "[" or "[1," at the very end is the start of a
// marker that has not arrived in full. Showing it as text and then swapping it for a
// chip a moment later would make the line jump, so it is held back until it completes.
const INCOMPLETE_MARKER_AT_END = /\[[\d\s,;]*$/;

function pushText(out: Segment[], text: string): void {
  if (text === "") return;
  const last = out[out.length - 1];
  if (last?.kind === "text") last.text += text;
  else out.push({ kind: "text", text });
}

export function splitCitations(
  text: string,
  sources: readonly Source[],
  options: { streaming?: boolean } = {},
): Segment[] {
  const byNumber = new Map<number, Source>();
  for (const s of sources) byNumber.set(s.n, s);

  let body = text;
  if (options.streaming) body = body.replace(INCOMPLETE_MARKER_AT_END, "");

  const out: Segment[] = [];
  let last = 0;
  for (const match of body.matchAll(CITATION)) {
    const index = match.index ?? 0;
    pushText(out, body.slice(last, index));
    last = index + match[0].length;

    const numbers = (match[1] ?? "")
      .split(/[,;]/)
      .map((part) => Number(part.trim()));
    if (!numbers.some((n) => byNumber.has(n))) {
      pushText(out, match[0]); // nothing behind it: leave the marker exactly as written
      continue;
    }
    numbers.forEach((n, k) => {
      if (k > 0) pushText(out, " ");
      const source = byNumber.get(n);
      if (source) out.push({ kind: "citation", n, source });
      else pushText(out, `[${n}]`);
    });
  }
  pushText(out, body.slice(last));
  return out;
}

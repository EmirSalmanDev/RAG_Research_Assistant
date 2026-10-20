// A review's reference in the shape the sources side panel (the chat's) expects, so a citation
// chip in a draft opens the existing panel with the cited chunk. Plain TypeScript.

import type { Source } from "../chat/events.ts";
import type { ReferenceOut } from "./model.ts";

export function referenceToSource(ref: ReferenceOut): Source {
  return {
    n: ref.n,
    chunk_id: ref.chunk_id,
    arxiv_id: ref.arxiv_id,
    title: ref.title,
    section_path: ref.section_path ?? null,
    chunk_type: ref.chunk_type,
    published_at: ref.published_at,
    abs_url: ref.abs_url ?? null,
  };
}

/** All the references as panel sources, in number order. */
export function referenceSources(refs: readonly ReferenceOut[]): Source[] {
  return [...refs].sort((a, b) => a.n - b.n).map(referenceToSource);
}

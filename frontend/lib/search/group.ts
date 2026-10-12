// /search returns passages (chunks), best first. The page shows them grouped by paper.
// Plain TypeScript.

/** The fields of a search hit this module needs (a subset of the API's SearchHitOut). */
export interface HitLike {
  arxiv_id: string;
  title: string;
  published_at: string;
}

export interface PaperGroup<H extends HitLike> {
  arxivId: string;
  title: string;
  publishedAt: string;
  /** This paper's hits, in the order the API ranked them. */
  hits: H[];
}

/**
 * Group hits by paper. Papers are ordered by their best-ranked hit, and each paper keeps
 * its hits in rank order, so the first thing on the page is still the best match.
 */
export function groupByPaper<H extends HitLike>(
  hits: readonly H[],
): PaperGroup<H>[] {
  const groups = new Map<string, PaperGroup<H>>();
  for (const hit of hits) {
    const group = groups.get(hit.arxiv_id);
    if (group) group.hits.push(hit);
    else {
      groups.set(hit.arxiv_id, {
        arxivId: hit.arxiv_id,
        title: hit.title,
        publishedAt: hit.published_at,
        hits: [hit],
      });
    }
  }
  return [...groups.values()];
}

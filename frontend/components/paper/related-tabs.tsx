"use client";

import Link from "next/link";
import { type KeyboardEvent, useRef, useState } from "react";

import type { RelatedOut, RelatedPaperOut } from "@/lib/api/queries";
import { formatDate } from "@/lib/format";
import { arxivAbsUrl, paperPath } from "@/lib/links";
import {
  RELATED_TAB_LABELS,
  RELATED_TABS,
  type RelatedTab,
  tabAfterKey,
} from "@/lib/paper/tabs";

const LINK =
  "text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

const EMPTY: Record<RelatedTab, string> = {
  similar: "No similar paper was found in the local index.",
  cites: "No references with an arXiv id are known for this paper.",
  cited_by: "No paper in the local index cites this paper yet.",
};

const SIMILAR_STATUS: Record<string, string> = {
  not_indexed:
    "This paper's abstract is not indexed yet, so similar papers cannot be found.",
  unavailable:
    "Similar papers could not be loaded: the vector index did not answer.",
};

// "Similar", "Cites" and "Cited by" (GET /papers/{id}/related) as WAI-ARIA tabs: one tab
// stop, arrow keys move between tabs, Home and End jump to the first and last.
export function RelatedTabs({ related }: { related: RelatedOut }) {
  const [tab, setTab] = useState<RelatedTab>("similar");
  const buttons = useRef<Partial<Record<RelatedTab, HTMLButtonElement>>>({});
  const lists: Record<RelatedTab, RelatedPaperOut[]> = {
    similar: related.similar,
    cites: related.cites,
    cited_by: related.cited_by,
  };

  function onKeyDown(event: KeyboardEvent) {
    const next = tabAfterKey(tab, event.key);
    if (next === null) return;
    event.preventDefault();
    setTab(next);
    buttons.current[next]?.focus();
  }

  const statusNote =
    tab === "similar" ? SIMILAR_STATUS[related.similar_status] : undefined;
  const items = lists[tab];

  return (
    <section aria-labelledby="related-heading" className="space-y-3">
      <h2 id="related-heading" className="text-lg font-semibold">
        Related papers
      </h2>
      <div
        role="tablist"
        aria-label="Related papers"
        onKeyDown={onKeyDown}
        className="flex gap-1 border-b"
      >
        {RELATED_TABS.map((t) => (
          <button
            key={t}
            ref={(el) => {
              if (el) buttons.current[t] = el;
            }}
            type="button"
            role="tab"
            id={`related-tab-${t}`}
            aria-selected={tab === t}
            aria-controls={`related-panel-${t}`}
            tabIndex={tab === t ? 0 : -1}
            onClick={() => setTab(t)}
            className={`-mb-px border-b-2 px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
              tab === t
                ? "border-primary font-medium text-foreground"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {RELATED_TAB_LABELS[t]} ({lists[t].length})
          </button>
        ))}
      </div>
      <div
        role="tabpanel"
        id={`related-panel-${tab}`}
        aria-labelledby={`related-tab-${tab}`}
        tabIndex={0}
        className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {statusNote ? (
          <p className="text-muted-foreground">{statusNote}</p>
        ) : items.length === 0 ? (
          <p className="text-muted-foreground">{EMPTY[tab]}</p>
        ) : (
          <ul className="space-y-2">
            {items.map((p) => (
              <li key={p.arxiv_id}>
                <RelatedItem paper={p} />
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

function RelatedItem({ paper }: { paper: RelatedPaperOut }) {
  const abs = arxivAbsUrl(paper.abs_url);
  if (paper.held && paper.title) {
    return (
      <div>
        <Link
          href={paperPath(paper.arxiv_id)}
          className={`font-medium ${LINK}`}
        >
          {paper.title}
        </Link>
        <p className="text-sm text-muted-foreground">
          arXiv {paper.arxiv_id}
          {paper.published_at ? `, ${formatDate(paper.published_at)}` : ""}
        </p>
      </div>
    );
  }
  // A referenced paper that is not in the local index: only its arXiv id is known.
  return (
    <p>
      {abs ? (
        <a
          href={abs}
          target="_blank"
          rel="noopener noreferrer"
          className={LINK}
        >
          arXiv {paper.arxiv_id}
        </a>
      ) : (
        <span>arXiv {paper.arxiv_id}</span>
      )}{" "}
      <span className="text-sm text-muted-foreground">
        (not in the local index)
      </span>
    </p>
  );
}

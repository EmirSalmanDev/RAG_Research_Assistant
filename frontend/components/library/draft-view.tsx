"use client";

import Link from "next/link";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import type { Source } from "@/lib/chat/events";
import {
  EVIDENCE_NOTE,
  citedNumbers,
  reviewFileName,
  reviewToMarkdown,
} from "@/lib/library/markdown";
import { REFERENCE_EVIDENCE_LABEL } from "@/lib/library/model";
import type { ReferenceOut, ReviewOut } from "@/lib/library/model";
import { referenceSources, referenceToSource } from "@/lib/library/panel";
import { arxivAbsUrl, paperPath } from "@/lib/links";
import { cn } from "@/lib/utils";

import { copyText, downloadText } from "./download";

// A citation in the draft: the source number and, in words, how strong its evidence is.
// A dashed border and the word "abstract" mean the paper's abstract only; "analysis" means a
// model-generated analysis. Neither is the paper's full text.
function ReviewChip({
  reference,
  active,
  onOpen,
}: {
  reference: ReferenceOut;
  active: boolean;
  onOpen: (reference: ReferenceOut) => void;
}) {
  const label = REFERENCE_EVIDENCE_LABEL[reference.evidence];
  return (
    <button
      type="button"
      onClick={() => onOpen(reference)}
      aria-label={`Source ${reference.n}: ${reference.title} (${label})`}
      aria-haspopup="dialog"
      className={cn(
        "mx-0.5 inline-flex items-baseline gap-1 rounded border px-1.5 font-sans text-xs font-medium leading-5 align-baseline transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        reference.evidence === "abstract"
          ? "border-dashed border-primary/60"
          : "border-primary/40",
        active
          ? "bg-primary text-primary-foreground"
          : "bg-accent text-accent-foreground hover:bg-primary hover:text-primary-foreground",
      )}
    >
      <span>{reference.n}</span>
      <span className="text-[10px] font-normal">
        {reference.evidence === "abstract" ? "abstract" : "analysis"}
      </span>
    </button>
  );
}

function ReferenceItem({ reference }: { reference: ReferenceOut }) {
  const abs = arxivAbsUrl(reference.abs_url);
  return (
    <li
      className="space-y-0.5 text-sm"
      data-testid={`reference-${reference.n}`}
    >
      <p>
        <span className="font-medium">[{reference.n}]</span>{" "}
        <code className="rounded bg-muted px-1 text-xs">
          {reference.bibtex_key}
        </code>{" "}
        {reference.title} ({reference.year})
      </p>
      <p className="text-xs text-muted-foreground">
        {REFERENCE_EVIDENCE_LABEL[reference.evidence]}
        {" · "}
        {abs ? (
          <a
            href={abs}
            target="_blank"
            rel="noopener noreferrer"
            className="underline underline-offset-2"
          >
            arXiv abstract page
          </a>
        ) : (
          "uploaded document, no arXiv page"
        )}
        {" · "}
        <Link
          href={paperPath(reference.arxiv_id)}
          className="underline underline-offset-2"
        >
          paper page
        </Link>
      </p>
    </li>
  );
}

function ExportBar({ review }: { review: ReviewOut }) {
  const [copied, setCopied] = useState<"yes" | "failed" | null>(null);
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        type="button"
        variant="outline"
        onClick={() =>
          downloadText(
            reviewFileName(review),
            reviewToMarkdown(review),
            "text/markdown;charset=utf-8",
          )
        }
      >
        Download as markdown
      </Button>
      <Button
        type="button"
        variant="outline"
        onClick={async () =>
          setCopied(
            (await copyText(reviewToMarkdown(review))) ? "yes" : "failed",
          )
        }
      >
        Copy markdown
      </Button>
      <span role="status" className="text-sm text-muted-foreground">
        {copied === "yes" && "Copied to the clipboard."}
        {copied === "failed" && "Could not copy. Use the download instead."}
      </span>
    </div>
  );
}

// The draft: sections of sentences, each cited sentence followed by its chips. Only cited
// claims are shown (the server removes the rest and says how many), and the page says
// plainly what the draft rests on.
export function DraftView({
  review,
  activeChunkId,
  onOpen,
}: {
  review: ReviewOut;
  activeChunkId: string | null;
  onOpen: (source: Source, all: readonly Source[]) => void;
}) {
  const byNumber = new Map(review.references.map((r) => [r.n, r]));
  const all = referenceSources(review.references);
  const open = (reference: ReferenceOut) =>
    onOpen(referenceToSource(reference), all);
  const cited = new Set(citedNumbers(review));
  const citedRefs = review.references.filter((r) => cited.has(r.n));
  const otherRefs = review.references.filter((r) => !cited.has(r.n));
  const { counts } = review;
  const calls = counts.map_calls + 1;

  return (
    <section aria-label="Related-work draft" className="space-y-5">
      <div
        className="space-y-1 rounded-md border bg-muted/40 p-3 text-sm"
        data-testid="evidence-note"
      >
        <p>{EVIDENCE_NOTE}</p>
        <p className="text-muted-foreground">
          {counts.papers} papers used, {counts.papers_without_analysis} of them
          with an abstract only (no generated analysis yet).
          {counts.papers_omitted > 0 &&
            ` ${counts.papers_omitted} more were left out by the paper limit.`}
          {counts.papers_uncitable > 0 &&
            ` ${counts.papers_uncitable} have nothing indexed to cite.`}
        </p>
      </div>

      {review.sections.length === 0 ? (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-3 text-sm text-destructive"
        >
          No sentence of the draft had a valid citation, so none is shown. Try
          again, or add papers with an analysis.
        </p>
      ) : (
        <div className="space-y-6">
          {review.sections.map((section, i) => (
            <div key={`${section.heading}-${i}`} className="space-y-3">
              <h3 className="text-lg font-semibold tracking-tight">
                {section.heading}
              </h3>
              {section.paragraphs.map((paragraph, j) => (
                <p key={j} className="font-serif text-base leading-7">
                  {paragraph.sentences.map((sentence, k) => (
                    <span key={k}>
                      {sentence.text}
                      {sentence.source_numbers.map((n) => {
                        const reference = byNumber.get(n);
                        return reference ? (
                          <ReviewChip
                            key={n}
                            reference={reference}
                            active={reference.chunk_id === activeChunkId}
                            onOpen={open}
                          />
                        ) : null;
                      })}{" "}
                    </span>
                  ))}
                </p>
              ))}
            </div>
          ))}
        </div>
      )}

      <p className="text-sm text-muted-foreground" data-testid="removed-note">
        {counts.dropped_uncited === 0
          ? "No claim had to be removed."
          : `${counts.dropped_uncited} claim${counts.dropped_uncited === 1 ? "" : "s"} removed because ${counts.dropped_uncited === 1 ? "it" : "they"} had no valid citation.`}
        {counts.invalid_citations_dropped > 0 &&
          ` ${counts.invalid_citations_dropped} citation number${counts.invalid_citations_dropped === 1 ? "" : "s"} that did not exist ${counts.invalid_citations_dropped === 1 ? "was" : "were"} dropped.`}
      </p>

      <div className="space-y-2">
        <h3 className="text-sm font-semibold">References</h3>
        {citedRefs.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No reference is cited.
          </p>
        ) : (
          <ul className="space-y-2">
            {citedRefs.map((r) => (
              <ReferenceItem key={r.n} reference={r} />
            ))}
          </ul>
        )}
        <p className="text-xs text-muted-foreground">
          The keys are the ones in the collection&rsquo;s .bib file.
        </p>
        {otherRefs.length > 0 && (
          <details className="text-sm">
            <summary className="cursor-pointer text-muted-foreground">
              {otherRefs.length} other paper
              {otherRefs.length === 1 ? "" : "s"} used but not cited
            </summary>
            <ul className="mt-2 space-y-2">
              {otherRefs.map((r) => (
                <ReferenceItem key={r.n} reference={r} />
              ))}
            </ul>
          </details>
        )}
      </div>

      <ExportBar review={review} />
      <p className="text-xs text-muted-foreground">
        {calls} model call{calls === 1 ? "" : "s"} (
        {Object.values(review.prompt_versions).join(", ")}): {review.tokens_in}{" "}
        tokens in, {review.tokens_out} out, about ${review.cost_usd.toFixed(4)}.
      </p>
    </section>
  );
}

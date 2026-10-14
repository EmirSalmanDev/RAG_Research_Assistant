"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import type { ReactNode } from "react";

import { AnalysisSection } from "@/components/paper/analysis-section";
import { RelatedTabs } from "@/components/paper/related-tabs";
import { ResourcesSection } from "@/components/paper/resources-section";
import { Passage } from "@/components/passage";
import { Badge } from "@/components/ui/badge";
import {
  fetchPaper,
  fetchPaperChunks,
  fetchPaperDetails,
  fetchPaperRelated,
} from "@/lib/api/queries";
import { ApiError } from "@/lib/api/errors";
import { formatDate, fulltextLabel } from "@/lib/format";
import { arxivAbsUrl, safeHttpUrl } from "@/lib/links";

const MAX_AUTHORS = 12;
const EXTERNAL_LINK =
  "text-primary underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

// One paper: metadata and abstract (GET /papers/{id}), its analysis, code and Hugging
// Face artifacts (GET /papers/{id}/details), related papers (GET /papers/{id}/related)
// and its passages (GET /papers/{id}/chunks). None of them calls OpenAI: opening a page
// costs nothing. "Similar" papers come from Qdrant, the rest from Postgres.
export function PaperView({ id }: { id: string }) {
  const paper = useQuery({
    queryKey: ["paper", id],
    queryFn: () => fetchPaper(id),
  });
  const chunks = useQuery({
    queryKey: ["paper-chunks", id],
    queryFn: () => fetchPaperChunks(id),
  });
  const details = useQuery({
    queryKey: ["paper-details", id],
    queryFn: () => fetchPaperDetails(id),
  });
  const related = useQuery({
    queryKey: ["paper-related", id],
    queryFn: () => fetchPaperRelated(id),
  });

  if (paper.isPending) {
    return (
      <p
        role="status"
        className="mx-auto w-full max-w-3xl px-4 py-8 text-muted-foreground"
      >
        Loading the paper&hellip;
      </p>
    );
  }
  if (paper.isError) {
    const notFound =
      paper.error instanceof ApiError && paper.error.status === 404;
    return (
      <div className="mx-auto w-full max-w-3xl space-y-3 px-4 py-8">
        <h1 className="text-2xl font-semibold tracking-tight">
          {notFound ? "Paper not found" : "The paper could not be loaded"}
        </h1>
        <p role="alert" className="text-muted-foreground">
          {notFound
            ? `arXiv ${id} is not in the local index. Only papers that were ingested have a page.`
            : paper.error.message}
        </p>
        <Link href="/search" className={EXTERNAL_LINK}>
          Back to search
        </Link>
      </div>
    );
  }

  const p = paper.data;
  const abs = arxivAbsUrl(p.abs_url);
  const github = safeHttpUrl(p.github_url);
  const project = safeHttpUrl(p.project_url);
  const shownAuthors = p.authors.slice(0, MAX_AUTHORS);
  const moreAuthors = p.authors.length - shownAuthors.length;

  return (
    <article className="mx-auto w-full max-w-3xl space-y-8 px-4 py-8">
      <header className="space-y-3">
        <Link href="/search" className={`text-sm ${EXTERNAL_LINK}`}>
          Back to search
        </Link>
        <h1 className="font-serif text-3xl font-semibold leading-tight tracking-tight">
          {p.title}
        </h1>
        {shownAuthors.length > 0 && (
          <p className="text-muted-foreground">
            {shownAuthors.join(", ")}
            {moreAuthors > 0 ? ` and ${moreAuthors} more` : ""}
          </p>
        )}
        <p className="text-sm text-muted-foreground">
          arXiv {p.arxiv_id}, version {p.version}, published{" "}
          {formatDate(p.published_at)}
        </p>
        {p.categories.length > 0 && (
          <ul aria-label="Categories" className="flex flex-wrap gap-1.5">
            {p.categories.map((category) => (
              <li key={category}>
                <Badge
                  variant={
                    category === p.primary_category ? "secondary" : "outline"
                  }
                  className="font-normal"
                >
                  {category}
                </Badge>
              </li>
            ))}
          </ul>
        )}
        <ul className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
          {abs && (
            <li>
              <a
                href={abs}
                target="_blank"
                rel="noopener noreferrer"
                className={EXTERNAL_LINK}
              >
                Read the abstract on arXiv
              </a>
            </li>
          )}
          {github && (
            <li>
              <a
                href={github}
                target="_blank"
                rel="noopener noreferrer nofollow"
                className={EXTERNAL_LINK}
              >
                Code
              </a>
            </li>
          )}
          {project && (
            <li>
              <a
                href={project}
                target="_blank"
                rel="noopener noreferrer nofollow"
                className={EXTERNAL_LINK}
              >
                Project page
              </a>
            </li>
          )}
        </ul>
        <p className="text-sm text-muted-foreground">
          {fulltextLabel(p.fulltext_status)}
          {p.citation_count != null ? `, cited by ${p.citation_count}` : ""}
          {p.hf_upvotes != null
            ? `, ${p.hf_upvotes} upvotes on Hugging Face`
            : ""}
        </p>
      </header>

      <section aria-labelledby="abstract-heading" className="space-y-2">
        <h2 id="abstract-heading" className="text-lg font-semibold">
          Abstract
        </h2>
        <p className="whitespace-pre-wrap font-serif text-[1.0625rem] leading-8">
          {p.abstract}
        </p>
      </section>

      <Loaded what="analysis" query={details}>
        {(d) => (
          <AnalysisSection
            details={d}
            fulltextIndexed={p.fulltext_status === "indexed"}
          />
        )}
      </Loaded>

      <Loaded what="code and models" query={details}>
        {(d) => <ResourcesSection details={d} />}
      </Loaded>

      <Loaded what="related papers" query={related}>
        {(r) => <RelatedTabs related={r} />}
      </Loaded>

      <section aria-labelledby="passages-heading" className="space-y-4">
        <div className="space-y-1">
          <h2 id="passages-heading" className="text-lg font-semibold">
            Indexed passages
          </h2>
          <p className="text-sm text-muted-foreground">
            The pieces of this paper that search and chat can find and cite.
          </p>
        </div>
        {chunks.isPending && (
          <p role="status" className="text-muted-foreground">
            Loading the passages&hellip;
          </p>
        )}
        {chunks.isError && (
          <p role="alert" className="text-sm text-destructive">
            {chunks.error.message}
          </p>
        )}
        {chunks.data && chunks.data.chunks.length === 0 && (
          <p className="text-muted-foreground">
            No passages are indexed for this paper.
          </p>
        )}
        {chunks.data && chunks.data.chunks.length > 0 && (
          <ol className="space-y-5">
            {chunks.data.chunks.map((chunk) => (
              <li key={chunk.chunk_id} className="border-t pt-5">
                <Passage
                  sectionPath={chunk.section_path ?? null}
                  chunkType={chunk.chunk_type}
                  text={chunk.text}
                />
              </li>
            ))}
          </ol>
        )}
      </section>
    </article>
  );
}

// A section that waits for its own query: a loading line, its error, or its content.
// Each section fails on its own, so one slow or broken endpoint never hides the others.
function Loaded<T>({
  what,
  query,
  children,
}: {
  what: string;
  query: {
    isPending: boolean;
    isError: boolean;
    error: Error | null;
    data?: T;
  };
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) {
    return (
      <p role="status" className="text-muted-foreground">
        Loading the {what}&hellip;
      </p>
    );
  }
  if (query.isError || query.data === undefined) {
    return (
      <p role="alert" className="text-sm text-destructive">
        The {what} could not be loaded
        {query.error ? `: ${query.error.message}` : "."}
      </p>
    );
  }
  return <>{children(query.data)}</>;
}

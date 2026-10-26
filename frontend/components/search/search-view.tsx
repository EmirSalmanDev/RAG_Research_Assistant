"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";

import { Passage } from "@/components/passage";
import { Button } from "@/components/ui/button";
import { ErrorState } from "@/components/ui/error-state";
import { Input } from "@/components/ui/input";
import { LoadingState } from "@/components/ui/loading-state";
import { fetchSearch } from "@/lib/api/queries";
import { formatDate } from "@/lib/format";
import { paperPath } from "@/lib/links";
import { groupByPaper } from "@/lib/search/group";
import {
  KNOWN_CATEGORIES,
  MAX_QUERY_CHARS,
  parseSearchParams,
  toSearchString,
  validateSearch,
} from "@/lib/search/params";
import type { SearchParams } from "@/lib/search/params";
import { cn } from "@/lib/utils";

const MODE_LABEL: Record<string, string> = {
  hybrid: "Hybrid search (meaning and keywords)",
  dense: "Semantic search",
  bm25: "Keyword search",
};

function SearchForm({
  initial,
  initialProblems,
  onSubmit,
}: {
  initial: SearchParams | null;
  initialProblems: string[];
  onSubmit: (params: SearchParams) => void;
}) {
  const [q, setQ] = useState(initial?.q ?? "");
  const [categories, setCategories] = useState<string[]>(
    initial?.categories ?? [],
  );
  const [from, setFrom] = useState(initial?.from ?? "");
  const [to, setTo] = useState(initial?.to ?? "");
  const [problems, setProblems] = useState<string[]>(initialProblems);

  function toggle(category: string) {
    setCategories((cur) =>
      cur.includes(category)
        ? cur.filter((c) => c !== category)
        : [...cur, category],
    );
  }

  return (
    <form
      className="space-y-4"
      onSubmit={(e) => {
        e.preventDefault();
        const result = validateSearch({
          q,
          categories,
          from: from || null,
          to: to || null,
        });
        setProblems(result.problems);
        if (result.params) onSubmit(result.params);
      }}
    >
      <div className="flex gap-2">
        <label htmlFor="search-q" className="sr-only">
          Question or topic
        </label>
        <Input
          id="search-q"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          maxLength={MAX_QUERY_CHARS}
          placeholder="A method, a result, a topic"
          className="h-10 bg-card"
        />
        <Button type="submit" className="h-10" disabled={q.trim() === ""}>
          Search
        </Button>
      </div>

      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <fieldset className="space-y-1.5">
          <legend className="text-sm text-muted-foreground">Category</legend>
          <div className="flex flex-wrap gap-1.5">
            {KNOWN_CATEGORIES.map((category) => {
              const on = categories.includes(category);
              return (
                <button
                  key={category}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggle(category)}
                  className={cn(
                    "rounded-md border px-2.5 py-1 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    on
                      ? "border-primary bg-primary text-primary-foreground"
                      : "bg-card hover:bg-accent",
                  )}
                >
                  {category}
                </button>
              );
            })}
          </div>
        </fieldset>
        <div className="space-y-1.5">
          <label
            htmlFor="search-from"
            className="block text-sm text-muted-foreground"
          >
            Published from
          </label>
          <Input
            id="search-from"
            type="date"
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            className="w-40 bg-card"
          />
        </div>
        <div className="space-y-1.5">
          <label
            htmlFor="search-to"
            className="block text-sm text-muted-foreground"
          >
            Published until
          </label>
          <Input
            id="search-to"
            type="date"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            className="w-40 bg-card"
          />
        </div>
      </div>

      {problems.length > 0 && (
        <ul role="alert" className="space-y-0.5 text-sm text-destructive">
          {problems.map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
      )}
    </form>
  );
}

function Results({ params }: { params: SearchParams | null }) {
  // One request per distinct search. No retry and no refetch on focus (set once in
  // app/providers.tsx): a search embeds the question, which is a paid OpenAI request.
  const query = useQuery({
    queryKey: ["search", params],
    queryFn: ({ signal }) => fetchSearch(params as SearchParams, signal),
    enabled: params !== null,
  });
  const groups = useMemo(
    () => groupByPaper(query.data?.hits ?? []),
    [query.data],
  );

  if (params === null) {
    return (
      <p className="text-muted-foreground">
        Search finds passages in the indexed papers. For an answer with
        citations, use{" "}
        <Link
          href="/chat"
          className="text-primary underline-offset-4 hover:underline"
        >
          Chat
        </Link>
        .
      </p>
    );
  }
  if (query.isPending) {
    return <LoadingState label="Searching…" />;
  }
  if (query.isError) {
    return (
      <ErrorState error={query.error} onRetry={() => void query.refetch()} />
    );
  }

  const { hits, mode, reranked } = query.data;
  if (hits.length === 0) {
    return (
      <p className="text-muted-foreground">
        No passages matched. Try other words, or remove a category or date
        filter.
      </p>
    );
  }
  return (
    <section aria-label="Results" className="space-y-8">
      <p className="text-sm text-muted-foreground">
        {hits.length} {hits.length === 1 ? "passage" : "passages"} from{" "}
        {groups.length} {groups.length === 1 ? "paper" : "papers"}.{" "}
        {MODE_LABEL[mode] ?? mode}
        {reranked ? "" : ", not reranked"}.
      </p>
      <ol className="space-y-8">
        {groups.map((group) => (
          <li key={group.arxivId} className="space-y-4 border-t pt-6">
            <div className="space-y-0.5">
              <h2 className="font-serif text-xl font-semibold leading-snug">
                <Link
                  href={paperPath(group.arxivId)}
                  className="underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {group.title}
                </Link>
              </h2>
              <p className="text-sm text-muted-foreground">
                arXiv {group.arxivId}, {formatDate(group.publishedAt)}
              </p>
            </div>
            <ul className="space-y-5">
              {group.hits.map((hit) => (
                <li key={hit.chunk_id}>
                  <Passage
                    sectionPath={hit.section_path ?? null}
                    chunkType={hit.chunk_type}
                    text={hit.text}
                  />
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function SearchView() {
  const router = useRouter();
  const queryString = useSearchParams().toString();
  const parsed = useMemo(() => parseSearchParams(queryString), [queryString]);

  return (
    <div className="mx-auto w-full max-w-3xl space-y-8 px-4 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">
        Search the papers
      </h1>
      {/* keyed by the URL: opening another search (back button, a link) resets the form */}
      <SearchForm
        key={queryString}
        initial={parsed.params}
        initialProblems={parsed.problems}
        onSubmit={(params) => router.push(`/search?${toSearchString(params)}`)}
      />
      <Results params={parsed.params} />
    </div>
  );
}

"use client";

import { useMutation } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { postTimeline } from "@/lib/api/queries";
import type { TimelinePoint } from "@/lib/api/queries";
import { formatDate } from "@/lib/format";
import { arxivAbsUrl, paperPath } from "@/lib/links";
import { KNOWN_CATEGORIES } from "@/lib/search/params";
import {
  DEFAULT_KEY_PAPERS,
  DEFAULT_MAX_PAPERS,
  KEY_PAPERS_CAP,
  MAX_PAPERS_CAP,
  TOPIC_MAX,
  describeNotes,
  validateTimeline,
} from "@/lib/timeline/form";
import { cn } from "@/lib/utils";

import { TimelineChart, entryId } from "./timeline-chart";

function Entry({ point }: { point: TimelinePoint }) {
  const abs = arxivAbsUrl(point.abs_url);
  return (
    <li
      id={entryId(point.arxiv_id)}
      data-key={point.is_key}
      className={cn(
        "scroll-mt-20 space-y-1.5 border-l-2 py-1 pl-4",
        point.is_key ? "border-primary" : "border-border",
      )}
    >
      <p className="text-sm text-muted-foreground">
        {formatDate(point.published_at)}
        {point.is_key && (
          <span className="ml-2 rounded bg-primary px-1.5 py-0.5 text-xs font-medium text-primary-foreground">
            Key paper
          </span>
        )}
      </p>
      <h3
        className={cn(
          "leading-snug",
          point.is_key ? "font-serif text-lg font-semibold" : "text-sm",
        )}
      >
        <Link
          href={paperPath(point.arxiv_id)}
          className="underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {point.title}
        </Link>
      </h3>
      {point.line && (
        <p className="text-sm leading-6">
          {point.line}{" "}
          <span
            className="rounded border border-dashed px-1.5 py-0.5 text-xs text-muted-foreground"
            title={
              point.line_source === "abstract_fallback"
                ? "The first sentence of the abstract (the model's line was missing or failed a check)."
                : "Written by a model from this paper's abstract only."
            }
          >
            from the abstract
            {point.line_source === "abstract_fallback"
              ? " (first sentence)"
              : ""}
          </span>
        </p>
      )}
      <p className="flex flex-wrap gap-x-4 text-xs text-muted-foreground">
        <span>
          {point.citation_count == null
            ? "citation count unknown"
            : `cited by ${point.citation_count}`}
        </span>
        <Link
          href={paperPath(point.arxiv_id)}
          className="text-primary underline-offset-4 hover:underline"
        >
          Open the paper page
        </Link>
        {abs && (
          <a
            href={abs}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline-offset-4 hover:underline"
          >
            arXiv {point.arxiv_id}
          </a>
        )}
      </p>
    </li>
  );
}

export function TimelineView() {
  const [topic, setTopic] = useState("");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [categories, setCategories] = useState<string[]>([]);
  const [maxPapers, setMaxPapers] = useState(DEFAULT_MAX_PAPERS);
  const [keyPapers, setKeyPapers] = useState(DEFAULT_KEY_PAPERS);
  const [problems, setProblems] = useState<string[]>([]);

  // The ONLY paid call on this page, started by the Run button (or Enter in the form).
  const run = useMutation({
    mutationFn: (body: Parameters<typeof postTimeline>[0]) =>
      postTimeline(body),
  });

  function toggle(category: string) {
    setCategories((cur) =>
      cur.includes(category)
        ? cur.filter((c) => c !== category)
        : [...cur, category],
    );
  }

  const out = run.data;
  return (
    <div className="mx-auto w-full max-w-4xl space-y-8 px-4 py-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">
          Topic timeline
        </h1>
        <p className="text-sm text-muted-foreground">
          Papers on a topic by date, with the key ones marked and one line on
          what each changed, written from its abstract. Nothing is sent until
          you press Run (one embeddings request and one model call).
        </p>
      </div>

      <form
        className="space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          const result = validateTimeline({
            topic,
            from,
            to,
            categories,
            maxPapers,
            keyPapers,
          });
          setProblems(result.problems);
          if (result.body && !run.isPending) run.mutate(result.body);
        }}
      >
        <div className="flex gap-2">
          <label htmlFor="timeline-topic" className="sr-only">
            Topic
          </label>
          <Input
            id="timeline-topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            maxLength={TOPIC_MAX}
            placeholder="A topic, for example speculative decoding"
            className="h-10 bg-card"
            autoComplete="off"
          />
          <Button
            type="submit"
            className="h-10"
            disabled={topic.trim() === "" || run.isPending}
          >
            Run
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
              htmlFor="timeline-from"
              className="block text-sm text-muted-foreground"
            >
              Published from
            </label>
            <Input
              id="timeline-from"
              type="date"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              className="w-40 bg-card"
            />
          </div>
          <div className="space-y-1.5">
            <label
              htmlFor="timeline-to"
              className="block text-sm text-muted-foreground"
            >
              Published until
            </label>
            <Input
              id="timeline-to"
              type="date"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              className="w-40 bg-card"
            />
          </div>
          <div className="space-y-1.5">
            <label
              htmlFor="timeline-max"
              className="block text-sm text-muted-foreground"
            >
              Papers shown
            </label>
            <Input
              id="timeline-max"
              type="number"
              min={1}
              max={MAX_PAPERS_CAP}
              value={maxPapers}
              onChange={(e) => setMaxPapers(Number(e.target.value))}
              className="w-24 bg-card"
            />
          </div>
          <div className="space-y-1.5">
            <label
              htmlFor="timeline-key"
              className="block text-sm text-muted-foreground"
            >
              Key papers
            </label>
            <Input
              id="timeline-key"
              type="number"
              min={1}
              max={KEY_PAPERS_CAP}
              value={keyPapers}
              onChange={(e) => setKeyPapers(Number(e.target.value))}
              className="w-24 bg-card"
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

      {run.isPending && (
        <p role="status" className="text-muted-foreground">
          Building the timeline&hellip; this takes a few seconds.
        </p>
      )}
      {run.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-3 text-sm text-destructive"
        >
          {run.error.message}
        </p>
      )}
      {!out && !run.isPending && !run.isError && (
        <p className="text-muted-foreground">
          The timeline appears here. Key papers rank by how well they match the
          topic and by citation count.
        </p>
      )}

      {out && out.points.length === 0 && (
        <p className="text-muted-foreground">
          No paper matched &ldquo;{out.topic}&rdquo;. Try other words, or remove
          a category or date filter.
        </p>
      )}
      {out && out.points.length > 0 && (
        <section aria-label="Timeline" className="space-y-6">
          <div className="space-y-2 rounded-md border bg-card p-3">
            <TimelineChart points={out.points} />
            <p className="text-xs text-muted-foreground">
              Large dots are key papers; click one to jump to its entry.
            </p>
          </div>
          <div
            className="space-y-1 text-sm text-muted-foreground"
            data-testid="timeline-notes"
          >
            {describeNotes(out).map((line) => (
              <p key={line}>{line}</p>
            ))}
            <p>
              One model call ({out.prompt_version}): {out.tokens_in} tokens in,{" "}
              {out.tokens_out} out, about ${out.cost_usd.toFixed(4)}.
            </p>
          </div>
          <ol className="space-y-5" aria-label="Papers by date">
            {out.points.map((point) => (
              <Entry key={point.arxiv_id} point={point} />
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}

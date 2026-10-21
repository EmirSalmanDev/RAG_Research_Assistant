"use client";

import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { useEvalResult } from "@/lib/api/queries";
import type { EvalResultDetailOut, RetrievedChunkOut } from "@/lib/api/queries";
import type { JoinedRow } from "@/lib/evals/compare";
import { formatValue, metricLabel, orderKeys } from "@/lib/evals/metrics";
import type { PanelTarget, Side } from "@/lib/evals/panel";
import { runTitle } from "@/lib/evals/runs";
import type { EvalRunSummary } from "@/lib/evals/runs";
import { arxivAbsUrl } from "@/lib/links";
import { cn } from "@/lib/utils";

import { JudgeVerdicts } from "./judge-verdicts";
import { RunStatus } from "./run-status";

function Block({
  title,
  children,
  testId,
}: {
  title: string;
  children: React.ReactNode;
  testId?: string;
}) {
  return (
    <section className="space-y-2" data-testid={testId}>
      <h3 className="text-sm font-semibold">{title}</h3>
      {children}
    </section>
  );
}

function Answer({ detail }: { detail: EvalResultDetailOut }) {
  if (detail.outcome === "error")
    return (
      <p role="alert" className="text-sm text-destructive">
        The turn failed: {detail.error ?? "no reason stored"}
      </p>
    );
  if (detail.outcome === "refused")
    return (
      <p className="text-sm">
        <span className="font-medium">Refused</span>
        {detail.refusal ? ` (${detail.refusal})` : ""}
        {detail.answer ? `: ${detail.answer}` : "."}
      </p>
    );
  if (detail.outcome === "retrieved")
    return (
      <p className="text-sm text-muted-foreground">
        A retrieval run: only the search was evaluated, no answer was written.
      </p>
    );
  return (
    <blockquote className="whitespace-pre-wrap border-l-2 border-primary/50 pl-4 font-serif text-base leading-7">
      {detail.answer}
    </blockquote>
  );
}

function Chunk({ chunk }: { chunk: RetrievedChunkOut }) {
  // The link comes from the API; only a real arXiv abstract page is ever linked (no PDF).
  const abs = arxivAbsUrl(chunk.abs_url);
  return (
    <li
      data-testid="retrieved-chunk"
      data-reference={chunk.is_reference}
      className={cn(
        "space-y-2 rounded-md border p-3",
        chunk.is_reference && "border-primary/60",
      )}
    >
      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span className="font-medium text-foreground">Rank {chunk.rank}</span>
        {chunk.n != null && <span>source [{chunk.n}]</span>}
        {chunk.chunk_type && (
          <Badge variant="outline">{chunk.chunk_type}</Badge>
        )}
        {chunk.is_reference && (
          <Badge data-testid="reference-badge">Reference passage</Badge>
        )}
        {chunk.score != null && (
          <span className="tabular-nums">
            score {formatValue("score", chunk.score)}
          </span>
        )}
      </div>
      <p className="text-sm font-medium leading-snug">
        {chunk.title ?? "Untitled"}
        {chunk.section_path && (
          <span className="font-normal text-muted-foreground">
            {" "}
            · {chunk.section_path}
          </span>
        )}
      </p>
      {chunk.text ? (
        <p
          data-testid="chunk-text"
          className="whitespace-pre-wrap font-serif text-sm leading-6"
        >
          {chunk.text}
        </p>
      ) : (
        <p className="text-sm italic text-muted-foreground">
          The text of this chunk was not stored with the run.
        </p>
      )}
      <p className="flex flex-wrap gap-x-4 text-xs">
        {abs ? (
          <a
            href={abs}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline-offset-4 hover:underline"
          >
            arXiv {chunk.arxiv_id} (abstract page)
          </a>
        ) : (
          <span className="text-muted-foreground">
            {chunk.arxiv_id ?? "no arXiv id"}
          </span>
        )}
      </p>
    </li>
  );
}

function Detail({ detail }: { detail: EvalResultDetailOut }) {
  const scores = orderKeys(Object.keys(detail.scores));
  const errors = Object.entries(detail.judge_errors);
  return (
    <div className="space-y-6">
      <Block title="Reference answer">
        {detail.reference_answer ? (
          <p className="whitespace-pre-wrap text-sm leading-6">
            {detail.reference_answer}
          </p>
        ) : (
          <p className="text-sm text-muted-foreground">
            None: the question has no answer in the corpus (the right outcome is
            a refusal), or no reference was written.
          </p>
        )}
      </Block>

      <Block title="Answer" testId="eval-answer">
        <Answer detail={detail} />
        <p className="text-xs text-muted-foreground">
          {detail.latency_ms != null &&
            `${formatValue("latency_ms", detail.latency_ms)}`}
          {detail.latency_ms != null && detail.cost_usd != null && ", "}
          {detail.cost_usd != null &&
            `${formatValue("cost_usd", detail.cost_usd)}`}
        </p>
      </Block>

      <Block title="Scores">
        {scores.length === 0 ? (
          <p className="text-sm text-muted-foreground">No score was stored.</p>
        ) : (
          <table
            className="w-full border-collapse text-sm"
            data-testid="question-scores"
          >
            <tbody>
              {scores.map((key) => (
                <tr key={key} className="border-b last:border-b-0">
                  <th scope="row" className="py-1 pr-3 text-left font-normal">
                    {metricLabel(key)}
                  </th>
                  <td className="py-1 text-right tabular-nums">
                    {formatValue(key, detail.scores[key])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Block>

      <Block title="The judge's verdicts">
        <JudgeVerdicts judge={detail.judge} />
        {errors.length > 0 && (
          <div className="space-y-1" data-testid="judge-errors">
            <p className="text-sm font-medium">
              Metrics the judge could not score
            </p>
            <ul className="space-y-0.5 text-sm">
              {errors.map(([metric, reason]) => (
                <li key={metric}>
                  <span className="font-medium">{metricLabel(metric)}:</span>{" "}
                  <span className="text-muted-foreground">{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Block>

      <Block
        title={`Retrieved chunks (${detail.retrieved.length}), in rank order`}
      >
        {detail.retrieved.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Nothing was retrieved.
          </p>
        ) : (
          <ol className="space-y-3">
            {detail.retrieved.map((chunk) => (
              <Chunk key={`${chunk.rank}-${chunk.chunk_id}`} chunk={chunk} />
            ))}
          </ol>
        )}
      </Block>
    </div>
  );
}

// One question of a run, beside the tables: like the chat's source panel it is
// non-modal (the tables stay usable, a click on another question switches it) and its
// data is fetched when it opens. A question in both runs can be switched between A and B.
export function QuestionPanel({
  row,
  targets,
  target,
  runs,
  onSide,
  onClose,
}: {
  row: JoinedRow | null;
  targets: readonly PanelTarget[];
  target: PanelTarget | null;
  runs: readonly EvalRunSummary[];
  onSide: (side: Side) => void;
  onClose: () => void;
}) {
  const result = useEvalResult(
    target ? { runId: target.runId, itemId: target.itemId } : null,
  );
  const run = runs.find((r) => r.id === target?.runId) ?? null;
  const open = row !== null && target !== null;

  return (
    <Sheet
      modal={false}
      open={open}
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
    >
      <SheetContent
        side="right"
        overlay={false}
        aria-label="Question"
        // Clicking the tables behind does not close it (Escape and the close button do).
        onInteractOutside={(event) => event.preventDefault()}
        className="flex w-full flex-col gap-0 overflow-y-auto border-l p-0 sm:max-w-2xl"
      >
        {open && (
          <>
            <SheetHeader className="space-y-2 border-b p-6 pr-12 text-left">
              <SheetTitle className="font-serif text-xl leading-snug">
                {row.question}
              </SheetTitle>
              <SheetDescription>
                {row.questionId ?? "no question id"}
                {row.split
                  ? ` · ${row.split === "heldout" ? "held-out (golden)" : row.split}`
                  : ""}
                {row.type ? ` · ${row.type}` : ""}
              </SheetDescription>
              {targets.length > 1 && (
                <div
                  className="flex gap-1.5"
                  role="group"
                  aria-label="Run shown"
                >
                  {targets.map((t) => (
                    <button
                      key={t.side}
                      type="button"
                      aria-pressed={t.side === target.side}
                      onClick={() => onSide(t.side)}
                      className={cn(
                        "rounded-md border px-2.5 py-1 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                        t.side === target.side
                          ? "border-primary bg-primary text-primary-foreground"
                          : "bg-card hover:bg-accent",
                      )}
                    >
                      Run {t.side.toUpperCase()} #{t.runId}
                    </button>
                  ))}
                </div>
              )}
              {run && (
                <p className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <span>
                    Run {target.side.toUpperCase()}: {runTitle(run)}
                  </span>
                  <RunStatus status={run.status} />
                </p>
              )}
            </SheetHeader>
            <div className="flex-1 p-6">
              {result.isPending && (
                <p role="status" className="text-sm text-muted-foreground">
                  Loading the question&hellip;
                </p>
              )}
              {result.isError && (
                <p role="alert" className="text-sm text-destructive">
                  {result.error.message}
                </p>
              )}
              {result.data && <Detail detail={result.data} />}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}

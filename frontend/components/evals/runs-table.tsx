"use client";

import { formatDate } from "@/lib/format";
import { formatValue, metricLabel } from "@/lib/evals/metrics";
import { seriesLabel } from "@/lib/evals/runs";
import type { EvalRunSummary } from "@/lib/evals/runs";
import { metricValue } from "@/lib/evals/trend";
import { cn } from "@/lib/utils";

import { RunStatus } from "./run-status";

function time(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? ""
    : date.toLocaleTimeString("en", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        timeZone: "UTC",
      });
}

// Every run, newest first, selections included: the table view of the chart (the chosen
// metric has a column) and a quick way to pick the two runs to compare.
export function RunsTable({
  runs,
  metric,
  runA,
  runB,
  onPick,
}: {
  runs: readonly EvalRunSummary[];
  metric: string;
  runA: number | null;
  runB: number | null;
  onPick: (side: "a" | "b", runId: number) => void;
}) {
  return (
    <section aria-labelledby="runs-heading" className="space-y-3">
      <h2 id="runs-heading" className="text-lg font-semibold">
        Runs ({runs.length})
      </h2>
      <div className="max-h-[28rem] overflow-auto rounded-md border bg-card">
        <table className="w-full min-w-[48rem] border-collapse text-left text-sm">
          <caption className="sr-only">
            Evaluation runs, newest first, with {metricLabel(metric)}
          </caption>
          <thead className="sticky top-0 bg-muted">
            <tr className="border-b">
              <th scope="col" className="p-2 font-medium">
                Run
              </th>
              <th scope="col" className="p-2 font-medium">
                Kind
              </th>
              <th scope="col" className="p-2 font-medium">
                Created (UTC)
              </th>
              <th scope="col" className="p-2 text-right font-medium">
                Questions
              </th>
              <th scope="col" className="p-2 font-medium">
                Status
              </th>
              <th scope="col" className="p-2 text-right font-medium">
                {metricLabel(metric)}
              </th>
              <th scope="col" className="p-2 font-medium">
                Commit
              </th>
              <th scope="col" className="p-2 font-medium">
                <span className="sr-only">Compare</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => {
              const value = metricValue(run, metric);
              const picked =
                run.id === runA ? "A" : run.id === runB ? "B" : null;
              return (
                <tr
                  key={run.id}
                  data-testid="run-row"
                  data-run={run.id}
                  className={cn(
                    "border-b align-top last:border-b-0",
                    picked && "bg-accent/60",
                  )}
                >
                  <td className="p-2">
                    <span className="font-medium">#{run.id}</span>{" "}
                    {run.kind === "selection" ? (
                      <span>{run.name}</span>
                    ) : (
                      <span>{seriesLabel(run)}</span>
                    )}
                    {run.name && run.kind !== "selection" && (
                      <span className="block text-xs text-muted-foreground">
                        {run.name}
                      </span>
                    )}
                  </td>
                  <td className="p-2">{run.kind}</td>
                  <td className="whitespace-nowrap p-2">
                    {formatDate(run.created_at)} {time(run.created_at)}
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {run.kind === "selection" ? (
                      <span className="text-xs text-muted-foreground">
                        no per-question results
                      </span>
                    ) : (
                      (run.n_questions ?? "—")
                    )}
                  </td>
                  <td className="p-2">
                    <RunStatus status={run.status} />
                  </td>
                  <td className="p-2 text-right tabular-nums">
                    {formatValue(metric, value)}
                  </td>
                  <td className="p-2 font-mono text-xs">
                    {run.git_sha ? run.git_sha.slice(0, 7) : "—"}
                  </td>
                  <td className="whitespace-nowrap p-2">
                    <div className="flex gap-1">
                      {(["a", "b"] as const).map((side) => {
                        const on = (side === "a" ? runA : runB) === run.id;
                        return (
                          <button
                            key={side}
                            type="button"
                            aria-pressed={on}
                            aria-label={`Compare run #${run.id} as ${side.toUpperCase()}`}
                            onClick={() => onPick(side, run.id)}
                            className={cn(
                              "rounded border px-2 py-0.5 text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                              on
                                ? "border-primary bg-primary text-primary-foreground"
                                : "hover:bg-accent",
                            )}
                          >
                            {side.toUpperCase()}
                          </button>
                        );
                      })}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

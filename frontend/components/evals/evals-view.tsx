"use client";

import { useState } from "react";

import { useEvalRuns } from "@/lib/api/queries";
import type { JoinedRow } from "@/lib/evals/compare";
import { panelTargets, pickTarget } from "@/lib/evals/panel";
import type { Side } from "@/lib/evals/panel";
import { defaultMetric } from "@/lib/evals/trend";

import { QuestionPanel } from "./question-panel";
import { RunCompare } from "./run-compare";
import { RunsTable } from "./runs-table";
import { TrendSection } from "./trend-section";

// The /evals page: metric trends over the runs, the runs table, two runs side by side
// and one question in a side panel. Everything is read from stored runs (GET /evals/...,
// Postgres only): opening the page runs nothing and costs nothing.
export function EvalsView() {
  const runs = useEvalRuns();
  const [metric, setMetric] = useState<string | null>(null);
  const [runA, setRunA] = useState<number | null>(null);
  const [runB, setRunB] = useState<number | null>(null);
  const [panel, setPanel] = useState<{
    row: JoinedRow;
    side: Side | null;
  } | null>(null);

  const list = runs.data ?? [];
  // Until the reader picks a metric, show the first one that some run has.
  const shownMetric = metric ?? defaultMetric(list);
  const targets = panel ? panelTargets(panel.row, runA, runB) : [];
  const target = pickTarget(targets, panel?.side ?? null);

  function pick(side: "a" | "b", runId: number | null) {
    if (side === "a") setRunA(runId);
    else setRunB(runId);
    setPanel(null); // the open question may not be in the new run
  }

  return (
    <div className="mx-auto w-full max-w-6xl space-y-10 px-4 py-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Evaluations</h1>
        <p className="text-sm text-muted-foreground">
          Every stored evaluation run: how the metrics moved, two runs side by
          side, and each question with what was retrieved, the answer and the
          judge&rsquo;s verdicts. This page only reads stored results; it runs
          nothing and costs nothing.
        </p>
      </div>

      {runs.isPending && (
        <p role="status" className="text-muted-foreground">
          Loading the runs&hellip;
        </p>
      )}
      {runs.isError && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-3 text-sm text-destructive"
        >
          {runs.error.message}
        </p>
      )}
      {runs.data && list.length === 0 && (
        <p className="text-muted-foreground" data-testid="no-runs">
          No evaluation run is stored yet. Runs appear here after{" "}
          <code className="rounded bg-muted px-1">make eval-retrieval</code> or
          an answer evaluation.
        </p>
      )}

      {list.length > 0 && (
        <>
          <TrendSection runs={list} metric={shownMetric} onMetric={setMetric} />
          <RunsTable
            runs={list}
            metric={shownMetric}
            runA={runA}
            runB={runB}
            onPick={(side, id) =>
              // a second click on a picked run's button unpicks it
              pick(side, (side === "a" ? runA : runB) === id ? null : id)
            }
          />
          <RunCompare
            runs={list}
            runA={runA}
            runB={runB}
            onPick={pick}
            preferredScore={shownMetric}
            onOpen={(row) => setPanel({ row, side: panel?.side ?? null })}
            activeKey={panel?.row.key ?? null}
          />
        </>
      )}

      <QuestionPanel
        row={panel?.row ?? null}
        targets={targets}
        target={target}
        runs={list}
        onSide={(side) => setPanel((cur) => (cur ? { ...cur, side } : cur))}
        onClose={() => setPanel(null)}
      />
    </div>
  );
}

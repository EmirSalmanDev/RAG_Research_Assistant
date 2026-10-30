"use client";

import { useStatus } from "@/lib/api/queries";
import type { StatusOut } from "@/lib/api/queries";
import { formatAge, paperRows, summarize } from "@/lib/health/status";
import type { Level } from "@/lib/health/status";
import { formatUsd } from "@/lib/settings/form";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ErrorState } from "@/components/ui/error-state";
import { LoadingState } from "@/components/ui/loading-state";

const LEVEL_TEXT: Record<Level, string> = {
  ok: "Everything is running.",
  warning: "Running, but something needs attention.",
  down: "Something is down.",
};

const LEVEL_CLASS: Record<Level, string> = {
  ok: "border-emerald-600/40 text-emerald-700 dark:text-emerald-400",
  warning: "border-amber-600/40 text-amber-700 dark:text-amber-400",
  down: "border-destructive/40 text-destructive",
};

export function HealthView() {
  const status = useStatus();
  return (
    <div className="mx-auto w-full max-w-4xl space-y-6 px-4 py-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Health</h1>
        <p className="text-sm text-muted-foreground">
          The stores, the nightly pipeline and this month&apos;s spending. Read
          again every 30 seconds.
        </p>
      </div>
      {status.isPending && <LoadingState label="Reading the system status…" />}
      {status.isError && (
        <ErrorState
          what="The system status"
          error={status.error}
          onRetry={() => void status.refetch()}
        />
      )}
      {status.data && <StatusSections status={status.data} />}
    </div>
  );
}

function StatusSections({ status }: { status: StatusOut }) {
  const summary = summarize(status);
  const now = new Date();
  const database = status.database;
  return (
    <>
      <section
        aria-label="Summary"
        data-testid="health-summary"
        data-level={summary.level}
        className={`space-y-1 rounded-md border p-3 text-sm ${LEVEL_CLASS[summary.level]}`}
      >
        <p className="font-medium">{LEVEL_TEXT[summary.level]}</p>
        {summary.problems.length > 0 && (
          <ul className="list-disc pl-5">
            {summary.problems.map((problem) => (
              <li key={problem}>{problem}</li>
            ))}
          </ul>
        )}
      </section>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Stores</CardTitle>
            <CardDescription>Probed on every read.</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1 text-sm">
              {Object.entries(status.stores).map(([name, state]) => (
                <li key={name} className="flex justify-between">
                  <span className="capitalize">{name}</span>
                  <Badge variant={state === "up" ? "secondary" : "destructive"}>
                    {state}
                  </Badge>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>This month</CardTitle>
            <CardDescription>
              Spending from the usage log, against the cap on Settings.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            {database ? (
              <>
                <p data-testid="health-spend">
                  Spent {formatUsd(database.budget.spent_usd)} of{" "}
                  {database.budget.cap_usd === null
                    ? "no cap set"
                    : formatUsd(database.budget.cap_usd)}
                </p>
                <p data-testid="health-budget-state">
                  Background jobs that spend money:{" "}
                  <strong>
                    {database.budget.paused ? "paused" : "running"}
                  </strong>
                </p>
              </>
            ) : (
              <p className="text-muted-foreground">Not available.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Papers</CardTitle>
            <CardDescription>
              {database
                ? `${database.papers_total} in the database`
                : "Not available"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1 text-sm">
              {paperRows(database?.papers_by_fulltext_status ?? {}).map(
                (row) => (
                  <li key={row.status} className="flex justify-between">
                    <span>{row.label}</span>
                    <span className="tabular-nums">{row.count}</span>
                  </li>
                ),
              )}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Search index and queue</CardTitle>
            <CardDescription>Qdrant points and waiting tasks.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-1 text-sm">
            <p data-testid="health-points">
              Indexed points:{" "}
              {status.qdrant_points === null
                ? "not available"
                : status.qdrant_points.toLocaleString("en")}
            </p>
            <p data-testid="health-queue">
              Tasks waiting:{" "}
              {status.queue_length === null
                ? "not available"
                : status.queue_length}
            </p>
          </CardContent>
        </Card>
      </div>

      <section aria-label="Pipeline runs" className="space-y-2">
        <h2 className="text-lg font-semibold tracking-tight">
          Last run of each step
        </h2>
        {database === null || database.last_runs.length === 0 ? (
          <p className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
            No pipeline step has run yet. The scheduler starts them at 07:00
            (see the README).
          </p>
        ) : (
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-left text-sm">
              <thead className="bg-muted/50 text-muted-foreground">
                <tr>
                  <th className="px-3 py-2 font-medium">Step</th>
                  <th className="px-3 py-2 font-medium">Started</th>
                  <th className="px-3 py-2 font-medium">Result</th>
                </tr>
              </thead>
              <tbody>
                {database.last_runs.map((run) => (
                  <tr key={run.source} className="border-t">
                    <td className="px-3 py-2 font-mono text-xs">
                      {run.source}
                    </td>
                    <td className="px-3 py-2">
                      {formatAge(run.started_at, now)}
                    </td>
                    <td className="px-3 py-2">
                      {run.error ? (
                        <span className="text-destructive">{run.error}</span>
                      ) : run.skipped ? (
                        <span className="text-amber-700 dark:text-amber-400">
                          skipped: {run.skipped}
                        </span>
                      ) : run.finished_at === null ? (
                        "running"
                      ) : (
                        "ok"
                      )}
                      {run.note ? (
                        <span className="block text-xs text-muted-foreground">
                          {run.note}
                        </span>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}

"use client";

import { useState } from "react";

import { METRICS } from "@/lib/evals/metrics";
import type { EvalRunSummary } from "@/lib/evals/runs";
import { buildTrend, metricCounts } from "@/lib/evals/trend";
import { cn } from "@/lib/utils";

import { TrendChart, seriesClass } from "./trend-chart";

const SELECT_CLASS =
  "h-9 rounded-md border border-input bg-card px-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

// The metric picker, the chart and its legend. The legend's buttons switch a series on
// or off; the first eight series of a kind have a colour and start on, any further ones
// are grey and start off (a ninth colour would be too close to another one).
export function TrendSection({
  runs,
  metric,
  onMetric,
}: {
  runs: readonly EvalRunSummary[];
  metric: string;
  onMetric: (metric: string) => void;
}) {
  const [toggled, setToggled] = useState<ReadonlySet<string>>(new Set());
  const trend = buildTrend(runs, metric, toggled);
  const counts = metricCounts(runs);
  const hasPoints = trend?.series.some((s) => s.points.length > 0) ?? false;

  function toggle(key: string) {
    setToggled((cur) => {
      const next = new Set(cur);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  return (
    <section aria-labelledby="trend-heading" className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h2 id="trend-heading" className="text-lg font-semibold">
          Metric trends
        </h2>
        <div className="flex items-center gap-2">
          <label
            htmlFor="trend-metric"
            className="text-sm text-muted-foreground"
          >
            Metric
          </label>
          <select
            id="trend-metric"
            value={metric}
            onChange={(e) => onMetric(e.target.value)}
            className={SELECT_CLASS}
          >
            {(["retrieval", "answers"] as const).map((group) => (
              <optgroup
                key={group}
                label={group === "retrieval" ? "Retrieval runs" : "Answer runs"}
              >
                {METRICS.filter((m) => m.group === group).map((m) => {
                  const n = counts.get(m.key) ?? 0;
                  return (
                    <option key={m.key} value={m.key} disabled={n === 0}>
                      {m.label}
                      {m.better === "lower" ? " (lower is better)" : ""}
                      {n === 0 ? " (no runs)" : ""}
                    </option>
                  );
                })}
              </optgroup>
            ))}
          </select>
        </div>
      </div>

      <div className="space-y-3 rounded-md border bg-card p-3">
        {trend && hasPoints ? (
          <TrendChart trend={trend} />
        ) : (
          <p className="py-8 text-center text-sm text-muted-foreground">
            {trend && trend.series.length > 0
              ? "Every series is switched off. Switch one on below."
              : "No run has this metric yet."}
          </p>
        )}
        {trend && trend.series.length > 0 && (
          <ul aria-label="Series" className="flex flex-wrap gap-1.5">
            {trend.series.map((s) => (
              <li key={s.key}>
                <button
                  type="button"
                  aria-pressed={s.visible}
                  onClick={() => toggle(s.key)}
                  title={s.key}
                  className={cn(
                    "flex items-center gap-2 rounded-md border px-2 py-1 text-xs transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    !s.visible && "text-muted-foreground line-through",
                  )}
                >
                  <span
                    aria-hidden
                    className={cn(
                      "h-0.5 w-4 rounded",
                      seriesClass(s.slot).bg,
                      !s.visible && "opacity-30",
                    )}
                  />
                  {s.label}
                  <span className="text-muted-foreground">
                    {s.count} run{s.count === 1 ? "" : "s"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className="text-xs text-muted-foreground">
          Runs in time order, oldest left. Lines join complete runs only. A
          hollow point is a run that stopped, is still running or has no status:
          its numbers cover only the questions it got through, so it is never
          part of a line. Shares are drawn on a fixed 0 to 1 scale; latency and
          cost on their own. The runs table below has every value.
        </p>
      </div>
    </section>
  );
}

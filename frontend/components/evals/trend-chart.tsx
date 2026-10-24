"use client";

import { formatDate } from "@/lib/format";
import { formatValue } from "@/lib/evals/metrics";
import { runState, runTitle } from "@/lib/evals/runs";
import type { Trend, TrendSeries } from "@/lib/evals/trend";
import { cn } from "@/lib/utils";

// Plain SVG: no chart library. One line per series over the runs in time order (x is the
// run order, see lib/evals/trend.ts), y is the metric. Lines join only COMPLETE runs: a
// stopped, running or status-less run is a hollow point off the line, because its
// numbers cover only part of the questions. Every point has a tooltip (its <title>).
const WIDTH = 800;
const HEIGHT = 300;
const LEFT = 64; // room for the y labels
const RIGHT = 20;
const TOP = 14;
const BOTTOM = 42; // two lines of x labels: the run id and its date
const PLOT_W = WIDTH - LEFT - RIGHT;
const PLOT_H = HEIGHT - TOP - BOTTOM;

// Eight categorical colours, light and dark steps of the same hues, in a fixed order
// chosen so that neighbouring slots stay apart for colour-blind readers. Written out in
// full so that Tailwind finds every class. Series past eight are grey.
const SLOT_CLASS = [
  {
    stroke: "stroke-[#2a78d6] dark:stroke-[#3987e5]",
    fill: "fill-[#2a78d6] dark:fill-[#3987e5]",
    bg: "bg-[#2a78d6] dark:bg-[#3987e5]",
  },
  {
    stroke: "stroke-[#eb6834] dark:stroke-[#d95926]",
    fill: "fill-[#eb6834] dark:fill-[#d95926]",
    bg: "bg-[#eb6834] dark:bg-[#d95926]",
  },
  {
    stroke: "stroke-[#1baf7a] dark:stroke-[#199e70]",
    fill: "fill-[#1baf7a] dark:fill-[#199e70]",
    bg: "bg-[#1baf7a] dark:bg-[#199e70]",
  },
  {
    stroke: "stroke-[#eda100] dark:stroke-[#c98500]",
    fill: "fill-[#eda100] dark:fill-[#c98500]",
    bg: "bg-[#eda100] dark:bg-[#c98500]",
  },
  {
    stroke: "stroke-[#e87ba4] dark:stroke-[#d55181]",
    fill: "fill-[#e87ba4] dark:fill-[#d55181]",
    bg: "bg-[#e87ba4] dark:bg-[#d55181]",
  },
  {
    stroke: "stroke-[#008300] dark:stroke-[#008300]",
    fill: "fill-[#008300] dark:fill-[#008300]",
    bg: "bg-[#008300] dark:bg-[#008300]",
  },
  {
    stroke: "stroke-[#6250d6] dark:stroke-[#9085e9]",
    fill: "fill-[#6250d6] dark:fill-[#9085e9]",
    bg: "bg-[#6250d6] dark:bg-[#9085e9]",
  },
  {
    stroke: "stroke-[#e34948] dark:stroke-[#e66767]",
    fill: "fill-[#e34948] dark:fill-[#e66767]",
    bg: "bg-[#e34948] dark:bg-[#e66767]",
  },
] as const;
const GREY = {
  stroke: "stroke-muted-foreground",
  fill: "fill-muted-foreground",
  bg: "bg-muted-foreground",
} as const;

export function seriesClass(slot: number | null) {
  return slot === null ? GREY : (SLOT_CLASS[slot] ?? GREY);
}

/** A y tick: shares as 0.25, the rest in their unit. */
function tickLabel(trend: Trend, value: number): string {
  if (trend.metric.unit === "share") return value.toFixed(2);
  if (trend.metric.unit === "usd") return `$${value}`;
  return formatValue(trend.metric.key, value);
}

function shortDate(iso: string): string {
  return formatDate(iso).replace(/,\s*\d{4}$/, ""); // "Oct 21"
}

export function TrendChart({ trend }: { trend: Trend }) {
  const count = trend.runs.length;
  // A single run sits in the middle instead of on the left edge.
  const xAt = (x: number) =>
    count <= 1 ? LEFT + PLOT_W / 2 : LEFT + (x / (count - 1)) * PLOT_W;
  const yAt = (v: number) => TOP + (1 - trend.y.positionOf(v)) * PLOT_H;
  const visible = trend.series.filter((s) => s.visible);

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="group"
      aria-label={`${trend.metric.label} over ${count} runs, ${visible.length} series`}
      className="w-full"
      data-testid="trend-chart"
      data-metric={trend.metric.key}
    >
      {trend.y.ticks.map((tick) => (
        <g key={tick}>
          <line
            x1={LEFT}
            x2={WIDTH - RIGHT}
            y1={yAt(tick)}
            y2={yAt(tick)}
            stroke="currentColor"
            strokeOpacity={0.12}
          />
          <text
            x={LEFT - 8}
            y={yAt(tick) + 4}
            textAnchor="end"
            fontSize={11}
            fill="currentColor"
            fillOpacity={0.7}
          >
            {tickLabel(trend, tick)}
          </text>
        </g>
      ))}
      {trend.ticks.map((i) => {
        const run = trend.runs[i];
        if (!run) return null;
        const x = xAt(i);
        return (
          <g key={run.id}>
            <line
              x1={x}
              x2={x}
              y1={TOP + PLOT_H}
              y2={TOP + PLOT_H + 4}
              stroke="currentColor"
              strokeOpacity={0.5}
            />
            <text
              x={x}
              y={TOP + PLOT_H + 17}
              textAnchor="middle"
              fontSize={11}
              fill="currentColor"
              fillOpacity={0.7}
            >
              #{run.id}
            </text>
            <text
              x={x}
              y={TOP + PLOT_H + 31}
              textAnchor="middle"
              fontSize={10}
              fill="currentColor"
              fillOpacity={0.55}
            >
              {shortDate(run.created_at)}
            </text>
          </g>
        );
      })}
      {visible.map((series) => (
        <SeriesMarks
          key={series.key}
          series={series}
          xAt={xAt}
          yAt={yAt}
          trend={trend}
        />
      ))}
    </svg>
  );
}

function SeriesMarks({
  series,
  trend,
  xAt,
  yAt,
}: {
  series: TrendSeries;
  trend: Trend;
  xAt: (x: number) => number;
  yAt: (v: number) => number;
}) {
  const cls = seriesClass(series.slot);
  const line = series.points
    .filter((p) => p.complete)
    .map((p) => `${xAt(p.x)},${yAt(p.value)}`)
    .join(" ");
  return (
    <g data-testid="trend-series" data-series={series.key}>
      {line && (
        <polyline
          points={line}
          fill="none"
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
          className={cls.stroke}
          data-testid="trend-line"
        />
      )}
      {series.points.map((p) => {
        const state = runState(p.run.status);
        const cx = xAt(p.x);
        const cy = yAt(p.value);
        return (
          <g
            key={p.run.id}
            data-testid="trend-point"
            data-run={p.run.id}
            data-state={state}
          >
            <title>{`${runTitle(p.run)}, ${formatDate(p.run.created_at)}: ${trend.metric.label} ${formatValue(trend.metric.key, p.value)}${p.complete ? "" : ` (${p.run.status ?? "no status"}: not a complete run)`}`}</title>
            {/* a larger invisible target, so the tooltip is easy to reach */}
            <circle cx={cx} cy={cy} r={10} fill="transparent" />
            <circle
              cx={cx}
              cy={cy}
              r={p.complete ? 4 : 4.5}
              strokeWidth={2}
              // complete: filled, with a ring in the background colour where it
              // crosses a line; otherwise hollow, ringed in the series colour
              className={
                p.complete
                  ? cn(cls.fill, "stroke-background")
                  : cn("fill-background", cls.stroke)
              }
            />
          </g>
        );
      })}
    </g>
  );
}

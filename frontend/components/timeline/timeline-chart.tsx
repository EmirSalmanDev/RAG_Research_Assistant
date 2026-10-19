"use client";

import type { TimelinePoint } from "@/lib/api/queries";
import { formatDate } from "@/lib/format";
import { assignLanes, buildAxis } from "@/lib/timeline/axis";

// Plain SVG: no chart library. Every paper is a dot on a time axis; key papers are larger
// and filled, and link to their entry in the list below. Dots that would overlap are put
// in separate rows. The axis unit (day, month, year) follows the range of the papers.
const WIDTH = 800;
const PAD_X = 28;
const TOP = 18; // room above the highest row of dots
const LANE_HEIGHT = 20;
const MIN_GAP = 12 / (WIDTH - 2 * PAD_X); // about one dot's width, as a fraction of the axis
const BELOW = 34; // the axis ticks and their labels

export function entryId(arxivId: string): string {
  return `paper-${arxivId.replace(/[^0-9A-Za-z.-]/g, "-")}`;
}

export function TimelineChart({
  points,
}: {
  points: readonly TimelinePoint[];
}) {
  const axis = buildAxis(points.map((p) => p.published_at));
  const xs = points.map(
    (p) =>
      PAD_X + axis.positionOf(Date.parse(p.published_at)) * (WIDTH - 2 * PAD_X),
  );
  const lanes = assignLanes(
    xs.map((x) => (x - PAD_X) / (WIDTH - 2 * PAD_X)),
    MIN_GAP,
    6,
  );

  // The chart is only as tall as the rows of dots it needs.
  const rows = Math.max(1, ...lanes.map((lane) => lane + 1));
  const AXIS_Y = TOP + rows * LANE_HEIGHT;
  const HEIGHT = AXIS_Y + BELOW;

  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      role="group"
      aria-label={`Timeline of ${points.length} papers from ${formatDate(new Date(axis.startMs).toISOString())} to ${formatDate(new Date(axis.endMs).toISOString())}`}
      className="w-full"
      data-testid="timeline-chart"
      data-axis-unit={axis.unit}
    >
      <line
        x1={PAD_X}
        x2={WIDTH - PAD_X}
        y1={AXIS_Y}
        y2={AXIS_Y}
        stroke="currentColor"
        strokeOpacity={0.4}
      />
      {axis.ticks.map((tick) => {
        const x = PAD_X + tick.at * (WIDTH - 2 * PAD_X);
        return (
          <g key={tick.label + tick.at}>
            <line
              x1={x}
              x2={x}
              y1={AXIS_Y}
              y2={AXIS_Y + 5}
              stroke="currentColor"
              strokeOpacity={0.5}
            />
            <text
              x={x}
              y={AXIS_Y + 20}
              textAnchor="middle"
              fontSize={11}
              fill="currentColor"
              fillOpacity={0.7}
            >
              {tick.label}
            </text>
          </g>
        );
      })}
      {points.map((p, i) => {
        const x = xs[i] ?? PAD_X;
        const y = AXIS_Y - 14 - (lanes[i] ?? 0) * LANE_HEIGHT;
        const dot = (
          <circle
            cx={x}
            cy={y}
            r={p.is_key ? 7 : 4}
            className={
              p.is_key
                ? "fill-primary stroke-background"
                : "fill-muted-foreground/50"
            }
            strokeWidth={p.is_key ? 1.5 : 0}
          >
            <title>{`${formatDate(p.published_at)}: ${p.title}${p.is_key ? " (key paper)" : ""}`}</title>
          </circle>
        );
        return (
          <g key={p.arxiv_id} data-testid="timeline-dot" data-key={p.is_key}>
            {p.is_key ? <a href={`#${entryId(p.arxiv_id)}`}>{dot}</a> : dot}
          </g>
        );
      })}
    </svg>
  );
}

// The time axis of the topic timeline, in plain numbers (the view draws them as SVG).
//
// The corpus is thin in time: a stored range can be a week, or years apart (three old
// papers sit beside the daily windows). So the axis picks its unit from the range:
// days for a narrow one, months for up to about two years, years beyond that.
// Plain TypeScript.

const DAY_MS = 86_400_000;
export const MAX_TICKS = 8;

export type TickUnit = "day" | "month" | "year";
export interface Tick {
  /** 0..1 along the axis */
  at: number;
  label: string;
}
export interface Axis {
  startMs: number;
  endMs: number;
  spanDays: number;
  unit: TickUnit;
  ticks: Tick[];
  /** 0..1 along the axis for a timestamp */
  positionOf: (ms: number) => number;
}

function utcDay(ms: number): number {
  return Math.floor(ms / DAY_MS) * DAY_MS;
}

function label(ms: number, unit: TickUnit): string {
  const d = new Date(ms);
  if (unit === "year") return String(d.getUTCFullYear());
  if (unit === "month")
    return d.toLocaleDateString("en", {
      month: "short",
      year: "numeric",
      timeZone: "UTC",
    });
  return d.toLocaleDateString("en", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

/** The unit for a range: days up to 45 days, months up to 800 days, years beyond. */
export function unitFor(spanDays: number): TickUnit {
  if (spanDays <= 45) return "day";
  if (spanDays <= 800) return "month";
  return "year";
}

function firstOfMonth(ms: number, monthsStep: number): number {
  const d = new Date(ms);
  const index = d.getUTCFullYear() * 12 + d.getUTCMonth();
  let aligned = Math.ceil(index / monthsStep) * monthsStep;
  let first = Date.UTC(Math.floor(aligned / 12), aligned % 12, 1);
  if (first < ms) {
    // the first of the start month is before the start itself: begin one step later
    aligned += monthsStep;
    first = Date.UTC(Math.floor(aligned / 12), aligned % 12, 1);
  }
  return first;
}

function tickTimes(startMs: number, endMs: number, unit: TickUnit): number[] {
  const out: number[] = [];
  if (unit === "day") {
    for (const step of [1, 2, 3, 7, 14]) {
      const times: number[] = [];
      for (let t = startMs; t <= endMs; t += step * DAY_MS) times.push(t);
      if (times.length <= MAX_TICKS) return times;
    }
    return [startMs, endMs];
  }
  if (unit === "month") {
    for (const step of [1, 2, 3, 6, 12]) {
      const times: number[] = [];
      let t = firstOfMonth(startMs, step);
      while (t <= endMs) {
        times.push(t);
        const d = new Date(t);
        t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + step, 1);
      }
      if (times.length <= MAX_TICKS) return times;
    }
    return out;
  }
  for (const step of [1, 2, 5, 10, 20]) {
    const times: number[] = [];
    const first = Math.ceil(new Date(startMs).getUTCFullYear() / step) * step;
    for (let y = first; Date.UTC(y, 0, 1) <= endMs; y += step)
      times.push(Date.UTC(y, 0, 1));
    if (times.length <= MAX_TICKS) return times;
  }
  return out;
}

/**
 * The axis for papers published at `timestamps` (ISO strings). The range runs from the
 * first to the last day; a single day (or none) gets one day on each side so a dot never
 * sits on the edge or divides by zero.
 */
export function buildAxis(timestamps: readonly string[]): Axis {
  const times = timestamps.map((t) => Date.parse(t)).filter(Number.isFinite);
  const today = utcDay(Date.now());
  let startMs = times.length > 0 ? utcDay(Math.min(...times)) : today;
  let endMs = times.length > 0 ? utcDay(Math.max(...times)) : today;
  if (endMs === startMs) {
    startMs -= DAY_MS;
    endMs += DAY_MS;
  }
  const spanDays = Math.round((endMs - startMs) / DAY_MS);
  const unit = unitFor(spanDays);
  const length = endMs - startMs + DAY_MS; // the last day has a width of its own
  const positionOf = (ms: number) =>
    Math.min(1, Math.max(0, (ms - startMs) / length));
  const ticks = tickTimes(startMs, endMs, unit).map((t) => ({
    at: positionOf(t),
    label: label(t, unit),
  }));
  return { startMs, endMs, spanDays, unit, ticks, positionOf };
}

/**
 * Rows ("lanes") for dots that would overlap. Dots are taken left to right; each goes to
 * the first lane whose last dot is at least `minGap` (a fraction of the axis) to its left.
 * Returns one lane index per input, in the input order.
 */
export function assignLanes(
  positions: readonly number[],
  minGap: number,
  maxLanes = 6,
): number[] {
  const order = positions
    .map((x, i) => ({ x, i }))
    .sort((a, b) => a.x - b.x || a.i - b.i);
  const lastX: number[] = [];
  const lanes = new Array<number>(positions.length).fill(0);
  for (const { x, i } of order) {
    let lane = lastX.findIndex((last) => x - last >= minGap);
    if (lane === -1)
      lane = lastX.length < maxLanes ? lastX.length : maxLanes - 1;
    lastX[lane] = x;
    lanes[i] = lane;
  }
  return lanes;
}

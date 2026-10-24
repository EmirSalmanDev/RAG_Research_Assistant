// The metrics of the /evals page: which ones the trend chart offers, how each is read
// (a share from 0 to 1, milliseconds, dollars) and which direction is better. Plain
// TypeScript.
//
// The names are the keys the evaluation stores (backend/eval/run.py, eval/answers.py,
// eval/runner.py). Retrieval runs have the retrieval metrics, answer runs the answer
// metrics; a run without a metric is simply left out where that metric is shown.

export type MetricGroup = "retrieval" | "answers";
/** share: a fraction 0..1. ms: milliseconds. usd: dollars. count: a whole number. */
export type MetricUnit = "share" | "ms" | "usd" | "count" | "number";

export interface MetricDef {
  key: string;
  label: string;
  group: MetricGroup;
  unit: MetricUnit;
  better: "higher" | "lower";
}

export const METRICS: readonly MetricDef[] = [
  {
    key: "recall@8",
    label: "Recall@8",
    group: "retrieval",
    unit: "share",
    better: "higher",
  },
  {
    key: "mrr",
    label: "MRR",
    group: "retrieval",
    unit: "share",
    better: "higher",
  },
  {
    key: "ndcg@10",
    label: "nDCG@10",
    group: "retrieval",
    unit: "share",
    better: "higher",
  },
  {
    key: "coverage@8",
    label: "Coverage@8",
    group: "retrieval",
    unit: "share",
    better: "higher",
  },
  {
    key: "faithfulness",
    label: "Faithfulness",
    group: "answers",
    unit: "share",
    better: "higher",
  },
  {
    key: "response_relevancy",
    label: "Response relevancy",
    group: "answers",
    unit: "share",
    better: "higher",
  },
  {
    key: "context_precision",
    label: "Context precision",
    group: "answers",
    unit: "share",
    better: "higher",
  },
  {
    key: "context_recall",
    label: "Context recall",
    group: "answers",
    unit: "share",
    better: "higher",
  },
  {
    key: "citation_accuracy",
    label: "Citation accuracy",
    group: "answers",
    unit: "share",
    better: "higher",
  },
  {
    key: "uncited_sentence_share",
    label: "Uncited sentences",
    group: "answers",
    unit: "share",
    better: "lower",
  },
  {
    key: "refusal_accuracy",
    label: "Refusal accuracy",
    group: "answers",
    unit: "share",
    better: "higher",
  },
  {
    key: "false_refusal_rate",
    label: "False refusals",
    group: "answers",
    unit: "share",
    better: "lower",
  },
  {
    key: "source_coverage",
    label: "Source coverage",
    group: "answers",
    unit: "share",
    better: "higher",
  },
  {
    key: "latency_ms_p50",
    label: "Latency p50",
    group: "answers",
    unit: "ms",
    better: "lower",
  },
  {
    key: "cost_per_answer_usd",
    label: "Cost per answer",
    group: "answers",
    unit: "usd",
    better: "lower",
  },
];

const BY_KEY = new Map(METRICS.map((m) => [m.key, m]));

export function metricDef(key: string): MetricDef | undefined {
  return BY_KEY.get(key);
}

/** The label of a metric; an unknown key is shown as stored. */
export function metricLabel(key: string): string {
  return BY_KEY.get(key)?.label ?? key;
}

/** How a value is read. Unknown keys go by their name (cost_usd, latency_ms, n_...). */
export function unitOf(key: string): MetricUnit {
  const known = BY_KEY.get(key);
  if (known) return known.unit;
  if (/_usd$/.test(key)) return "usd";
  if (/(^|_)ms(_|$)/.test(key)) return "ms";
  if (/^n_|^errors$|_tokens$/.test(key)) return "count";
  return "number";
}

/**
 * Known metrics first (in the order above), then every other key alphabetically. Used
 * for the tables, so the numbers that matter come first and nothing stored is hidden.
 */
export function orderKeys(keys: Iterable<string>): string[] {
  const unique = [...new Set(keys)];
  const rank = (k: string) => {
    const i = METRICS.findIndex((m) => m.key === k);
    return i === -1 ? METRICS.length : i;
  };
  return unique.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

/** A stored value as text: "0.912", "2,100 ms", "$0.004100"; a dash when there is none. */
export function formatValue(
  key: string,
  value: number | null | undefined,
): string {
  if (value == null || !Number.isFinite(value)) return "—";
  switch (unitOf(key)) {
    case "share":
      return value.toFixed(3);
    case "ms":
      return `${Math.round(value).toLocaleString("en")} ms`;
    case "usd":
      // a cost per answer is a fraction of a cent: show enough digits to compare
      return `$${value.toFixed(Math.abs(value) < 0.01 && value !== 0 ? 6 : 4)}`;
    case "count":
      return Number.isInteger(value) ? String(value) : value.toFixed(1);
    default:
      return Number.isInteger(value)
        ? String(value)
        : String(Number(value.toFixed(4)));
  }
}

const EPSILON = 1e-9; // differences below this are float noise, not a change

/** B - A with its sign: "+0.050", "−0.020", "±0.000"; a dash when either is missing. */
export function formatDiff(
  key: string,
  diff: number | null | undefined,
): string {
  if (diff == null || !Number.isFinite(diff)) return "—";
  if (Math.abs(diff) < EPSILON) return `±${formatValue(key, 0)}`;
  const sign = diff > 0 ? "+" : "−"; // a real minus sign, so the column lines up
  return `${sign}${formatValue(key, Math.abs(diff))}`;
}

export type Direction = "better" | "worse" | "same";

/** Whether a difference is an improvement; null for a metric without a known direction. */
export function direction(
  key: string,
  diff: number | null | undefined,
): Direction | null {
  const def = BY_KEY.get(key);
  if (!def || diff == null || !Number.isFinite(diff)) return null;
  if (Math.abs(diff) < EPSILON) return "same";
  const up = diff > 0;
  return up === (def.better === "higher") ? "better" : "worse";
}

/** The difference B - A, or null when either value is missing. */
export function difference(
  a: number | null | undefined,
  b: number | null | undefined,
): number | null {
  if (a == null || b == null || !Number.isFinite(a) || !Number.isFinite(b))
    return null;
  return b - a;
}

/**
 * A stored number, as the API reads it (backend/app/api/evals.py, _number): numbers and
 * numeric strings (costs are stored as decimal strings); null for anything else, booleans
 * included.
 */
export function toNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

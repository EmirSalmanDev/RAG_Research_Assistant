// The judge's verdicts of one question, shaped for display. Plain TypeScript.
//
// The verdicts are stored as the evaluation wrote them (backend/eval/answers.py): for
// faithfulness a list of {statement, supported, reason}, for citation accuracy an object
// with a list of items and a count, for response relevancy the generated questions and a
// flag. The shapes can change from one evaluation version to the next, so nothing here
// depends on a metric's name: a list of flat objects becomes a table, a list of plain
// values a list, a small object a set of fields, and anything else is shown as JSON.

import { orderKeys } from "./metrics.ts";

export interface Cell {
  text: string;
  /** a yes/no verdict (supported, useful, attributed), so the view can mark it */
  verdict?: "yes" | "no";
}

export type JudgeView =
  | { kind: "text"; text: string }
  | { kind: "list"; items: Cell[] }
  | { kind: "table"; columns: string[]; rows: Cell[][] }
  | { kind: "fields"; fields: { key: string; view: JudgeView }[] }
  | { kind: "json"; text: string };

type Primitive = string | number | boolean | null;

function isPrimitive(value: unknown): value is Primitive {
  return (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** One value as a cell: booleans read as yes / no. */
export function cell(value: unknown): Cell {
  if (value === true) return { text: "yes", verdict: "yes" };
  if (value === false) return { text: "no", verdict: "no" };
  if (value === null || value === undefined) return { text: "—" };
  if (typeof value === "number")
    return {
      text: Number.isInteger(value)
        ? String(value)
        : String(Number(value.toFixed(4))),
    };
  if (typeof value === "string") return { text: value };
  return { text: JSON.stringify(value) };
}

function json(value: unknown): JudgeView {
  let text: string;
  try {
    text = JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    text = String(value); // a cycle cannot come from JSON, but never throw while rendering
  }
  return { kind: "json", text };
}

/**
 * How to show one stored value. `depth` limits the nesting: an object inside an object
 * inside the verdicts is shown as JSON rather than as ever smaller fields.
 */
export function judgeView(value: unknown, depth = 0): JudgeView {
  if (isPrimitive(value) || value === undefined)
    return { kind: "text", text: cell(value).text };
  if (Array.isArray(value)) {
    if (value.length === 0) return { kind: "text", text: "(none)" };
    if (value.every(isPrimitive))
      return { kind: "list", items: value.map(cell) };
    if (
      value.every(
        (v) => isPlainObject(v) && Object.values(v).every(isPrimitive),
      )
    ) {
      const columns: string[] = [];
      for (const row of value as Record<string, Primitive>[])
        for (const key of Object.keys(row))
          if (!columns.includes(key)) columns.push(key);
      return {
        kind: "table",
        columns,
        rows: (value as Record<string, Primitive>[]).map((row) =>
          columns.map((c) => (c in row ? cell(row[c]) : { text: "" })),
        ),
      };
    }
    return json(value);
  }
  if (isPlainObject(value)) {
    const keys = Object.keys(value);
    if (keys.length === 0) return { kind: "text", text: "(none)" };
    if (depth >= 1) return json(value);
    return {
      kind: "fields",
      fields: keys.map((key) => ({
        key,
        view: judgeView(value[key], depth + 1),
      })),
    };
  }
  return json(value);
}

/** The verdicts per metric, the answer metrics first (faithfulness, relevancy, ...). */
export function judgeSections(
  judge: Readonly<Record<string, unknown>> | null | undefined,
): { key: string; view: JudgeView }[] {
  if (!isPlainObject(judge)) return [];
  return orderKeys(Object.keys(judge)).map((key) => ({
    key,
    view: judgeView(judge[key]),
  }));
}

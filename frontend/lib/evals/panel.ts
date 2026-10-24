// Which run's result the question panel shows. A question of the comparison can be in
// run A, run B or both; the panel opens on A when it can and switches between the two.
// Plain TypeScript.

import type { JoinedRow } from "./compare.ts";

export type Side = "a" | "b";

export interface PanelTarget {
  side: Side;
  runId: number;
  /** eval_items id: GET /evals/runs/{runId}/results/{itemId} */
  itemId: number;
}

/** The runs this question can be opened in, A first. */
export function panelTargets(
  row: JoinedRow,
  runA: number | null,
  runB: number | null,
): PanelTarget[] {
  const out: PanelTarget[] = [];
  if (row.a && runA !== null)
    out.push({ side: "a", runId: runA, itemId: row.a.item_id });
  if (row.b && runB !== null)
    out.push({ side: "b", runId: runB, itemId: row.b.item_id });
  return out;
}

/** The target for `side`, or the first one there is (a question only B has opens on B). */
export function pickTarget(
  targets: readonly PanelTarget[],
  side: Side | null,
): PanelTarget | null {
  return targets.find((t) => t.side === side) ?? targets[0] ?? null;
}

// What the /health page says about the system, worked out from GET /status. Plain
// TypeScript (type-only imports), so Node's test runner can check it without a browser.

import type { StatusOut } from "../api/endpoints.ts";

export type Level = "ok" | "warning" | "down";

export interface Summary {
  level: Level;
  /** One sentence per thing that needs attention, most serious first. */
  problems: string[];
}

const STORE_NAMES: Record<string, string> = {
  postgres: "Postgres",
  qdrant: "Qdrant",
  redis: "Redis",
};

/**
 * "down" when a store is unreachable or the database numbers cannot be read (nothing
 * works without them); "warning" when the system runs but needs attention (the budget
 * cap paused the background jobs, a step failed, a part of this page could not be read).
 */
export function summarize(status: StatusOut): Summary {
  const down: string[] = [];
  const warnings: string[] = [];

  for (const [key, state] of Object.entries(status.stores)) {
    if (state !== "up")
      down.push(`${STORE_NAMES[key] ?? key} is not reachable.`);
  }
  const database = status.database;
  if (database === null) {
    down.push("The database numbers could not be read.");
  } else {
    if (database.budget.paused) {
      warnings.push(
        "The monthly budget cap is reached: background jobs that spend money are paused.",
      );
    }
    const failed = database.last_runs.filter(
      (run) => run.finished_at !== null && !run.ok,
    );
    if (failed.length > 0) {
      warnings.push(
        `${failed.length === 1 ? "A step failed" : `${failed.length} steps failed`} in its last run: ${failed.map((r) => r.source).join(", ")}.`,
      );
    }
    const failedPapers = database.papers_by_fulltext_status["failed"] ?? 0;
    if (failedPapers > 0) {
      warnings.push(
        `${failedPapers} ${failedPapers === 1 ? "paper" : "papers"} could not get full text.`,
      );
    }
  }
  // A part that failed to read, other than the database (already said above).
  for (const [part, reason] of Object.entries(status.errors)) {
    if (part !== "database")
      warnings.push(`${part} could not be read: ${reason}`);
  }

  const problems = [...down, ...warnings];
  const level: Level =
    down.length > 0 ? "down" : warnings.length > 0 ? "warning" : "ok";
  return { level, problems };
}

/** "just now", "5 min ago", "3 h ago", "2 d ago"; the input itself if it is not a date. */
export function formatAge(iso: string, now: Date): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return iso;
  const minutes = Math.floor((now.getTime() - then) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return `${Math.floor(hours / 24)} d ago`;
}

const FULLTEXT_ORDER = [
  "indexed",
  "fetched",
  "requested",
  "none",
  "unavailable",
  "failed",
];

const FULLTEXT_LABEL: Record<string, string> = {
  indexed: "Full text indexed",
  fetched: "Text fetched, not indexed yet",
  requested: "Full text requested",
  none: "Abstract only",
  unavailable: "No full text available",
  failed: "Failed",
};

/** The paper counts as labelled rows in a fixed order; a status this code does not know comes last. */
export function paperRows(
  byStatus: Record<string, number>,
): { status: string; label: string; count: number }[] {
  const known = FULLTEXT_ORDER.filter((s) => s in byStatus);
  const unknown = Object.keys(byStatus)
    .filter((s) => !FULLTEXT_ORDER.includes(s))
    .sort();
  return [...known, ...unknown].map((status) => ({
    status,
    label: FULLTEXT_LABEL[status] ?? status,
    count: byStatus[status] ?? 0,
  }));
}

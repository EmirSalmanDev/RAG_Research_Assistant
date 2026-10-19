// The timeline form: what is allowed to be sent, and the sentences under the result.
// Plain TypeScript. The form never calls the API by itself: the Run button does.

import type { components } from "../api/schema";

export type TimelineBody = components["schemas"]["TimelineBody"];
export type TimelineOut = components["schemas"]["TimelineOut"];

// The API's limits (backend/app/features/timeline.py).
export const TOPIC_MIN = 3;
export const TOPIC_MAX = 200;
export const DEFAULT_MAX_PAPERS = 30;
export const MAX_PAPERS_CAP = 50;
export const DEFAULT_KEY_PAPERS = 8;
export const KEY_PAPERS_CAP = 15;

export interface TimelineForm {
  topic: string;
  from: string; // YYYY-MM-DD or ""
  to: string;
  categories: readonly string[];
  maxPapers: number;
  keyPapers: number;
}

export interface Validated {
  body: TimelineBody | null;
  problems: string[];
}

function wholeNumber(n: number, min: number, max: number): boolean {
  return Number.isInteger(n) && n >= min && n <= max;
}

export function validateTimeline(form: TimelineForm): Validated {
  const problems: string[] = [];
  const topic = form.topic.replace(/\s+/g, " ").trim();
  if (topic.length < TOPIC_MIN)
    problems.push(`The topic needs at least ${TOPIC_MIN} characters.`);
  if (topic.length > TOPIC_MAX)
    problems.push(`The topic is longer than ${TOPIC_MAX} characters.`);
  if (form.from && form.to && form.from > form.to)
    problems.push("The start date is after the end date.");
  if (!wholeNumber(form.maxPapers, 1, MAX_PAPERS_CAP))
    problems.push(`Papers shown must be 1 to ${MAX_PAPERS_CAP}.`);
  if (!wholeNumber(form.keyPapers, 1, KEY_PAPERS_CAP))
    problems.push(`Key papers must be 1 to ${KEY_PAPERS_CAP}.`);
  if (problems.length > 0) return { body: null, problems };
  return {
    body: {
      topic,
      date_from: form.from || null,
      date_to: form.to || null,
      categories: [...form.categories],
      max_papers: form.maxPapers,
      key_papers: form.keyPapers,
    },
    problems,
  };
}

/** Sentences about what the result can and cannot say (shown under the chart). */
export function describeNotes(out: TimelineOut): string[] {
  const { notes } = out;
  const lines: string[] = [];
  if (notes.papers === 0) return lines;
  lines.push(
    `${notes.papers} ${notes.papers === 1 ? "paper" : "papers"}, ${notes.key_papers} marked as key.`,
  );
  if (notes.known_citation_counts < notes.papers) {
    lines.push(
      `A citation count is known for ${notes.known_citation_counts} of ${notes.papers} papers; the others count as zero when key papers are ranked.`,
    );
  } else {
    lines.push(`A citation count is known for all ${notes.papers} papers.`);
  }
  lines.push(
    notes.narrow_range
      ? `The papers span only ${notes.range_days} ${notes.range_days === 1 ? "day" : "days"}, so the axis is by day: the stored papers cover a short time.`
      : `The papers span ${notes.range_days} days.`,
  );
  if (notes.lines_call_failed) {
    lines.push(
      "The model call for the one-line summaries failed; every line below is the first sentence of the abstract.",
    );
  } else if (notes.fallback_lines > 0) {
    lines.push(
      `${notes.fallback_lines} ${notes.fallback_lines === 1 ? "line is" : "lines are"} the first sentence of the abstract because the model's line did not pass the checks.`,
    );
  }
  return lines;
}

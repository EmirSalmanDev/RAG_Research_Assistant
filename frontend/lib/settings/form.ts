// The settings page's forms: turning what was typed into a request body, with the problems
// listed in plain words. The same rules the API enforces (a request that breaks them is a
// 422 there), checked first so the reader sees what to fix. Plain TypeScript.

import type { InterestIn, SettingsIn } from "../api/endpoints.ts";

export const NAME_MAX = 100;
export const DESCRIPTION_MAX = 1000;
export const KEYWORD_MAX = 60;
export const KEYWORDS_MAX = 30;
export const CATEGORIES_MAX = 30;
export const CATEGORY_MAX = 32;
export const DIGEST_SIZE_RANGE = [1, 100] as const;
export const TOP_N_RANGE = [0, 100] as const;
export const BUDGET_MAX = 100_000;

// arXiv category ids: "cs.CL", "stat.ML", "hep-th", "astro-ph.GA", "physics.comp-ph".
const CATEGORY = /^[a-z]+(-[a-z]+)*(\.[A-Za-z]+(-[A-Za-z]+)*)?$/;

export function isCategory(value: string): boolean {
  return value.length <= CATEGORY_MAX && CATEGORY.test(value);
}

/** "a, b\nc;d" -> ["a", "b", "c", "d"]: trimmed, no blanks, no repeats (first spelling kept). */
export function splitList(text: string): string[] {
  const seen = new Set<string>();
  const items: string[] = [];
  for (const part of text.split(/[,\n;]/)) {
    const item = part.trim().replace(/\s+/g, " ");
    if (item !== "" && !seen.has(item.toLowerCase())) {
      seen.add(item.toLowerCase());
      items.push(item);
    }
  }
  return items;
}

export type Parsed<T> =
  { ok: true; body: T } | { ok: false; problems: string[] };

export interface InterestFields {
  name: string;
  description: string;
  keywords: string;
  categories: string;
  isActive: boolean;
}

export function parseInterest(fields: InterestFields): Parsed<InterestIn> {
  const problems: string[] = [];
  const name = fields.name.trim().replace(/\s+/g, " ");
  if (name === "") problems.push("Give the interest a name.");
  if (name.length > NAME_MAX)
    problems.push(`The name is longer than ${NAME_MAX} characters.`);
  const description = fields.description.trim();
  if (description.length > DESCRIPTION_MAX)
    problems.push(
      `The description is longer than ${DESCRIPTION_MAX} characters.`,
    );
  const keywords = splitList(fields.keywords);
  if (keywords.length > KEYWORDS_MAX)
    problems.push(`Use at most ${KEYWORDS_MAX} keywords.`);
  const long = keywords.find((k) => k.length > KEYWORD_MAX);
  if (long)
    problems.push(
      `The keyword "${long.slice(0, 20)}…" is longer than ${KEYWORD_MAX} characters.`,
    );
  const categories = splitList(fields.categories);
  if (categories.length > CATEGORIES_MAX)
    problems.push(`Use at most ${CATEGORIES_MAX} categories.`);
  const bad = categories.find((c) => !isCategory(c));
  if (bad)
    problems.push(
      `"${bad.slice(0, 40)}" is not an arXiv category (like cs.CL or hep-th).`,
    );
  if (problems.length > 0) return { ok: false, problems };
  return {
    ok: true,
    body: {
      name,
      description: description === "" ? null : description,
      keywords,
      categories,
      is_active: fields.isActive,
    },
  };
}

export interface SettingsFields {
  digestSize: string;
  fulltextTopN: string;
  categories: string;
  budgetCap: string;
}

function wholeNumber(text: string): number | null {
  return /^\d+$/.test(text.trim()) ? Number(text.trim()) : null;
}

export function parseSettings(fields: SettingsFields): Parsed<SettingsIn> {
  const problems: string[] = [];
  const size = wholeNumber(fields.digestSize);
  const [sizeMin, sizeMax] = DIGEST_SIZE_RANGE;
  if (size === null || size < sizeMin || size > sizeMax)
    problems.push(
      `Digest size must be a whole number from ${sizeMin} to ${sizeMax}.`,
    );
  const topN = wholeNumber(fields.fulltextTopN);
  const [topMin, topMax] = TOP_N_RANGE;
  if (topN === null || topN < topMin || topN > topMax)
    problems.push(
      `Full-text papers per night must be a whole number from ${topMin} to ${topMax}.`,
    );
  const categories = splitList(fields.categories);
  if (categories.length > CATEGORIES_MAX)
    problems.push(`Use at most ${CATEGORIES_MAX} categories.`);
  const bad = categories.find((c) => !isCategory(c));
  if (bad)
    problems.push(
      `"${bad.slice(0, 40)}" is not an arXiv category (like cs.CL or hep-th).`,
    );
  let budget: number | null = null;
  const budgetText = fields.budgetCap.trim().replace(/^\$/, "");
  if (budgetText !== "") {
    budget = /^\d+(\.\d{1,2})?$/.test(budgetText) ? Number(budgetText) : NaN;
    if (!Number.isFinite(budget) || budget > BUDGET_MAX) {
      problems.push(
        `The budget cap must be an amount in dollars from 0 to ${BUDGET_MAX} (or empty for none).`,
      );
      budget = null;
    }
  }
  if (problems.length > 0 || size === null || topN === null)
    return { ok: false, problems };
  return {
    ok: true,
    body: {
      digest_size: size,
      fulltext_top_n: topN,
      categories,
      budget_cap_usd: budget,
    },
  };
}

/** "$0.0450" for the month's spend: small amounts keep four decimals. */
export function formatUsd(amount: number): string {
  return `$${amount.toFixed(amount >= 100 ? 2 : 4)}`;
}

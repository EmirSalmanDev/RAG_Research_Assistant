import { formatDate } from "../format.ts";
import type { PlanMeta } from "./events.ts";

// The filters the query plan applied while searching, as short labels for the chips above
// an answer. The planner's dates arrive as full timestamps (a start at 00:00 and an end at
// 23:59:59.999999, both UTC); the reader only needs the day.
export function planChips(plan: PlanMeta): string[] {
  const chips: string[] = [];
  if (plan.intent !== "qa")
    chips.push(`Mode: ${plan.intent.replace("_", " ")}`);
  for (const id of plan.arxiv_ids) chips.push(`Paper ${id}`);
  for (const category of plan.categories) chips.push(category);
  if (plan.published_from)
    chips.push(`From ${formatDate(plan.published_from)}`);
  if (plan.published_to) chips.push(`Until ${formatDate(plan.published_to)}`);
  return chips;
}

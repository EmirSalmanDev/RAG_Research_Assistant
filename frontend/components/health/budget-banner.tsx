"use client";

import Link from "next/link";

import { useStatus } from "@/lib/api/queries";
import { formatUsd } from "@/lib/settings/form";

/**
 * Shown on every page while the monthly budget cap has paused the background jobs. It
 * says nothing while the cap is not reached, and nothing when the status cannot be read
 * (the /health page is where that is reported).
 */
export function BudgetBanner() {
  const budget = useStatus().data?.database?.budget;
  if (!budget?.paused) return null;
  return (
    <div
      role="status"
      data-testid="budget-banner"
      className="border-b border-amber-600/40 bg-amber-500/10 px-4 py-2 text-center text-sm text-amber-800 dark:text-amber-300"
    >
      The monthly budget cap is reached ({formatUsd(budget.spent_usd)} of{" "}
      {budget.cap_usd === null ? "?" : formatUsd(budget.cap_usd)}). Background
      jobs that spend money are paused; chat and search still work.{" "}
      <Link href="/settings" className="underline underline-offset-4">
        Change the cap
      </Link>
    </div>
  );
}

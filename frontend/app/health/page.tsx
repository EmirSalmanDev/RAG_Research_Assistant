import type { Metadata } from "next";

import { HealthView } from "@/components/health/health-view";

export const metadata: Metadata = {
  title: "Health | AI Research Copilot",
  description:
    "Stores, nightly pipeline runs, paper counts, queue and this month's spending.",
};

export default function HealthPage() {
  return <HealthView />;
}

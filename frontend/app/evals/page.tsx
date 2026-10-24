import type { Metadata } from "next";

import { EvalsView } from "@/components/evals/evals-view";

export const metadata: Metadata = {
  title: "Evaluations | AI Research Copilot",
  description:
    "Stored evaluation runs: metric trends, two runs side by side, and each question.",
};

export default function EvalsPage() {
  return <EvalsView />;
}

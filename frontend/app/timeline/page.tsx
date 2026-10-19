import type { Metadata } from "next";

import { TimelineView } from "@/components/timeline/timeline-view";

export const metadata: Metadata = { title: "Timeline | AI Research Copilot" };

export default function TimelinePage() {
  return <TimelineView />;
}

import type { Metadata } from "next";
import { Suspense } from "react";

import { DigestView } from "@/components/digest/digest-view";

export const metadata: Metadata = {
  title: "Digest | AI Research Copilot",
  description: "Today's new papers, ranked against your interests.",
};

export default function Home() {
  // useSearchParams (in DigestView, for ?date=) needs a Suspense boundary above it.
  return (
    <Suspense fallback={null}>
      <DigestView />
    </Suspense>
  );
}

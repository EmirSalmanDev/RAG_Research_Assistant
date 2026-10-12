import type { Metadata } from "next";
import { Suspense } from "react";

import { SearchView } from "@/components/search/search-view";

export const metadata: Metadata = { title: "Search | AI Research Copilot" };

export default function SearchPage() {
  // useSearchParams (in SearchView) needs a Suspense boundary above it.
  return (
    <Suspense fallback={null}>
      <SearchView />
    </Suspense>
  );
}

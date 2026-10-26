"use client";

import { useEffect } from "react";

import { ErrorState } from "@/components/ui/error-state";

// Shown when a page throws while rendering (a bug, not a failed request: failed requests
// are handled inside each view). In this Next version the recovery function is `retry`,
// which re-fetches and re-renders the segment; `reset` would only clear the error.
export default function RouteError({
  error,
  retry,
}: {
  error: Error & { digest?: string };
  retry: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);
  return (
    <div className="mx-auto w-full max-w-3xl space-y-3 px-4 py-8">
      <h1 className="text-2xl font-semibold tracking-tight">
        Something went wrong
      </h1>
      <ErrorState error={error} onRetry={retry} />
    </div>
  );
}

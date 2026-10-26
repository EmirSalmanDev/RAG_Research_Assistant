import { LoadingState } from "@/components/ui/loading-state";

// The fallback while a route segment streams in (navigation between pages).
export default function Loading() {
  return (
    <LoadingState
      label="Loading the page…"
      className="mx-auto w-full max-w-3xl px-4 py-8"
    />
  );
}

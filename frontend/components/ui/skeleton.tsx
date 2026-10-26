import { cn } from "@/lib/utils";

/** A grey bar that pulses: the shape of something that is still loading. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={cn("animate-pulse rounded-md bg-muted", className)}
    />
  );
}

import { cn } from "@/lib/utils";

import { Skeleton } from "./skeleton";

/**
 * The loading state every view uses: the sentence is real text (screen readers and tests
 * read it through role="status"), the bars only show that something is on its way.
 */
export function LoadingState({
  label,
  lines = 3,
  className,
}: {
  label: string;
  lines?: number;
  className?: string;
}) {
  return (
    <div role="status" className={cn("space-y-3", className)}>
      <p className="text-sm text-muted-foreground">{label}</p>
      <div className="space-y-2">
        {Array.from({ length: lines }, (_, i) => (
          <Skeleton key={i} className={i === lines - 1 ? "h-4 w-2/3" : "h-4"} />
        ))}
      </div>
    </div>
  );
}

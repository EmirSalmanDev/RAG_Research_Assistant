import { describeError } from "@/lib/api/errors";
import { cn } from "@/lib/utils";

import { Button } from "./button";

/**
 * The error state every view uses. `error` is whatever was thrown (see describeError);
 * `what` names the thing that failed ("The digest"), so the line reads "The digest could
 * not be loaded: ...". With `onRetry` there is a button that asks again.
 */
export function ErrorState({
  error,
  what,
  onRetry,
  className,
}: {
  error: unknown;
  what?: string;
  onRetry?: () => void;
  className?: string;
}) {
  const reason = describeError(error);
  return (
    <div
      role="alert"
      data-testid="error-state"
      className={cn(
        "space-y-2 rounded-md border border-destructive/40 p-3 text-sm text-destructive",
        className,
      )}
    >
      <p>{what ? `${what} could not be loaded: ${reason}` : reason}</p>
      {onRetry && (
        <Button type="button" variant="outline" size="sm" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

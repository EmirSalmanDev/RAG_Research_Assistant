import { runState, STATE_LABEL, stopReason } from "@/lib/evals/runs";
import type { RunState } from "@/lib/evals/runs";
import { cn } from "@/lib/utils";

const STATE_CLASS: Record<RunState, string> = {
  complete: "border-border text-muted-foreground",
  stopped: "border-destructive/50 bg-destructive/10 text-destructive",
  running:
    "border-amber-500/60 bg-amber-500/10 text-amber-700 dark:text-amber-300",
  unknown: "border-dashed text-muted-foreground",
};

// A run's state as a label (never colour alone). Only "complete" is a finished result:
// a stopped run's numbers cover the questions done before it stopped, a running one's
// are not written yet.
export function RunStatus({
  status,
  className,
}: {
  status: string | null;
  className?: string;
}) {
  const state = runState(status);
  const reason = stopReason(status);
  return (
    <span
      data-testid="run-status"
      data-state={state}
      title={status ?? "No status was stored with this run."}
      className={cn(
        "inline-flex items-center whitespace-nowrap rounded border px-1.5 py-0.5 text-xs font-medium",
        STATE_CLASS[state],
        className,
      )}
    >
      {STATE_LABEL[state]}
      {reason && (
        <span className="ml-1 max-w-[14rem] truncate font-normal">
          ({reason})
        </span>
      )}
      {state === "running" && (
        <span className="ml-1 font-normal">(not finished)</span>
      )}
    </span>
  );
}

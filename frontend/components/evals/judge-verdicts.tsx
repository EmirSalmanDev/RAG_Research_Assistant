import { judgeSections } from "@/lib/evals/judge";
import type { Cell, JudgeView } from "@/lib/evals/judge";
import { metricLabel } from "@/lib/evals/metrics";
import { cn } from "@/lib/utils";

function CellText({ cell }: { cell: Cell }) {
  if (!cell.verdict) return <>{cell.text}</>;
  return (
    <span
      className={cn(
        "rounded px-1 text-xs font-medium",
        cell.verdict === "yes"
          ? "bg-emerald-500/15 text-emerald-800 dark:text-emerald-300"
          : "bg-destructive/10 text-destructive",
      )}
    >
      {cell.text}
    </span>
  );
}

// One stored value, drawn by its shape (lib/evals/judge.ts decides the shape): a table
// for a list of verdicts, a list, fields of a small object, or the raw JSON.
export function JudgeValue({ view }: { view: JudgeView }) {
  switch (view.kind) {
    case "text":
      return <p className="text-sm">{view.text}</p>;
    case "list":
      return (
        <ul className="list-disc space-y-0.5 pl-5 text-sm">
          {view.items.map((item, i) => (
            <li key={i}>
              <CellText cell={item} />
            </li>
          ))}
        </ul>
      );
    case "table":
      return (
        <div className="overflow-x-auto rounded border">
          <table className="w-full border-collapse text-left text-xs">
            <thead>
              <tr className="border-b bg-muted/50">
                {view.columns.map((c) => (
                  <th key={c} scope="col" className="p-1.5 font-medium">
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {view.rows.map((row, i) => (
                <tr key={i} className="border-b align-top last:border-b-0">
                  {row.map((cell, j) => (
                    <td key={j} className="p-1.5 leading-5">
                      <CellText cell={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "fields":
      return (
        <dl className="space-y-2">
          {view.fields.map((field) => (
            <div key={field.key} className="space-y-1">
              <dt className="text-xs font-medium text-muted-foreground">
                {field.key}
              </dt>
              <dd>
                <JudgeValue view={field.view} />
              </dd>
            </div>
          ))}
        </dl>
      );
    case "json":
      return (
        <pre className="overflow-x-auto rounded bg-muted p-2 text-xs">
          {view.text}
        </pre>
      );
  }
}

/** Every metric's verdicts, the answer metrics first. */
export function JudgeVerdicts({ judge }: { judge: Record<string, unknown> }) {
  const sections = judgeSections(judge);
  if (sections.length === 0)
    return (
      <p className="text-sm text-muted-foreground">
        No verdicts were stored for this question (a retrieval run, a failed
        turn, or a run from before verdicts were kept).
      </p>
    );
  return (
    <div className="space-y-4">
      {sections.map(({ key, view }) => (
        <section
          key={key}
          aria-label={`Verdicts: ${metricLabel(key)}`}
          data-testid="judge-section"
          data-metric={key}
          className="space-y-1.5"
        >
          <h4 className="text-sm font-medium">{metricLabel(key)}</h4>
          <JudgeValue view={view} />
        </section>
      ))}
    </div>
  );
}

"use client";

import { Badge } from "@/components/ui/badge";
import type { PlanMeta, Source } from "@/lib/chat/events";
import { planChips } from "@/lib/chat/plan";
import type { AssistantMessage, ChatMessage } from "@/lib/chat/reducer";

import { AnswerText } from "./answer-text";

// The filters the query plan applied while searching, so the reader can see why some
// papers were left out. Shown only when there is something to show.
function PlanChips({ plan }: { plan: PlanMeta }) {
  const chips = planChips(plan);
  if (chips.length === 0) return null;
  return (
    <ul
      aria-label="Filters applied to the search"
      className="flex flex-wrap gap-1.5"
    >
      {chips.map((chip) => (
        <li key={chip}>
          <Badge variant="outline" className="font-normal">
            {chip}
          </Badge>
        </li>
      ))}
    </ul>
  );
}

function verificationNotes(message: AssistantMessage): string[] {
  const v = message.verification;
  if (!v || v.status === "skipped") return [];
  const notes: string[] = [];
  if (v.status === "unavailable")
    notes.push("The automatic support check did not run.");
  if (v.uncited_sentences.length > 0) {
    const n = v.uncited_sentences.length;
    notes.push(
      `${n} ${n === 1 ? "sentence has" : "sentences have"} no citation.`,
    );
  }
  if (v.unsupported_sentences.length > 0) {
    const n = v.unsupported_sentences.length;
    notes.push(
      `${n} ${n === 1 ? "sentence is" : "sentences are"} not clearly supported by its source.`,
    );
  }
  if (v.invalid_citations.length > 0) {
    notes.push(
      `Cited source ${v.invalid_citations.join(", ")} does not exist.`,
    );
  }
  return notes;
}

function Assistant({
  message,
  activeChunkId,
  onOpen,
}: {
  message: AssistantMessage;
  activeChunkId?: string;
  onOpen: (source: Source, all: readonly Source[]) => void;
}) {
  const streaming = message.status === "streaming";
  const open = (source: Source) => onOpen(source, message.sources);
  const notes = verificationNotes(message);
  const cost = message.done ? Number(message.done.cost_usd) : null;

  return (
    <div className="space-y-3">
      {message.meta && <PlanChips plan={message.meta.plan} />}

      {message.refusal ? (
        <p className="font-serif text-[1.0625rem] leading-8 text-muted-foreground">
          {message.refusal.text}
        </p>
      ) : message.text || streaming ? (
        <AnswerText
          text={message.text}
          sources={message.sources}
          streaming={streaming}
          activeChunkId={activeChunkId}
          onOpen={open}
        />
      ) : null}

      {streaming && !message.text && !message.refusal && (
        <p role="status" className="text-sm text-muted-foreground">
          {message.sources.length > 0
            ? `Reading ${message.sources.length} sources`
            : message.meta
              ? "Searching the papers"
              : "Working on it"}
          &hellip;
        </p>
      )}

      {message.status === "stopped" && (
        <p className="text-sm text-muted-foreground">
          Stopped. The answer above is incomplete.
        </p>
      )}
      {message.status === "error" && message.error && (
        <p
          role="alert"
          className="rounded-md border border-destructive/40 p-3 text-sm text-destructive"
        >
          {message.error}
        </p>
      )}

      {message.done?.abstract_only && (
        <p className="text-sm text-muted-foreground">
          Based on abstracts only. The full text of the cited papers is not
          indexed.
        </p>
      )}
      {notes.length > 0 && (
        <ul className="space-y-0.5 text-sm text-muted-foreground">
          {notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      {message.sources.length > 0 && !streaming && !message.refusal && (
        <details className="text-sm">
          <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
            All {message.sources.length} sources
          </summary>
          <ul className="mt-2 space-y-1">
            {message.sources.map((source) => (
              <li key={source.chunk_id}>
                <button
                  type="button"
                  onClick={() => open(source)}
                  className="flex w-full items-baseline gap-2 rounded-md px-2 py-1 text-left transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <span className="w-5 shrink-0 font-medium">{source.n}</span>
                  <span className="flex-1">{source.title}</span>
                  {source.chunk_type === "abstract" && (
                    <span className="shrink-0 text-xs text-muted-foreground">
                      abstract
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}

      {message.done && cost !== null && Number.isFinite(cost) && (
        <p className="text-xs text-muted-foreground">
          {message.done.tokens_in + message.done.tokens_out} tokens, $
          {cost.toFixed(4)}
        </p>
      )}
    </div>
  );
}

export function MessageView({
  message,
  activeChunkId,
  onOpen,
}: {
  message: ChatMessage;
  activeChunkId?: string;
  onOpen: (source: Source, all: readonly Source[]) => void;
}) {
  if (message.role === "user") {
    return (
      <h2 className="text-lg font-semibold leading-snug tracking-tight">
        {message.text}
      </h2>
    );
  }
  return (
    <Assistant
      message={message}
      activeChunkId={activeChunkId}
      onOpen={onOpen}
    />
  );
}

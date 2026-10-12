// Invented payloads shaped like the backend's events (see events.ts). Used by the tests.
import type { Source } from "./events.ts";

export const ABSTRACT_SOURCE: Source = {
  n: 1,
  chunk_id: "00000000-0000-4000-8000-0000000000a1",
  arxiv_id: "2609.00001",
  title: "An Invented Paper About Adapters",
  section_path: null,
  chunk_type: "abstract",
  published_at: "2026-09-01T12:00:00+00:00",
  abs_url: "https://arxiv.org/abs/2609.00001",
};

export const SECTION_SOURCE: Source = {
  n: 2,
  chunk_id: "00000000-0000-4000-8000-0000000000a2",
  arxiv_id: "2609.00002",
  title: "A Second Invented Paper",
  section_path: "Method > Training",
  chunk_type: "section",
  published_at: "2026-09-02T12:00:00+00:00",
  abs_url: "https://arxiv.org/abs/2609.00002",
};

export const PLAN = {
  intent: "qa",
  rephrasings: ["how do adapters reduce trainable parameters"],
  arxiv_ids: [],
  categories: ["cs.CL"],
  published_from: "2026-09-01",
  published_to: null,
  dropped: [],
};

export const META = {
  conversation_id: "00000000-0000-4000-8000-0000000000c0",
  new_conversation: true,
  intent: "qa",
  standalone_question:
    "How do adapters reduce the number of trained parameters?",
  plan: PLAN,
};

export const VERIFICATION = {
  status: "checked",
  sentences: [
    {
      index: 0,
      text: "Adapters train a small module [1].",
      citations: [1],
      invalid_citations: [],
      supported: true,
      reason: "",
    },
  ],
  invalid_citations: [],
  uncited_sentences: [],
  unsupported_sentences: [],
  reason: "",
  prompt: "verify_v1",
};

export const DONE = {
  conversation_id: META.conversation_id,
  message_id: 12,
  refusal: null,
  abstract_only: true,
  citations: [ABSTRACT_SOURCE],
  tokens_in: 900,
  tokens_out: 60,
  cost_usd: "0.000171",
  trace_id: null,
};

/** One SSE block, as FastAPI writes it. */
export function sse(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** A complete, successful turn as raw SSE text, split into the pieces a model streams. */
export function fullTurn(): string {
  return (
    sse("meta", META) +
    sse("sources", { sources: [ABSTRACT_SOURCE, SECTION_SOURCE] }) +
    sse("token", { text: "Adapters train a " }) +
    sse("token", { text: "small module [1]. " }) +
    sse("token", { text: "They also freeze the base model [2]." }) +
    sse("verification", VERIFICATION) +
    sse("done", DONE)
  );
}

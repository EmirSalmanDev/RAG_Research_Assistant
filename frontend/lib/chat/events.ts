// The events POST /chat streams, as TypeScript types, and a checked parser for them.
// OpenAPI cannot describe server-sent events, so these mirror the backend by hand: the
// payload of every event is built in backend/app/rag/pipeline.py (and `source_object` in
// backend/app/rag/citations.py). The parser checks each payload instead of trusting a
// cast, so a backend change fails loudly here rather than as `undefined` in the UI.
//
// This file is plain TypeScript with no imports: Node's test runner runs it as is.

export type Intent = "qa" | "explain" | "compare" | "find_papers";
export type ChunkType = "abstract" | "section" | "analysis";
export type RefusalReason = "no_sources" | "insufficient_sources";
export type VerifyStatus = "checked" | "skipped" | "unavailable";

/** The checked query plan that did the searching (filters become chips in the UI). */
export interface PlanMeta {
  intent: Intent;
  rephrasings: string[];
  arxiv_ids: string[];
  categories: string[];
  published_from: string | null;
  published_to: string | null;
  dropped: string[];
}

/** One numbered source [n] given to the model. It carries no chunk text: GET /chunks/{id}. */
export interface Source {
  n: number;
  chunk_id: string;
  arxiv_id: string;
  title: string;
  section_path: string | null;
  chunk_type: ChunkType;
  published_at: string;
  abs_url: string | null; // null: an uploaded document has no arXiv page
}

export interface MetaEvent {
  conversation_id: string;
  new_conversation: boolean;
  intent: Intent;
  standalone_question: string;
  plan: PlanMeta;
}

export interface SentenceCheck {
  index: number;
  text: string;
  citations: number[];
  invalid_citations: number[];
  supported: boolean | null;
  reason: string;
}

export interface Verification {
  status: VerifyStatus;
  sentences: SentenceCheck[];
  invalid_citations: number[];
  uncited_sentences: number[];
  unsupported_sentences: number[];
  reason: string;
}

export interface RefusalEvent {
  reason: RefusalReason;
  text: string;
}

export interface DoneEvent {
  conversation_id: string;
  message_id: number;
  refusal: RefusalReason | null;
  abstract_only: boolean;
  citations: Source[];
  tokens_in: number;
  tokens_out: number;
  cost_usd: string;
  trace_id: string | null;
}

export interface ErrorEvent {
  stage: string;
  message: string;
}

export type ChatEvent =
  | { event: "meta"; data: MetaEvent }
  | { event: "sources"; data: { sources: Source[] } }
  | { event: "token"; data: { text: string } }
  | { event: "refusal"; data: RefusalEvent }
  | { event: "verification"; data: Verification }
  | { event: "done"; data: DoneEvent }
  | { event: "error"; data: ErrorEvent };

/** The stream is not what the client expects (bad JSON, missing field, cut short). */
export class ChatProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ChatProtocolError";
  }
}

// ---- small checked readers: each returns a value of the right type or throws ----

type Obj = Record<string, unknown>;

function fail(path: string, want: string): never {
  throw new ChatProtocolError(`bad chat event: ${path} should be ${want}`);
}

function obj(v: unknown, path: string): Obj {
  if (typeof v !== "object" || v === null || Array.isArray(v))
    fail(path, "an object");
  return v as Obj;
}

function str(o: Obj, k: string, path: string): string {
  const v = o[k];
  if (typeof v !== "string") fail(`${path}.${k}`, "a string");
  return v;
}

function strOrNull(o: Obj, k: string, path: string): string | null {
  const v = o[k];
  if (v === null || v === undefined) return null;
  if (typeof v !== "string") fail(`${path}.${k}`, "a string or null");
  return v;
}

function num(o: Obj, k: string, path: string): number {
  const v = o[k];
  if (typeof v !== "number" || !Number.isFinite(v))
    fail(`${path}.${k}`, "a number");
  return v;
}

function bool(o: Obj, k: string, path: string): boolean {
  const v = o[k];
  if (typeof v !== "boolean") fail(`${path}.${k}`, "a boolean");
  return v;
}

function arr(o: Obj, k: string, path: string): unknown[] {
  const v = o[k];
  if (!Array.isArray(v)) fail(`${path}.${k}`, "an array");
  return v;
}

function strArr(o: Obj, k: string, path: string): string[] {
  return arr(o, k, path).map((v, i) => {
    if (typeof v !== "string") fail(`${path}.${k}[${i}]`, "a string");
    return v;
  });
}

function numArr(o: Obj, k: string, path: string): number[] {
  return arr(o, k, path).map((v, i) => {
    if (typeof v !== "number") fail(`${path}.${k}[${i}]`, "a number");
    return v;
  });
}

function oneOf<T extends string>(
  o: Obj,
  k: string,
  path: string,
  allowed: readonly T[],
): T {
  const v = str(o, k, path);
  const hit = allowed.find((a) => a === v);
  if (hit === undefined) fail(`${path}.${k}`, `one of ${allowed.join(", ")}`);
  return hit;
}

const INTENTS = ["qa", "explain", "compare", "find_papers"] as const;
const CHUNK_TYPES = ["abstract", "section", "analysis"] as const;
const REFUSALS = ["no_sources", "insufficient_sources"] as const;
const VERIFY_STATUSES = ["checked", "skipped", "unavailable"] as const;

function plan(v: unknown, path: string): PlanMeta {
  const o = obj(v, path);
  return {
    intent: oneOf(o, "intent", path, INTENTS),
    rephrasings: strArr(o, "rephrasings", path),
    arxiv_ids: strArr(o, "arxiv_ids", path),
    categories: strArr(o, "categories", path),
    published_from: strOrNull(o, "published_from", path),
    published_to: strOrNull(o, "published_to", path),
    dropped: strArr(o, "dropped", path),
  };
}

function source(v: unknown, path: string): Source {
  const o = obj(v, path);
  return {
    n: num(o, "n", path),
    chunk_id: str(o, "chunk_id", path),
    arxiv_id: str(o, "arxiv_id", path),
    title: str(o, "title", path),
    section_path: strOrNull(o, "section_path", path),
    chunk_type: oneOf(o, "chunk_type", path, CHUNK_TYPES),
    published_at: str(o, "published_at", path),
    abs_url: strOrNull(o, "abs_url", path),
  };
}

function sentence(v: unknown, path: string): SentenceCheck {
  const o = obj(v, path);
  const supported = o["supported"];
  if (supported !== null && typeof supported !== "boolean") {
    fail(`${path}.supported`, "a boolean or null");
  }
  return {
    index: num(o, "index", path),
    text: str(o, "text", path),
    citations: numArr(o, "citations", path),
    invalid_citations: numArr(o, "invalid_citations", path),
    supported,
    reason: typeof o["reason"] === "string" ? o["reason"] : "",
  };
}

/**
 * One SSE message -> a typed event. Returns null for an event name this client does not
 * know (a newer backend may add events; ignoring them is safe). Throws ChatProtocolError
 * when the JSON is invalid or a known event has the wrong shape.
 */
export function parseChatEvent(name: string, raw: string): ChatEvent | null {
  switch (name) {
    case "meta":
    case "sources":
    case "token":
    case "refusal":
    case "verification":
    case "done":
    case "error":
      break;
    default:
      return null;
  }
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch {
    throw new ChatProtocolError(`bad chat event: ${name} data is not JSON`);
  }
  const o = obj(json, name);
  switch (name) {
    case "meta":
      return {
        event: "meta",
        data: {
          conversation_id: str(o, "conversation_id", name),
          new_conversation: bool(o, "new_conversation", name),
          intent: oneOf(o, "intent", name, INTENTS),
          standalone_question: str(o, "standalone_question", name),
          plan: plan(o["plan"], `${name}.plan`),
        },
      };
    case "sources":
      return {
        event: "sources",
        data: {
          sources: arr(o, "sources", name).map((s, i) =>
            source(s, `${name}.sources[${i}]`),
          ),
        },
      };
    case "token":
      return { event: "token", data: { text: str(o, "text", name) } };
    case "refusal":
      return {
        event: "refusal",
        data: {
          reason: oneOf(o, "reason", name, REFUSALS),
          text: str(o, "text", name),
        },
      };
    case "verification":
      return {
        event: "verification",
        data: {
          status: oneOf(o, "status", name, VERIFY_STATUSES),
          sentences: arr(o, "sentences", name).map((s, i) =>
            sentence(s, `${name}.sentences[${i}]`),
          ),
          invalid_citations: numArr(o, "invalid_citations", name),
          uncited_sentences: numArr(o, "uncited_sentences", name),
          unsupported_sentences: numArr(o, "unsupported_sentences", name),
          reason: typeof o["reason"] === "string" ? o["reason"] : "",
        },
      };
    case "done": {
      const refusal = o["refusal"];
      return {
        event: "done",
        data: {
          conversation_id: str(o, "conversation_id", name),
          message_id: num(o, "message_id", name),
          refusal:
            refusal === null || refusal === undefined
              ? null
              : oneOf(o, "refusal", name, REFUSALS),
          abstract_only: bool(o, "abstract_only", name),
          citations: arr(o, "citations", name).map((s, i) =>
            source(s, `${name}.citations[${i}]`),
          ),
          tokens_in: num(o, "tokens_in", name),
          tokens_out: num(o, "tokens_out", name),
          cost_usd: str(o, "cost_usd", name),
          trace_id: strOrNull(o, "trace_id", name),
        },
      };
    }
    default:
      return {
        event: "error",
        data: {
          stage: str(o, "stage", name),
          message: str(o, "message", name),
        },
      };
  }
}

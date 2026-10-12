// POST /chat and read the answer as it is written.
//
// Not done with the typed openapi-fetch client: the response is a stream of server-sent
// events, which an OpenAPI schema cannot describe. The request body is the schema's
// ChatRequest; the events are typed by hand in ./events.ts.

import { ChatProtocolError, parseChatEvent } from "./events.ts";
import type { ChatEvent } from "./events.ts";
import { readSse } from "./sse.ts";

export interface ChatRequest {
  message: string;
  /** Omit on the first question; send the id from the `meta` event to continue. */
  conversation_id?: string | null;
}

/** The server refused the request before streaming (422, 404, 503, ...). */
export class ChatHttpError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ChatHttpError";
    this.status = status;
  }
}

/** An event with the time (ms since the request was sent) at which it was received. */
export interface TimedEvent {
  event: ChatEvent;
  receivedAt: number;
}

export interface StreamOptions {
  /** Where the API is reachable; the Next rewrite in next.config.ts serves /api. */
  baseUrl?: string;
  signal?: AbortSignal;
  /** For tests. */
  fetchImpl?: typeof fetch;
  now?: () => number;
}

// FastAPI answers {"detail": "..."} for HTTPException and {"detail": [{msg, ...}]} for 422.
async function errorMessage(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail))
        return "The question was not accepted (check its length).";
    }
  } catch {
    // not JSON: fall through to the status text
  }
  return res.statusText || `HTTP ${res.status}`;
}

export async function* streamChat(
  request: ChatRequest,
  options: StreamOptions = {},
): AsyncGenerator<TimedEvent> {
  const doFetch = options.fetchImpl ?? fetch;
  const now = options.now ?? (() => performance.now());
  const startedAt = now();

  const res = await doFetch(`${options.baseUrl ?? "/api"}/chat`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "text/event-stream",
    },
    body: JSON.stringify({
      message: request.message,
      ...(request.conversation_id
        ? { conversation_id: request.conversation_id }
        : {}),
    }),
    signal: options.signal,
  });
  if (!res.ok) throw new ChatHttpError(res.status, await errorMessage(res));
  if (!res.body)
    throw new ChatProtocolError("the server sent no response body");

  let finished = false;
  for await (const message of readSse(res.body)) {
    const event = parseChatEvent(message.event, message.data);
    if (event === null) continue; // an event this client does not know
    yield { event, receivedAt: now() - startedAt };
    if (event.event === "done" || event.event === "error") {
      finished = true;
      break;
    }
  }
  if (!finished) {
    throw new ChatProtocolError(
      "the connection ended before the answer was complete",
    );
  }
}

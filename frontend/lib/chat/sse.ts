// A server-sent-events parser for fetch() bodies. The browser's EventSource cannot be
// used: /chat needs a POST body, and EventSource only does GET.
//
// It follows the SSE format (WHATWG HTML, "Server-sent events"): lines end in \n, \r or
// \r\n; `field: value` lines build up one message; a blank line sends it; lines starting
// with ":" are comments (FastAPI sends them as keep-alive pings) and are ignored.
// Chunks from the network can end anywhere, even between the \r and \n of one line break
// or inside a multi-byte character, so the parser keeps whatever is unfinished.
//
// Plain TypeScript with no imports: Node's test runner runs it as is.

export interface SseMessage {
  /** The `event:` name; "message" when the server sent none. */
  event: string;
  /** The `data:` lines joined with "\n". */
  data: string;
}

export class SseParser {
  private carry = ""; // the start of a line whose end has not arrived yet
  private skipLf = false; // the last chunk ended with \r: a \n next is part of the same break
  private started = false;
  private eventName = "";
  private dataLines: string[] = [];

  /** Feed decoded text; returns the messages completed by it (maybe none). */
  push(text: string): SseMessage[] {
    let s = this.carry + text;
    this.carry = "";
    if (!this.started && s.length > 0) {
      this.started = true;
      if (s.charCodeAt(0) === 0xfeff) s = s.slice(1); // a UTF-8 byte order mark
    }
    if (this.skipLf && s.length > 0) {
      if (s[0] === "\n") s = s.slice(1);
      this.skipLf = false;
    }
    const out: SseMessage[] = [];
    let start = 0;
    let i = 0;
    while (i < s.length) {
      const c = s[i];
      if (c !== "\n" && c !== "\r") {
        i++;
        continue;
      }
      const line = s.slice(start, i);
      i++;
      if (c === "\r") {
        if (i < s.length) {
          if (s[i] === "\n") i++;
        } else {
          this.skipLf = true;
        }
      }
      start = i;
      this.line(line, out);
    }
    this.carry = s.slice(start);
    return out;
  }

  // A message still open when the stream ends is dropped, as the specification says:
  // only a blank line completes one. (The backend always ends with one.)

  private line(line: string, out: SseMessage[]): void {
    if (line === "") {
      if (this.dataLines.length > 0) {
        out.push({
          event: this.eventName || "message",
          data: this.dataLines.join("\n"),
        });
      }
      this.eventName = "";
      this.dataLines = [];
      return;
    }
    if (line[0] === ":") return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value[0] === " ") value = value.slice(1); // one optional space after the colon
    if (field === "event") this.eventName = value;
    else if (field === "data") this.dataLines.push(value);
    // `id` and `retry` are not used by this client; unknown fields are ignored.
  }
}

/** Read a response body as SSE messages, as they arrive. */
export async function* readSse(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  const parser = new SseParser();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      // stream: true keeps the bytes of a character split between two chunks.
      for (const message of parser.push(
        decoder.decode(value, { stream: true }),
      ))
        yield message;
    }
    for (const message of parser.push(decoder.decode())) yield message;
  } finally {
    // Also runs when the consumer stops early (break, throw, abort): close the connection.
    await reader.cancel().catch(() => undefined);
  }
}

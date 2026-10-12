// Run with `pnpm test:unit` (Node's built-in test runner; it strips the types itself).
import { strict as assert } from "node:assert";
import { test } from "node:test";

import { readSse, SseParser } from "./sse.ts";
import type { SseMessage } from "./sse.ts";

function parseAll(chunks: string[]): SseMessage[] {
  const parser = new SseParser();
  return chunks.flatMap((chunk) => parser.push(chunk));
}

// Mixed line endings, a comment, a multi-line data field, a field with no space after the
// colon, and an id/retry the client ignores.
const STREAM =
  ": ping\r\n" +
  'event: token\r\ndata: {"text":"Hello"}\r\n\r\n' +
  'event: token\ndata:{"text":" world"}\n\n' +
  "id: 7\nretry: 1000\nevent: note\rdata: line one\rdata: line two\r\r" +
  "data: no event name\n\n";

const EXPECTED: SseMessage[] = [
  { event: "token", data: '{"text":"Hello"}' },
  { event: "token", data: '{"text":" world"}' },
  { event: "note", data: "line one\nline two" },
  { event: "message", data: "no event name" },
];

test("parses messages with CRLF, LF and CR line endings, comments and multi-line data", () => {
  assert.deepEqual(parseAll([STREAM]), EXPECTED);
});

test("the result does not depend on where the network cuts the stream", () => {
  // every single split point...
  for (let i = 0; i <= STREAM.length; i++) {
    assert.deepEqual(
      parseAll([STREAM.slice(0, i), STREAM.slice(i)]),
      EXPECTED,
      `split at ${i}`,
    );
  }
  // ...and one character at a time (this also cuts between a \r and its \n)
  assert.deepEqual(parseAll([...STREAM]), EXPECTED);
});

test("a message is only complete at its blank line; an unfinished one is dropped", () => {
  const parser = new SseParser();
  assert.deepEqual(parser.push("event: token\ndata: partial"), []);
  assert.deepEqual(parser.push("\n"), []); // the line ended, the message did not
  assert.deepEqual(parser.push("\n"), [{ event: "token", data: "partial" }]);
  assert.deepEqual(parser.push("event: token\ndata: cut off\n"), []); // stream ends here
});

test("a blank line with no data sends nothing and resets the event name", () => {
  assert.deepEqual(parseAll(["event: orphan\n\n", "data: x\n\n"]), [
    { event: "message", data: "x" },
  ]);
});

test("a leading byte order mark is ignored", () => {
  assert.deepEqual(parseAll(["﻿event: a\ndata: 1\n\n"]), [
    { event: "a", data: "1" },
  ]);
});

test("only one space after the colon is removed", () => {
  assert.deepEqual(parseAll(["data:   three\n\n"]), [
    { event: "message", data: "  three" },
  ]);
});

// keepOpen: after the last chunk the stream stays open, like a connection the server has
// not closed yet (only then does cancelling it reach the source).
function streamOf(
  chunks: Uint8Array[],
  onCancel?: () => void,
  keepOpen = false,
): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) controller.enqueue(chunks[i++]);
      else if (!keepOpen) controller.close();
      else return new Promise<void>(() => undefined);
    },
    cancel() {
      onCancel?.();
    },
  });
}

async function collect(
  stream: ReadableStream<Uint8Array>,
): Promise<SseMessage[]> {
  const out: SseMessage[] = [];
  for await (const message of readSse(stream)) out.push(message);
  return out;
}

test("readSse keeps a multi-byte character that is split between two chunks", async () => {
  const bytes = new TextEncoder().encode(
    'event: token\ndata: {"text":"café €5 🙂"}\n\n',
  );
  for (let i = 1; i < bytes.length; i++) {
    const got = await collect(streamOf([bytes.slice(0, i), bytes.slice(i)]));
    assert.deepEqual(
      got,
      [{ event: "token", data: '{"text":"café €5 🙂"}' }],
      `split at byte ${i}`,
    );
  }
});

test("readSse yields each message as soon as its chunk arrives, not at the end", async () => {
  const encoder = new TextEncoder();
  let release!: () => void;
  const gate = new Promise<void>((resolve) => (release = resolve));
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      controller.enqueue(encoder.encode("event: token\ndata: 1\n\n"));
      await gate; // the second chunk is not sent until the test lets it
      controller.enqueue(encoder.encode("event: token\ndata: 2\n\n"));
      controller.close();
    },
  });
  const iterator = readSse(stream);
  const first = await iterator.next(); // would hang here if the parser waited for the end
  assert.deepEqual(first.value, { event: "token", data: "1" });
  release();
  const second = await iterator.next();
  assert.deepEqual(second.value, { event: "token", data: "2" });
  assert.equal((await iterator.next()).done, true);
});

test("readSse closes the connection when the consumer stops early", async () => {
  let cancelled = false;
  const body = new TextEncoder().encode("data: a\n\ndata: b\n\n");
  const stream = streamOf([body], () => (cancelled = true), true);
  for await (const message of readSse(stream)) {
    assert.equal(message.data, "a");
    break;
  }
  assert.equal(cancelled, true);
});

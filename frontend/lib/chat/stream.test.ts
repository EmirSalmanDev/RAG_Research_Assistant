import { strict as assert } from "node:assert";
import { test } from "node:test";

import { ChatProtocolError } from "./events.ts";
import { fullTurn, META } from "./fixtures.ts";
import { ChatHttpError, streamChat } from "./stream.ts";
import type { TimedEvent } from "./stream.ts";

/** A fetch that answers with `body` and records the request it was given. */
function fakeFetch(
  body: BodyInit | null,
  init: ResponseInit = { status: 200 },
) {
  const calls: { url: string; init: RequestInit }[] = [];
  const impl = (async (
    url: string | URL | Request,
    requestInit?: RequestInit,
  ) => {
    calls.push({ url: String(url), init: requestInit ?? {} });
    return new Response(body, init);
  }) as typeof fetch;
  return { impl, calls };
}

function streamOfText(pieces: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < pieces.length) controller.enqueue(encoder.encode(pieces[i++]));
      else controller.close();
    },
  });
}

async function collect(gen: AsyncGenerator<TimedEvent>): Promise<TimedEvent[]> {
  const out: TimedEvent[] = [];
  for await (const item of gen) out.push(item);
  return out;
}

test("streams typed events in order, with the time each one arrived", async () => {
  const text = fullTurn();
  const cut = Math.floor(text.length / 3);
  const { impl } = fakeFetch(
    streamOfText([text.slice(0, cut), text.slice(cut)]),
  );
  let clock = 0;
  const events = await collect(
    streamChat(
      { message: "What are adapters?" },
      { fetchImpl: impl, now: () => (clock += 10) },
    ),
  );
  assert.deepEqual(
    events.map((e) => e.event.event),
    ["meta", "sources", "token", "token", "token", "verification", "done"],
  );
  // timestamps only go up (here, by the fake clock): the page can show that tokens trickle in
  const times = events.map((e) => e.receivedAt);
  assert.deepEqual(
    times,
    [...times].sort((a, b) => a - b),
  );
});

test("posts JSON to /api/chat and sends the conversation id only on a follow-up", async () => {
  const first = fakeFetch(streamOfText([fullTurn()]));
  await collect(streamChat({ message: "one" }, { fetchImpl: first.impl }));
  assert.equal(first.calls[0]?.url, "/api/chat");
  assert.equal(first.calls[0]?.init.method, "POST");
  assert.deepEqual(JSON.parse(String(first.calls[0]?.init.body)), {
    message: "one",
  });

  const second = fakeFetch(streamOfText([fullTurn()]));
  await collect(
    streamChat(
      { message: "two", conversation_id: META.conversation_id },
      { fetchImpl: second.impl, baseUrl: "http://localhost:8000" },
    ),
  );
  assert.equal(second.calls[0]?.url, "http://localhost:8000/chat");
  assert.deepEqual(JSON.parse(String(second.calls[0]?.init.body)), {
    message: "two",
    conversation_id: META.conversation_id,
  });
});

test("an HTTP error before the stream becomes ChatHttpError with the server's detail", async () => {
  const { impl } = fakeFetch(
    JSON.stringify({ detail: "OPENAI_API_KEY is not set (put it in .env)" }),
    {
      status: 503,
      headers: { "content-type": "application/json" },
    },
  );
  await assert.rejects(
    collect(streamChat({ message: "q" }, { fetchImpl: impl })),
    (err) => {
      assert.ok(err instanceof ChatHttpError);
      assert.equal(err.status, 503);
      assert.match(err.message, /OPENAI_API_KEY/);
      return true;
    },
  );
});

test("a 422 validation list gets a plain message; a non-JSON error falls back to the status", async () => {
  const validation = fakeFetch(
    JSON.stringify({ detail: [{ msg: "too long", loc: ["body"] }] }),
    {
      status: 422,
    },
  );
  await assert.rejects(
    collect(streamChat({ message: "q" }, { fetchImpl: validation.impl })),
    /not accepted/,
  );
  const plain = fakeFetch("Bad gateway", {
    status: 502,
    statusText: "Bad Gateway",
  });
  await assert.rejects(
    collect(streamChat({ message: "q" }, { fetchImpl: plain.impl })),
    /Bad Gateway/,
  );
});

test("a stream that ends before done or error is a protocol error", async () => {
  const cutShort = fullTurn().split("event: done")[0] ?? "";
  const { impl } = fakeFetch(streamOfText([cutShort]));
  await assert.rejects(
    collect(streamChat({ message: "q" }, { fetchImpl: impl })),
    (err) =>
      err instanceof ChatProtocolError &&
      /ended before the answer was complete/.test(err.message),
  );
});

test("an error event ends the stream normally (the reducer shows it)", async () => {
  const body =
    'event: error\ndata: {"stage":"search","message":"search failed: X"}\n\n';
  const { impl } = fakeFetch(streamOfText([body]));
  const events = await collect(
    streamChat({ message: "q" }, { fetchImpl: impl }),
  );
  assert.equal(events.length, 1);
  assert.equal(events[0]?.event.event, "error");
});

test("events it does not know are skipped; malformed known events throw", async () => {
  const withUnknown = 'event: progress\ndata: {"pct":1}\n\n' + fullTurn();
  const ok = fakeFetch(streamOfText([withUnknown]));
  const events = await collect(
    streamChat({ message: "q" }, { fetchImpl: ok.impl }),
  );
  assert.equal(events[0]?.event.event, "meta");

  const bad = fakeFetch(streamOfText(['event: token\ndata: {"text":5}\n\n']));
  await assert.rejects(
    collect(streamChat({ message: "q" }, { fetchImpl: bad.impl })),
    ChatProtocolError,
  );
});

test("the abort signal is passed to fetch", async () => {
  const controller = new AbortController();
  const { impl, calls } = fakeFetch(streamOfText([fullTurn()]));
  await collect(
    streamChat(
      { message: "q" },
      { fetchImpl: impl, signal: controller.signal },
    ),
  );
  assert.equal(calls[0]?.init.signal, controller.signal);
});

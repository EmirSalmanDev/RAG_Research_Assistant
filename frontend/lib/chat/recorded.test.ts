// Replays streams written by the REAL POST /chat route (backend/tests/unit/test_sse_fixtures.py
// regenerates them; its content is the backend tests' invented text) through the parser, the
// reducer and the stream client. This is the check that the hand-written event types and the
// backend agree: if the backend changes its stream, that test fails until the files are
// rewritten, and this one fails if the frontend no longer understands the new files.
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { splitCitations } from "./citations.ts";
import type { ChatEvent } from "./events.ts";
import { parseChatEvent } from "./events.ts";
import { chatReducer, initialChatState } from "./reducer.ts";
import type { AssistantMessage } from "./reducer.ts";
import { SseParser } from "./sse.ts";
import { streamChat } from "./stream.ts";

function load(name: string): string {
  return readFileSync(
    new URL(`./testdata/${name}.sse`, import.meta.url),
    "utf8",
  );
}

/** Parse a raw stream cut into pieces of `size` characters. */
function events(text: string, size = text.length): ChatEvent[] {
  const parser = new SseParser();
  const out: ChatEvent[] = [];
  for (let i = 0; i < text.length; i += size) {
    for (const m of parser.push(text.slice(i, i + size))) {
      const event = parseChatEvent(m.event, m.data);
      assert.notEqual(event, null, `unknown event ${m.event}`);
      out.push(event as ChatEvent);
    }
  }
  return out;
}

function answer(list: ChatEvent[]): AssistantMessage {
  let state = chatReducer(initialChatState, {
    type: "ask",
    text: "q",
    userId: "u",
    assistantId: "a",
  });
  for (const event of list)
    state = chatReducer(state, { type: "event", event });
  return state.messages[1] as AssistantMessage;
}

const names = (list: ChatEvent[]) => list.map((e) => e.event);

const FIXTURES = [
  "turn-normal",
  "turn-abstract-only",
  "turn-refusal-no-sources",
  "turn-refusal-insufficient",
  "turn-error-search",
];

test("every recorded stream parses the same however the network cuts it", () => {
  for (const name of FIXTURES) {
    const text = load(name);
    const whole = events(text);
    assert.ok(whole.length >= 2, name);
    for (const size of [1, 2, 7, 64, 1000])
      assert.deepEqual(events(text, size), whole, `${name} @${size}`);
  }
});

test("a normal turn: meta, sources, tokens, verification, done; cited sources become chips", () => {
  const list = events(load("turn-normal"));
  assert.deepEqual(names(list), [
    "meta",
    "sources",
    "token",
    "token",
    "verification",
    "done",
  ]);
  const a = answer(list);
  assert.equal(a.status, "done");
  assert.equal(a.sources.length, 2);
  assert.equal(a.done?.refusal, null);
  assert.deepEqual(
    a.done?.citations.map((c) => c.n),
    [1, 2],
  );
  const chips = splitCitations(a.text, a.sources).filter(
    (s) => s.kind === "citation",
  );
  assert.deepEqual(
    chips.map((c) => c.kind === "citation" && c.n),
    [1, 2],
  );
  assert.equal(a.verification?.status, "checked");
  assert.equal(a.meta?.plan.categories.length, 0);
});

test("an abstract-only turn says so, and its chip source is an abstract chunk", () => {
  const a = answer(events(load("turn-abstract-only")));
  assert.equal(a.status, "done");
  assert.equal(a.done?.abstract_only, true);
  assert.equal(a.sources[0]?.chunk_type, "abstract");
  const chip = splitCitations(a.text, a.sources).find(
    (s) => s.kind === "citation",
  );
  assert.ok(chip?.kind === "citation" && chip.source.chunk_type === "abstract");
});

test("a refusal because nothing was found: no sources, no tokens, status refused", () => {
  const list = events(load("turn-refusal-no-sources"));
  assert.deepEqual(names(list), ["meta", "refusal", "done"]);
  const a = answer(list);
  assert.equal(a.status, "refused");
  assert.equal(a.refusal?.reason, "no_sources");
  assert.equal(a.text, "");
  assert.deepEqual(a.sources, []);
  assert.equal(a.done?.refusal, "no_sources");
  assert.deepEqual(a.done?.citations, []);
  assert.deepEqual(a.meta?.plan.categories, ["cs.CL"]); // the filter chip the page shows
});

test("a refusal after sources were found: the sources are known, the answer is the refusal", () => {
  const list = events(load("turn-refusal-insufficient"));
  assert.deepEqual(names(list), ["meta", "sources", "refusal", "done"]);
  const a = answer(list);
  assert.equal(a.status, "refused");
  assert.equal(a.refusal?.reason, "insufficient_sources");
  assert.equal(a.sources.length, 2);
  assert.equal(a.text, "");
});

test("a failure after the stream started is an error event that ends the turn", () => {
  const list = events(load("turn-error-search"));
  assert.deepEqual(names(list), ["meta", "error"]);
  const a = answer(list);
  assert.equal(a.status, "error");
  assert.equal(a.error, "search failed: ConnectionError");
});

test("the stream client reads every recorded stream to its end", async () => {
  for (const name of FIXTURES) {
    const text = load(name);
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        const bytes = new TextEncoder().encode(text);
        for (let i = 0; i < bytes.length; i += 17)
          controller.enqueue(bytes.slice(i, i + 17));
        controller.close();
      },
    });
    const impl = (async () =>
      new Response(body, { status: 200 })) as typeof fetch;
    const seen: string[] = [];
    for await (const { event } of streamChat(
      { message: "q" },
      { fetchImpl: impl },
    ))
      seen.push(event.event);
    assert.deepEqual(seen, names(events(text)), name);
  }
});

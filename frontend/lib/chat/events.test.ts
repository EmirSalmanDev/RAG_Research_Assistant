import { strict as assert } from "node:assert";
import { test } from "node:test";

import { ChatProtocolError, parseChatEvent } from "./events.ts";
import {
  ABSTRACT_SOURCE,
  DONE,
  META,
  SECTION_SOURCE,
  VERIFICATION,
} from "./fixtures.ts";

const json = JSON.stringify;

test("parses every event the backend sends", () => {
  const meta = parseChatEvent("meta", json(META));
  assert.equal(meta?.event, "meta");
  assert.equal(meta?.event === "meta" && meta.data.plan.categories[0], "cs.CL");

  const sources = parseChatEvent(
    "sources",
    json({ sources: [ABSTRACT_SOURCE, SECTION_SOURCE] }),
  );
  assert.equal(sources?.event === "sources" && sources.data.sources.length, 2);

  assert.deepEqual(parseChatEvent("token", json({ text: "hi" })), {
    event: "token",
    data: { text: "hi" },
  });
  assert.deepEqual(
    parseChatEvent(
      "refusal",
      json({ reason: "no_sources", text: "Nothing found." }),
    ),
    {
      event: "refusal",
      data: { reason: "no_sources", text: "Nothing found." },
    },
  );
  const verification = parseChatEvent("verification", json(VERIFICATION));
  assert.equal(
    verification?.event === "verification" && verification.data.status,
    "checked",
  );

  const done = parseChatEvent("done", json(DONE));
  assert.equal(done?.event === "done" && done.data.abstract_only, true);
  assert.deepEqual(
    parseChatEvent(
      "error",
      json({ stage: "search", message: "search failed: X" }),
    ),
    {
      event: "error",
      data: { stage: "search", message: "search failed: X" },
    },
  );
});

test("keeps only the fields it knows (extra backend fields are dropped, not trusted)", () => {
  const event = parseChatEvent("token", json({ text: "a", surprise: 1 }));
  assert.deepEqual(event, { event: "token", data: { text: "a" } });
});

test("a done event for a refusal has a reason; null means a normal answer", () => {
  const refused = parseChatEvent(
    "done",
    json({ ...DONE, refusal: "insufficient_sources" }),
  );
  assert.equal(
    refused?.event === "done" && refused.data.refusal,
    "insufficient_sources",
  );
  const normal = parseChatEvent("done", json({ ...DONE, refusal: null }));
  assert.equal(normal?.event === "done" && normal.data.refusal, null);
});

test("an event name it does not know is ignored, so a newer backend still works", () => {
  assert.equal(parseChatEvent("progress", json({ pct: 10 })), null);
});

test("bad JSON and wrong shapes throw ChatProtocolError with the path", () => {
  assert.throws(() => parseChatEvent("token", "{not json"), ChatProtocolError);
  assert.throws(
    () => parseChatEvent("token", json({ text: 5 })),
    /token\.text should be a string/,
  );
  assert.throws(
    () => parseChatEvent("token", json([1])),
    /should be an object/,
  );
  assert.throws(
    () =>
      parseChatEvent(
        "sources",
        json({ sources: [{ ...ABSTRACT_SOURCE, chunk_type: "poster" }] }),
      ),
    /sources\.sources\[0\]\.chunk_type should be one of abstract, section, analysis/,
  );
  assert.throws(
    () => parseChatEvent("done", json({ ...DONE, refusal: "because" })),
    /done\.refusal should be one of/,
  );
  assert.throws(
    () => parseChatEvent("meta", json({ ...META, plan: null })),
    /meta\.plan/,
  );
});

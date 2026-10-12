// A stand-in for the FastAPI server, for the browser test only. Run by Playwright's
// webServer (`node e2e/mock-api.ts`; Node strips the types).
//
// WHAT THIS PROVES, AND WHAT IT DOES NOT: the browser test talks to this mock through the
// real Next.js /api rewrite, so it proves the UI (streaming, chips, the sources panel) and
// that the rewrite passes a server-sent-event stream through unbuffered. It does NOT prove
// anything about the real backend: the event shapes here come from lib/chat/fixtures.ts,
// which backend/tests/unit/test_sse_fixtures.py and lib/chat/recorded.test.ts keep honest,
// and the content is invented (never paper text). A run against the real API is a
// separate, deliberate step because it costs money.
//
// POST /chat sends the first part of an answer and then HOLDS the stream open until the
// test calls POST /__release. That makes "the text is on screen while the turn is still
// running" something the test can assert, instead of a race against a timer.
import { createServer } from "node:http";
import type { ServerResponse } from "node:http";

import {
  ABSTRACT_SOURCE,
  DONE,
  META,
  SECTION_SOURCE,
  VERIFICATION,
  sse,
} from "../lib/chat/fixtures.ts";
import { API_PORT } from "./containment.ts";

const SOURCES = [ABSTRACT_SOURCE, SECTION_SOURCE];

// Before the release: a sentence with no citation marker in it.
const FIRST_PART = [
  "Adapters train a ",
  "small module ",
  "beside the frozen model",
];
// After it: the citations, and enough lines for the page to need scrolling.
const SECOND_PART = [
  " [1]. They also freeze the base model [2].\n\n",
  ...Array.from(
    { length: 14 },
    (_, i) => `- Invented point ${i + 1} about adapters.\n`,
  ),
];

let released = false;
let onRelease: (() => void) | null = null;

function hold(): Promise<void> {
  if (released) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 40_000); // never hang a run for good
    onRelease = () => {
      clearTimeout(timer);
      resolve();
    };
  });
}

function json(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

async function chat(res: ServerResponse): Promise<void> {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    "x-accel-buffering": "no", // the headers the real route sends
  });
  res.write(sse("meta", META));
  res.write(sse("sources", { sources: SOURCES }));
  for (const text of FIRST_PART) res.write(sse("token", { text }));
  await hold();
  for (const text of SECOND_PART) res.write(sse("token", { text }));
  res.write(sse("verification", VERIFICATION));
  res.write(sse("done", { ...DONE, citations: SOURCES, abstract_only: false }));
  res.end();
}

createServer((req, res) => {
  const url = req.url ?? "";
  if (req.method === "GET" && url === "/health")
    return json(res, 200, { ok: true });
  if (req.method === "POST" && url === "/chat") {
    req.resume();
    void chat(res);
    return;
  }
  if (req.method === "POST" && url === "/__release") {
    released = true;
    onRelease?.();
    return json(res, 200, { released: true });
  }
  const chunk = /^\/chunks\/([0-9a-f-]+)$/.exec(url);
  if (req.method === "GET" && chunk) {
    const source = SOURCES.find((s) => s.chunk_id === chunk[1]);
    if (!source) return json(res, 404, { detail: "unknown chunk" });
    return json(res, 200, {
      chunk_id: source.chunk_id,
      arxiv_id: source.arxiv_id,
      title: source.title,
      chunk_type: source.chunk_type,
      section_path: source.section_path,
      ord: source.n,
      text: `Invented passage text for ${source.title}.`,
      published_at: source.published_at,
      abs_url: source.abs_url,
    });
  }
  return json(res, 404, { detail: "not found" });
}).listen(Number(process.env.PORT ?? API_PORT), "127.0.0.1");

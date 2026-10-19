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
import type { IncomingMessage, ServerResponse } from "node:http";

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

// --- the paper page (e2e/paper-page.spec.ts): one invented paper with everything ---
const PAPER_ID = "2609.00001";
const PUBLISHED = "2026-09-01T12:00:00+00:00";
const abs = (id: string) => `https://arxiv.org/abs/${id}`;
const PAPER = {
  arxiv_id: PAPER_ID,
  version: 1,
  title: "An Invented Paper About Adapters",
  abstract: "An invented abstract about adapters.",
  authors: ["Ada Example"],
  categories: ["cs.CL"],
  primary_category: "cs.CL",
  published_at: PUBLISHED,
  hf_upvotes: 12,
  citation_count: 3,
  github_url: "https://github.com/invented-org/adapters",
  project_url: null,
  fulltext_status: "indexed",
  abs_url: abs(PAPER_ID),
};
const model = (i: number) => ({
  repo_id: `invented-org/adapter-model-${i}`,
  downloads: 1000 * (12 - i),
  likes: 12 - i,
  pipeline_tag: "text-generation",
  url: `https://huggingface.co/invented-org/adapter-model-${i}`,
});
const DETAILS = {
  arxiv_id: PAPER_ID,
  analysis_status: "available",
  analysis: {
    tldr: "Invented TL;DR: small adapters match full fine-tuning.",
    contributions: ["An invented adapter.", "An invented evaluation."],
    method: "An invented method.",
    results: "Invented results.",
    limitations: "An invented limitation.",
    model: "gpt-4o-mini",
    prompt_version: "analyze_v1",
    created_at: PUBLISHED,
  },
  repos: [
    {
      full_name: "invented-org/adapters",
      stars: 1497,
      last_push: PUBLISHED,
      license: "Apache-2.0",
      url: "https://github.com/invented-org/adapters",
    },
  ],
  models: Array.from({ length: 11 }, (_, i) => model(i + 1)),
  datasets: [
    {
      repo_id: "invented-org/adapter-data",
      downloads: 50,
      likes: 1,
      pipeline_tag: null,
      url: "https://huggingface.co/datasets/invented-org/adapter-data",
    },
  ],
  spaces: [],
};
const related = (id: string, title: string | null, score: number | null) => ({
  arxiv_id: id,
  title,
  published_at: title ? PUBLISHED : null,
  held: title !== null,
  score,
  abs_url: abs(id),
});
const RELATED = {
  arxiv_id: PAPER_ID,
  similar_status: "ok",
  similar: [related("2609.00002", "A Similar Invented Paper", 0.91)],
  cites: [related("1706.00000", null, null)],
  cited_by: [related("2609.00003", "A Citing Invented Paper", null)],
};

function paperRoute(res: ServerResponse, url: string): boolean {
  const m = /^\/papers\/([^/?]+)(?:\/(chunks|details|related))?(?:\?.*)?$/.exec(
    url,
  );
  if (!m) return false;
  if (m[1] !== PAPER_ID) {
    json(res, 404, { detail: "unknown paper" });
    return true;
  }
  const bodies = {
    chunks: { arxiv_id: PAPER_ID, chunks: [] },
    details: DETAILS,
    related: RELATED,
  };
  json(res, 200, m[2] ? bodies[m[2] as keyof typeof bodies] : PAPER);
  return true;
}

// --- the digest, interests and settings (e2e/digest-flow.spec.ts): invented, in memory ---
type EntryStatus = "new" | "saved" | "dismissed";
const DIGEST_DATE = "2026-10-19";
const entry = (n: number, extra: Record<string, unknown> = {}) => ({
  arxiv_id: `2610.0000${n}`,
  rank: n,
  score: 0.9 - n / 10,
  status: "new" as EntryStatus,
  title: `Invented Digest Paper ${n}`,
  authors: ["Ada Example", "Bo Sample", "Cy Test"],
  author_count: n === 1 ? 5 : 3,
  categories: ["cs.CL", "cs.LG"],
  published_at: "2026-10-18T12:00:00+00:00",
  hf_upvotes: n === 1 ? 25 : null,
  trending: n === 1,
  best_interest: null as { id: number; name: string } | null,
  summary: `Invented one-line summary number ${n}.`,
  summary_source: "abstract",
  abs_url: abs(`2610.0000${n}`),
  ...extra,
});
const digestEntries = [entry(1), entry(2), entry(3)];
type MockInterest = {
  id: number;
  name: string;
  description: string | null;
  keywords: string[];
  categories: string[];
  is_active: boolean;
  weight: number;
  embedded: boolean;
  created_at: string;
  updated_at: string;
};
const interests: MockInterest[] = [];
let nextInterestId = 1;
let settings = {
  digest_size: 20,
  categories: ["cs.CL", "cs.LG", "cs.AI", "cs.IR"],
  categories_default: ["cs.CL", "cs.LG", "cs.AI", "cs.IR"],
  categories_overridden: false,
  fulltext_top_n: 20,
  budget_cap_usd: null as number | null,
  month_to_date_usd: 0.045026,
};

function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (chunk) => (raw += chunk));
    req.on("end", () => resolve(raw ? JSON.parse(raw) : {}));
  });
}

async function digestRoute(
  req: IncomingMessage,
  res: ServerResponse,
  url: string,
): Promise<boolean> {
  const path = url.split("?")[0];
  if (req.method === "GET" && path === "/digest") {
    json(res, 200, {
      requested_date: DIGEST_DATE,
      digest_date: DIGEST_DATE,
      is_fallback: false,
      has_interests: interests.some((i) => i.is_active),
      recent_dates: [DIGEST_DATE],
      entries: digestEntries.map((e) => ({
        ...e,
        best_interest: interests.find((i) => i.is_active)
          ? { id: interests[0].id, name: interests[0].name }
          : null,
      })),
    });
    return true;
  }
  const act = /^\/digest\/([^/]+)\/(save|dismiss|undo)$/.exec(path);
  if (req.method === "POST" && act) {
    const row = digestEntries.find((e) => e.arxiv_id === act[1]);
    if (!row) {
      json(res, 404, { detail: "that paper is not in this digest" });
      return true;
    }
    row.status = { save: "saved", dismiss: "dismissed", undo: "new" }[
      act[2] as "save" | "dismiss" | "undo"
    ] as EntryStatus;
    json(res, 200, row);
    return true;
  }
  if (path === "/interests" && req.method === "GET") {
    json(res, 200, interests);
    return true;
  }
  if (path === "/interests" && req.method === "POST") {
    const body = await readJson(req);
    if (interests.some((i) => i.name === body.name)) {
      json(res, 409, { detail: "an interest with this name exists" });
      return true;
    }
    const now = "2026-10-19T05:00:00+00:00";
    const row: MockInterest = {
      id: nextInterestId++,
      name: String(body.name),
      description: (body.description as string | null) ?? null,
      keywords: (body.keywords as string[]) ?? [],
      categories: (body.categories as string[]) ?? [],
      is_active: body.is_active !== false,
      weight: 1,
      embedded: false,
      created_at: now,
      updated_at: now,
    };
    interests.push(row);
    json(res, 201, row);
    return true;
  }
  const one = /^\/interests\/(\d+)$/.exec(path);
  if (one) {
    const index = interests.findIndex((i) => i.id === Number(one[1]));
    if (index < 0) {
      json(res, 404, { detail: "unknown interest" });
      return true;
    }
    if (req.method === "DELETE") {
      interests.splice(index, 1);
      res.writeHead(204).end();
      return true;
    }
    if (req.method === "PUT") {
      const body = await readJson(req);
      interests[index] = {
        ...interests[index],
        name: String(body.name),
        description: (body.description as string | null) ?? null,
        keywords: (body.keywords as string[]) ?? [],
        categories: (body.categories as string[]) ?? [],
        is_active: body.is_active !== false,
        embedded: false,
      };
      json(res, 200, interests[index]);
      return true;
    }
  }
  if (path === "/settings" && req.method === "GET") {
    json(res, 200, settings);
    return true;
  }
  if (path === "/settings" && req.method === "PUT") {
    const body = await readJson(req);
    const categories = (body.categories as string[]) ?? [];
    settings = {
      ...settings,
      digest_size: Number(body.digest_size),
      fulltext_top_n: Number(body.fulltext_top_n),
      categories: categories.length ? categories : settings.categories_default,
      categories_overridden: categories.length > 0,
      budget_cap_usd: (body.budget_cap_usd as number | null) ?? null,
    };
    json(res, 200, settings);
    return true;
  }
  return false;
}

createServer(async (req, res) => {
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
  if (await digestRoute(req, res, url)) return;
  if (req.method === "GET" && paperRoute(res, url)) return;
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

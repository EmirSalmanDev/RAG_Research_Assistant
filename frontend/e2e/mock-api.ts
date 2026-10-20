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
import { createHash } from "node:crypto";
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

// --- comparison and timeline (e2e/explore-flow.spec.ts): invented, no model behind them ---
// Papers 1 and 2 have full text, 3 to 6 only an abstract. The comparison fills every cell
// with a claim and a citation, except that an abstract-only paper says nothing about "data".
const EXPLORE_PAPERS = Array.from({ length: 6 }, (_, i) => ({
  arxiv_id: `2611.0000${i + 1}`,
  title: `Invented Explore Paper ${i + 1}`,
  published_at: `2026-10-${String(i * 2 + 1).padStart(2, "0")}T12:00:00+00:00`,
  primary_category: "cs.CL",
  has_fulltext: i < 2,
  abs_url: abs(`2611.0000${i + 1}`),
}));
// Chunk ids are readable on purpose: 0c<paper><k>, k=1 abstract, k=2 section.
const exploreChunkId = (paper: number, kind: 1 | 2) =>
  `00000000-0000-4000-8000-0000000c${paper}${kind}00`;
const exploreSource = (paper: number, kind: 1 | 2, n: number) => {
  const p = EXPLORE_PAPERS[paper - 1];
  return {
    n,
    chunk_id: exploreChunkId(paper, kind),
    arxiv_id: p.arxiv_id,
    chunk_type: kind === 1 ? "abstract" : "section",
    section_path: kind === 1 ? null : "3 Method",
    title: p.title,
    published_at: p.published_at,
    abs_url: p.abs_url,
  };
};
// GET /chunks/{id} for the ids made by exploreChunkId.
const exploreChunk = (id: string) => {
  const m = /^00000000-0000-4000-8000-0000000c(\d)(\d)00$/.exec(id);
  const paper = Number(m?.[1]);
  const kind = Number(m?.[2]);
  if (!m || !EXPLORE_PAPERS[paper - 1] || (kind !== 1 && kind !== 2))
    return null;
  const source = exploreSource(paper, kind, 1);
  return {
    chunk_id: id,
    arxiv_id: source.arxiv_id,
    title: source.title,
    chunk_type: source.chunk_type,
    section_path: source.section_path,
    ord: 1,
    text: `Invented ${source.chunk_type} text of ${source.title}.`,
    published_at: source.published_at,
    abs_url: source.abs_url,
  };
};

function listPapers(res: ServerResponse, url: string): void {
  const query = new URL(url, "http://mock").searchParams;
  const q = (query.get("q") ?? "").toLowerCase();
  const full = query.get("has_fulltext");
  const papers = EXPLORE_PAPERS.filter(
    (p) =>
      (!q || p.title.toLowerCase().includes(q) || p.arxiv_id === q) &&
      (full === null || String(p.has_fulltext) === full),
  ).slice(0, Number(query.get("limit") ?? 20));
  json(res, 200, { papers });
}

async function exploreRoute(
  req: IncomingMessage,
  res: ServerResponse,
  url: string,
): Promise<boolean> {
  const path = url.split("?")[0];
  if (req.method === "GET" && path === "/papers") {
    listPapers(res, url);
    return true;
  }
  if (req.method === "POST" && path === "/compare") {
    const body = await readJson(req);
    const ids = (body.arxiv_ids as string[]) ?? [];
    const dims = (body.dimensions as string[]) ?? [];
    if (ids.length < 2 || ids.length > 5) {
      json(res, 422, { detail: "pick 2 to 5 papers" });
      return true;
    }
    const rows = [];
    const sources = [];
    let n = 0;
    for (const id of ids) {
      const paper = EXPLORE_PAPERS.findIndex((p) => p.arxiv_id === id) + 1;
      if (paper === 0) {
        json(res, 404, { detail: `unknown paper(s): ${id}` });
        return true;
      }
      const kind = EXPLORE_PAPERS[paper - 1].has_fulltext ? 2 : 1;
      const source = exploreSource(paper, kind, ++n);
      sources.push(source);
      rows.push({
        arxiv_id: id,
        title: EXPLORE_PAPERS[paper - 1].title,
        has_fulltext: EXPLORE_PAPERS[paper - 1].has_fulltext,
        abs_url: abs(id),
        cells: dims.map((dimension) =>
          dimension === "data" && kind === 1
            ? {
                dimension,
                text: "Not stated in the sources.",
                status: "not_stated",
                citations: [],
              }
            : {
                dimension,
                text: `Invented ${dimension} claim about paper ${paper}.`,
                status: "cited",
                citations: [
                  {
                    n: source.n,
                    chunk_id: source.chunk_id,
                    chunk_type: source.chunk_type,
                    section_path: source.section_path,
                  },
                ],
              },
        ),
      });
    }
    json(res, 200, {
      dimensions: dims,
      rows,
      sources,
      notes: {
        invalid_citations_dropped: 0,
        uncited_cells_converted: 0,
        missing_cells: 0,
        truncated_cells: 0,
        cut_sources: 0,
        top_k_used: 3,
      },
      tokens_in: 4000,
      tokens_out: 400,
      cost_usd: 0.00084,
      prompt_version: "compare_v1",
    });
    return true;
  }
  if (req.method === "POST" && path === "/timeline") {
    const body = await readJson(req);
    const points = EXPLORE_PAPERS.map((p, i) => ({
      arxiv_id: p.arxiv_id,
      title: p.title,
      published_at: p.published_at,
      citation_count: i === 0 ? 120 : null,
      topic_relevance: 1 - i / 5,
      key_score: 0.9 - i / 10,
      is_key: i === 0 || i === 3,
      line:
        i === 0
          ? "Invented line: introduced the first adapter."
          : i === 3
            ? "Invented first sentence of the abstract."
            : null,
      line_source: i === 0 ? "model" : i === 3 ? "abstract_fallback" : null,
      chunk_id: exploreChunkId(i + 1, 1),
      abs_url: p.abs_url,
    }));
    json(res, 200, {
      topic: body.topic,
      range: { first: points[0].published_at, last: points[5].published_at },
      points,
      notes: {
        papers: 6,
        key_papers: 2,
        known_citation_counts: 1,
        narrow_range: true,
        range_days: 10,
        fallback_lines: 1,
        lines_call_failed: false,
      },
      prompt_version: "timeline_v1",
      tokens_in: 2500,
      tokens_out: 120,
      cost_usd: 0.000447,
    });
    return true;
  }
  return false;
}

// --- the library (e2e/library-flow.spec.ts): invented, in memory, no model behind it ---------
// Collections and items live in memory for the life of the server. The papers are the
// explore papers above (1 and 2 with full text and an analysis, 3 to 6 abstract only) plus
// uploaded documents. The review cites every item of the collection once.
type LibItem = {
  arxiv_id: string;
  collection: string;
  status: string;
  notes: string | null;
  added_at: string;
};
const libCollections = new Map<
  string,
  { description: string | null; created_at: string }
>();
let libItems: LibItem[] = [];
const uploads = new Map<string, { title: string; year: number }>();
const uploadLog: { bytes: number; title: string }[] = [];
const LIB_NOW = "2026-10-20T08:00:00+00:00";

function libPaper(id: string) {
  const explore = EXPLORE_PAPERS.find((p) => p.arxiv_id === id);
  if (explore)
    return {
      arxiv_id: id,
      title: explore.title,
      published_at: explore.published_at,
      evidence: explore.has_fulltext ? "full_text" : "abstract_only",
      abs_url: explore.abs_url,
      authors: ["Ada Example"],
      index: EXPLORE_PAPERS.indexOf(explore) + 1,
    };
  const upload = uploads.get(id);
  if (upload)
    return {
      arxiv_id: id,
      title: upload.title,
      published_at: `${upload.year}-01-01T00:00:00+00:00`,
      evidence: "uploaded",
      abs_url: null,
      authors: [] as string[],
      index: 0,
    };
  return null;
}
function libItemOut(item: LibItem) {
  const paper = libPaper(item.arxiv_id)!;
  return {
    ...item,
    title: paper.title,
    authors: paper.authors,
    published_at: paper.published_at,
    evidence: paper.evidence,
    abs_url: paper.abs_url,
  };
}
const libNames = () =>
  new Set([...libCollections.keys(), ...libItems.map((i) => i.collection)]);
function libCollectionOut(name: string) {
  const row = libCollections.get(name);
  return {
    name,
    description: row?.description ?? null,
    paper_count: libItems.filter((i) => i.collection === name).length,
    created_at: row?.created_at ?? null,
  };
}
const libOrdered = (collection: string) =>
  libItems
    .filter((i) => i.collection === collection)
    .sort((a, b) => a.arxiv_id.localeCompare(b.arxiv_id));
// The key a paper has in the .bib export AND in the review's references.
const libKey = (id: string) =>
  `example2026${(libPaper(id)?.title ?? "paper").toLowerCase().replace(/[^a-z0-9]/g, "")}`;
function libBibtex(collection: string): string {
  return libOrdered(collection)
    .map((item) => {
      const paper = libPaper(item.arxiv_id)!;
      const tail = paper.abs_url
        ? `  eprint = {${paper.arxiv_id}},\n  archivePrefix = {arXiv},\n  url = {${paper.abs_url}},\n`
        : "  note = {Uploaded document},\n";
      return `@misc{${libKey(item.arxiv_id)},\n  author = {Example, Ada},\n  title = {{${paper.title}}},\n  year = {${paper.published_at.slice(0, 4)}},\n${tail}}\n`;
    })
    .join("\n");
}
// Chunk ids of the library's citations: 0d<paper><kind>00, kind 1 abstract, 3 analysis.
const libChunkId = (paper: number, kind: 1 | 3) =>
  `00000000-0000-4000-8000-0000000d${paper}${kind}00`;
function libChunk(id: string) {
  const m = /^00000000-0000-4000-8000-0000000d(\d)(\d)00$/.exec(id);
  const paper = m ? EXPLORE_PAPERS[Number(m[1]) - 1] : undefined;
  if (!m || !paper || (m[2] !== "1" && m[2] !== "3")) return null;
  const kind = m[2] === "1" ? "abstract" : "analysis";
  return {
    chunk_id: id,
    arxiv_id: paper.arxiv_id,
    title: paper.title,
    chunk_type: kind,
    section_path: kind === "abstract" ? "Abstract" : "Analysis > Summary",
    ord: kind === "abstract" ? 0 : 10000,
    text: `Invented ${kind} text of ${paper.title}.`,
    published_at: paper.published_at,
    abs_url: paper.abs_url,
  };
}

function libReview(collection: string) {
  const items = libOrdered(collection);
  const refs = items.map((item, i) => {
    const paper = libPaper(item.arxiv_id)!;
    const analysed =
      paper.index > 0 && EXPLORE_PAPERS[paper.index - 1].has_fulltext;
    return {
      n: i + 1,
      arxiv_id: item.arxiv_id,
      bibtex_key: libKey(item.arxiv_id),
      title: paper.title,
      authors: paper.authors,
      year: Number(paper.published_at.slice(0, 4)),
      chunk_id:
        paper.index > 0
          ? libChunkId(paper.index, analysed ? 3 : 1)
          : libChunkId(9, 1),
      section_path: analysed ? "Analysis > Summary" : "Abstract",
      published_at: paper.published_at,
      chunk_type: analysed ? "analysis" : "abstract",
      evidence: analysed ? "analysis" : "abstract",
      abs_url: paper.abs_url,
      cited_count: 1,
    };
  });
  const sentences = [
    {
      text: "Several lines of work address adapters.",
      kind: "transition",
      source_numbers: [],
    },
    ...refs.map((r) => ({
      text: `Example (${r.year}) studied ${r.title.toLowerCase()}.`,
      kind: "claim",
      source_numbers: [r.n],
    })),
  ];
  return {
    collection,
    focus: null,
    sections: [{ heading: "Adapters", paragraphs: [{ sentences }] }],
    references: refs,
    counts: {
      papers: refs.length,
      papers_without_analysis: refs.filter((r) => r.evidence === "abstract")
        .length,
      papers_omitted: 0,
      papers_uncitable: 0,
      claims: refs.length,
      transitions: 1,
      dropped_uncited: 2,
      dropped_transitions: 0,
      invalid_citations_dropped: 0,
      map_calls: 0,
      map_notes_dropped: 0,
    },
    tokens_in: 2400,
    tokens_out: 300,
    cost_usd: 0.00054,
    prompt_versions: { write: "review_write_v1" },
  };
}

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve) => {
    const parts: Buffer[] = [];
    req.on("data", (chunk: Buffer) => parts.push(chunk));
    req.on("end", () => resolve(Buffer.concat(parts)));
  });
}

async function libraryRoute(
  req: IncomingMessage,
  res: ServerResponse,
  url: string,
): Promise<boolean> {
  const [path, queryText = ""] = url.split("?");
  const query = new URLSearchParams(queryText);
  const method = req.method ?? "GET";
  if (method === "GET" && path === "/__uploads") {
    json(res, 200, uploadLog);
    return true;
  }
  if (!path.startsWith("/library/")) return false;
  const decoded = decodeURIComponent(path);

  if (path === "/library/collections" && method === "GET") {
    json(res, 200, [...libNames()].map(libCollectionOut));
    return true;
  }
  if (path === "/library/collections" && method === "POST") {
    const body = await readJson(req);
    const name = String(body.name ?? "").trim();
    if (!name)
      return (
        json(res, 422, { detail: "a collection name has 1 to 100 characters" }),
        true
      );
    if (libNames().has(name))
      return (
        json(res, 409, {
          detail: `a collection named '${name}' exists already`,
        }),
        true
      );
    libCollections.set(name, {
      description: (body.description as string | null) ?? null,
      created_at: LIB_NOW,
    });
    json(res, 201, libCollectionOut(name));
    return true;
  }
  const one = /^\/library\/collections\/(.+)$/.exec(decoded);
  if (one) {
    const name = one[1];
    if (!libNames().has(name))
      return (
        json(res, 404, { detail: `no collection named '${name}'` }),
        true
      );
    if (method === "PUT") {
      const body = await readJson(req);
      const to = body.name ? String(body.name).trim() : name;
      if (to !== name && libNames().has(to))
        return (
          json(res, 409, {
            detail: `a collection named '${to}' exists already`,
          }),
          true
        );
      const row = libCollections.get(name) ?? {
        description: null,
        created_at: LIB_NOW,
      };
      libCollections.delete(name);
      libCollections.set(to, row);
      for (const item of libItems)
        if (item.collection === name) item.collection = to;
      json(res, 200, libCollectionOut(to));
      return true;
    }
    if (method === "DELETE") {
      const removed = libItems.filter((i) => i.collection === name).length;
      libItems = libItems.filter((i) => i.collection !== name);
      libCollections.delete(name);
      json(res, 200, { name, items_removed: removed });
      return true;
    }
  }
  if (path === "/library/items" && method === "GET") {
    const collection = query.get("collection") ?? "";
    if (!libNames().has(collection))
      return (
        json(res, 404, { detail: `no collection named '${collection}'` }),
        true
      );
    json(res, 200, libOrdered(collection).map(libItemOut));
    return true;
  }
  if (path === "/library/items" && method === "POST") {
    const body = await readJson(req);
    const id = /\d{4}\.\d{4,5}/.exec(String(body.reference ?? ""))?.[0] ?? "";
    if (!libPaper(id))
      return (
        json(res, 404, {
          detail: `${id || body.reference} is not in the database yet. Ingest the paper first (adding a paper that is not in the database is a later step).`,
        }),
        true
      );
    const collection = String(body.collection);
    const existing = libItems.find(
      (i) => i.arxiv_id === id && i.collection === collection,
    );
    if (existing) return (json(res, 200, libItemOut(existing)), true);
    const item: LibItem = {
      arxiv_id: id,
      collection,
      status: String(body.status ?? "to_read"),
      notes: null,
      added_at: LIB_NOW,
    };
    libItems.push(item);
    json(res, 201, libItemOut(item));
    return true;
  }
  const itemRoute = /^\/library\/items\/(.+)$/.exec(decoded);
  if (itemRoute) {
    const item = libItems.find(
      (i) =>
        i.arxiv_id === itemRoute[1] && i.collection === query.get("collection"),
    );
    if (!item)
      return (
        json(res, 404, { detail: `${itemRoute[1]} is not in that collection` }),
        true
      );
    if (method === "PATCH") {
      const body = await readJson(req);
      if ("status" in body && body.status) item.status = String(body.status);
      if ("notes" in body) item.notes = body.notes ? String(body.notes) : null;
      if (body.collection) item.collection = String(body.collection);
      json(res, 200, libItemOut(item));
      return true;
    }
    if (method === "DELETE") {
      libItems = libItems.filter((i) => i !== item);
      res.writeHead(204).end();
      return true;
    }
  }
  if (path === "/library/bibtex" && method === "GET") {
    const collection = query.get("collection") ?? "";
    if (!libNames().has(collection))
      return (json(res, 404, { detail: "no such collection" }), true);
    const slug = collection
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-|-$/g, "");
    res.writeHead(200, {
      "content-type": "text/plain; charset=utf-8",
      "content-disposition": `attachment; filename="${slug || "library"}.bib"`,
    });
    res.end(libBibtex(collection));
    return true;
  }
  if (path === "/library/review" && method === "POST") {
    const body = await readJson(req);
    const collection = String(body.collection ?? "");
    if (!libNames().has(collection))
      return (
        json(res, 404, { detail: `no collection named '${collection}'` }),
        true
      );
    if (libOrdered(collection).length < 2)
      return (
        json(res, 422, {
          detail: "a review needs at least 2 papers with something to cite",
        }),
        true
      );
    const review = libReview(collection);
    json(res, 200, { ...review, focus: (body.focus as string | null) ?? null });
    return true;
  }
  if (path === "/library/upload" && method === "POST") {
    const bytes = await readBody(req);
    const id = `local-${createHash("sha256").update(bytes).digest("hex").slice(0, 12)}`;
    const title = query.get("title") ?? "";
    const year = Number(query.get("year") ?? 2026);
    const created = !uploads.has(id);
    uploadLog.push({ bytes: bytes.length, title });
    if (created) uploads.set(id, { title, year });
    const collection = query.get("collection");
    if (
      collection &&
      !libItems.some((i) => i.arxiv_id === id && i.collection === collection)
    )
      libItems.push({
        arxiv_id: id,
        collection,
        status: "to_read",
        notes: null,
        added_at: LIB_NOW,
      });
    json(res, created ? 201 : 200, {
      arxiv_id: id,
      title: uploads.get(id)!.title,
      year,
      created,
      collection,
      abs_url: null,
      chunks: created
        ? { abstract: 1, section: 4 }
        : { abstract: 1, section: 4 },
      embedded: created ? 5 : 0,
      embedding_requests: created ? 1 : 0,
      tokens_in: created ? 1200 : 0,
      cost_usd: created ? 0.000024 : 0,
    });
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
  if (await exploreRoute(req, res, url)) return;
  if (await libraryRoute(req, res, url)) return;
  if (req.method === "GET" && paperRoute(res, url)) return;
  const chunk = /^\/chunks\/([0-9a-f-]+)$/.exec(url);
  if (req.method === "GET" && chunk) {
    const explore = exploreChunk(chunk[1]) ?? libChunk(chunk[1]);
    if (explore) return json(res, 200, explore);
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

// The endpoint functions against a MOCKED fetch: the tests check what would be sent and how
// answers and errors are handled. Nothing here touches a network or a real API.
import { strict as assert } from "node:assert";
import { test } from "node:test";

import createClient from "openapi-fetch";

import { ApiError } from "./errors.ts";
import {
  fetchChunk,
  fetchPaper,
  fetchPaperChunks,
  fetchPaperDetails,
  fetchPaperRelated,
  fetchSearch,
} from "./endpoints.ts";
import type { paths } from "./schema";

interface Call {
  method: string;
  url: URL;
}

function mocked(respond: (url: URL) => Response) {
  const calls: Call[] = [];
  const client = createClient<paths>({
    baseUrl: "http://api.test/api",
    fetch: async (request: Request) => {
      const url = new URL(request.url);
      calls.push({ method: request.method, url });
      return respond(url);
    },
  });
  return { client, calls };
}

const HIT = {
  chunk_id: "00000000-0000-4000-8000-0000000000a1",
  arxiv_id: "2609.00001",
  title: "An Invented Paper",
  section_path: null,
  chunk_type: "abstract",
  published_at: "2026-09-01T12:00:00+00:00",
  score: 0.5,
  text: "Invented text.",
  abs_url: "https://arxiv.org/abs/2609.00001",
};

test("search sends the question, repeated categories, dates, a limit and rerank=false", async () => {
  const { client, calls } = mocked(() =>
    Response.json({
      query: "adapters",
      mode: "hybrid",
      reranked: false,
      hits: [HIT],
    }),
  );
  const result = await fetchSearch(client, {
    q: "low-rank adapters?",
    categories: ["cs.CL", "cs.LG"],
    from: "2026-09-01",
    to: "2026-09-30",
  });
  assert.equal(result.hits.length, 1);
  assert.equal(calls.length, 1, "one request per search");
  const call = calls[0]!;
  assert.equal(call.method, "GET");
  assert.equal(call.url.pathname, "/api/search");
  assert.equal(call.url.searchParams.get("q"), "low-rank adapters?");
  assert.deepEqual(call.url.searchParams.getAll("category"), [
    "cs.CL",
    "cs.LG",
  ]);
  assert.equal(call.url.searchParams.get("published_from"), "2026-09-01");
  assert.equal(call.url.searchParams.get("published_to"), "2026-09-30");
  assert.equal(call.url.searchParams.get("limit"), "20");
  assert.equal(
    call.url.searchParams.get("rerank"),
    "false",
    "never rerank from the UI",
  );
});

test("search leaves out filters that are not set", async () => {
  const { client, calls } = mocked(() =>
    Response.json({ query: "x", mode: "hybrid", reranked: false, hits: [] }),
  );
  await fetchSearch(client, { q: "x", categories: [], from: null, to: null });
  const keys = [...calls[0]!.url.searchParams.keys()].sort();
  assert.deepEqual(keys, ["limit", "q", "rerank"]);
});

test("search errors become ApiError with the API's own message", async () => {
  const { client } = mocked(() =>
    Response.json(
      { detail: "OPENAI_API_KEY is not set (put it in .env)" },
      { status: 503 },
    ),
  );
  await assert.rejects(
    fetchSearch(client, { q: "x", categories: [], from: null, to: null }),
    (err) => {
      assert.ok(err instanceof ApiError);
      assert.equal(err.status, 503);
      assert.match(err.message, /OPENAI_API_KEY/);
      return true;
    },
  );
});

test("paper endpoints use the DB-only routes and encode the id", async () => {
  const paper = {
    arxiv_id: "2609.00001",
    version: 1,
    title: "T",
    abstract: "A",
    authors: [],
    categories: [],
    primary_category: null,
    published_at: "2026-09-01T00:00:00+00:00",
    hf_upvotes: null,
    citation_count: null,
    github_url: null,
    project_url: null,
    fulltext_status: "none",
    abs_url: "https://arxiv.org/abs/2609.00001",
  };
  const { client, calls } = mocked((url) =>
    url.pathname.endsWith("/chunks")
      ? Response.json({ arxiv_id: "2609.00001", chunks: [] })
      : Response.json(paper),
  );
  assert.equal((await fetchPaper(client, "2609.00001")).title, "T");
  assert.deepEqual((await fetchPaperChunks(client, "2609.00001")).chunks, []);
  await fetchPaper(client, "hep-th/9901001");
  assert.deepEqual(
    calls.map((c) => c.url.pathname + c.url.search),
    [
      "/api/papers/2609.00001",
      "/api/papers/2609.00001/chunks",
      "/api/papers/hep-th%2F9901001",
    ],
  );
  assert.ok(
    calls.every((c) => !c.url.pathname.startsWith("/api/search")),
    "no embedding route is called",
  );
});

test("an unknown paper or chunk is a readable 404", async () => {
  const { client } = mocked(() =>
    Response.json({ detail: "unknown paper" }, { status: 404 }),
  );
  await assert.rejects(fetchPaper(client, "0000.00000"), /unknown paper/);
  const noDetail = mocked(() => new Response("nope", { status: 404 }));
  await assert.rejects(
    fetchChunk(noDetail.client, "00000000-0000-4000-8000-000000000000"),
    /passage was not found/,
  );
});

test("paper details and related papers ask the right paths; 404 is an ApiError", async () => {
  const { client, calls } = mocked((url) =>
    url.pathname.endsWith("/details")
      ? Response.json({
          arxiv_id: "2609.00001",
          analysis_status: "none",
          analysis: null,
          repos: [],
          models: [],
          datasets: [],
          spaces: [],
        })
      : url.pathname.includes("0000.00000")
        ? Response.json({ detail: "unknown paper" }, { status: 404 })
        : Response.json({
            arxiv_id: "2609.00001",
            similar_status: "ok",
            similar: [],
            cites: [],
            cited_by: [],
          }),
  );
  const details = await fetchPaperDetails(client, "2609.00001");
  assert.equal(details.analysis_status, "none");
  const related = await fetchPaperRelated(client, "2609.00001");
  assert.equal(related.similar_status, "ok");
  assert.equal(calls[0]!.url.pathname, "/api/papers/2609.00001/details");
  assert.equal(calls[1]!.url.pathname, "/api/papers/2609.00001/related");
  assert.equal(calls[1]!.url.searchParams.get("limit"), "10");
  await assert.rejects(
    fetchPaperRelated(client, "0000.00000"),
    (err) => err instanceof ApiError && err.status === 404,
  );
  assert.equal(calls.length, 3, "one request each, no retries");
});

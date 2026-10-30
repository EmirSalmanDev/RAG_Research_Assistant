// The endpoint functions against a MOCKED fetch: the tests check what would be sent and how
// answers and errors are handled. Nothing here touches a network or a real API.
import { strict as assert } from "node:assert";
import { test } from "node:test";

import createClient from "openapi-fetch";

import { ApiError } from "./errors.ts";
import {
  fetchChunk,
  fetchPaper,
  fetchPapers,
  postCompare,
  postTimeline,
  fetchPaperChunks,
  fetchPaperDetails,
  fetchPaperRelated,
  fetchSearch,
  actOnEntry,
  createInterest,
  deleteInterest,
  fetchDigest,
  fetchInterests,
  fetchSettings,
  fetchStatus,
  saveSettings,
  updateInterest,
  addItem,
  createCollection,
  deleteCollection,
  deleteUpload,
  fetchBibtex,
  fetchCollections,
  fetchItems,
  postReview,
  removeItem,
  updateCollection,
  updateItem,
  uploadPdf,
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

// --- digest, interests and settings -----------------------------------------------------

test("digest: the date is sent only when given", async () => {
  const body = {
    requested_date: "2026-10-19",
    digest_date: null,
    is_fallback: false,
    has_interests: false,
    recent_dates: [],
    entries: [],
  };
  const { client, calls } = mocked(() => Response.json(body));
  assert.deepEqual(await fetchDigest(client), body);
  await fetchDigest(client, "2026-10-17");
  assert.equal(calls[0].method, "GET");
  assert.equal(calls[0].url.pathname, "/api/digest");
  assert.equal(calls[0].url.search, "");
  assert.equal(calls[1].url.searchParams.get("date"), "2026-10-17");
});

test("save, dismiss and undo are POSTs to their own path with the digest date", async () => {
  const { client, calls } = mocked(() =>
    Response.json({ arxiv_id: "2610.00001", status: "saved" }),
  );
  await actOnEntry(client, "save", "2026-10-19", "2610.00001");
  await actOnEntry(client, "dismiss", "2026-10-19", "2610.00001");
  await actOnEntry(client, "undo", "2026-10-19", "2610.00001");
  assert.deepEqual(
    calls.map((c) => [
      c.method,
      c.url.pathname,
      c.url.searchParams.get("date"),
    ]),
    [
      ["POST", "/api/digest/2610.00001/save", "2026-10-19"],
      ["POST", "/api/digest/2610.00001/dismiss", "2026-10-19"],
      ["POST", "/api/digest/2610.00001/undo", "2026-10-19"],
    ],
  );
});

test("an action on an entry that is not in the digest is an ApiError with the status", async () => {
  const { client } = mocked(() =>
    Response.json(
      { detail: "that paper is not in this digest" },
      { status: 404 },
    ),
  );
  await assert.rejects(
    actOnEntry(client, "save", "2026-10-19", "2610.99999"),
    (error: unknown) =>
      error instanceof ApiError &&
      error.status === 404 &&
      /not in this digest/.test(error.message),
  );
});

test("interests: create, update, delete and list use the right verbs and bodies", async () => {
  const row = { id: 3, name: "Alpha" };
  const { client, calls } = mocked((url) =>
    url.pathname === "/api/interests/3" && calls.at(-1)?.method === "DELETE"
      ? new Response(null, { status: 204 })
      : Response.json(
          url.pathname === "/api/interests" && calls.at(-1)?.method === "GET"
            ? [row]
            : row,
        ),
  );
  const body = {
    name: "Alpha",
    description: null,
    keywords: ["a"],
    categories: [],
    is_active: true,
  };
  assert.deepEqual(await createInterest(client, body), row);
  await updateInterest(client, 3, body);
  await deleteInterest(client, 3);
  await fetchInterests(client);
  assert.deepEqual(
    calls.map((c) => [c.method, c.url.pathname]),
    [
      ["POST", "/api/interests"],
      ["PUT", "/api/interests/3"],
      ["DELETE", "/api/interests/3"],
      ["GET", "/api/interests"],
    ],
  );
});

test("a duplicate interest name shows the server's sentence", async () => {
  const { client } = mocked(() =>
    Response.json(
      { detail: "an interest with this name exists" },
      { status: 409 },
    ),
  );
  await assert.rejects(
    createInterest(client, {
      name: "Alpha",
      description: null,
      keywords: [],
      categories: [],
      is_active: true,
    }),
    (error: unknown) =>
      error instanceof ApiError &&
      error.status === 409 &&
      error.message === "an interest with this name exists",
  );
});

test("settings: read and save", async () => {
  const settings = {
    digest_size: 20,
    categories: ["cs.CL"],
    categories_default: ["cs.CL"],
    categories_overridden: false,
    fulltext_top_n: 20,
    budget_cap_usd: null,
    month_to_date_usd: 0.5,
  };
  const { client, calls } = mocked(() => Response.json(settings));
  assert.deepEqual(await fetchSettings(client), settings);
  await saveSettings(client, {
    digest_size: 5,
    fulltext_top_n: 5,
    categories: [],
    budget_cap_usd: null,
  });
  assert.deepEqual(
    calls.map((c) => [c.method, c.url.pathname]),
    [
      ["GET", "/api/settings"],
      ["PUT", "/api/settings"],
    ],
  );
});

test("the status read asks /status and a failed answer becomes an ApiError", async () => {
  const body = {
    stores: { postgres: "up", qdrant: "up", redis: "up" },
    errors: {},
  };
  const ok = mocked(() => Response.json(body));
  assert.deepEqual(await fetchStatus(ok.client), body);
  assert.deepEqual(
    ok.calls.map((c) => [c.method, c.url.pathname]),
    [["GET", "/api/status"]],
  );
  const bad = mocked(() =>
    Response.json({ detail: "database is down" }, { status: 503 }),
  );
  await assert.rejects(fetchStatus(bad.client), (err: Error) => {
    assert.equal(err.name, "ApiError");
    assert.equal(err.message, "database is down");
    return true;
  });
});

// --- paper comparison and timeline -----------------------------------------------------------

test("the paper list sends the trimmed query, a limit and the full-text flag, and returns the papers", async () => {
  const item = {
    arxiv_id: "2609.00001",
    title: "An Invented Paper",
    published_at: "2026-09-01T12:00:00+00:00",
    primary_category: "cs.CL",
    has_fulltext: true,
    abs_url: "https://arxiv.org/abs/2609.00001",
  };
  const { client, calls } = mocked(() => Response.json({ papers: [item] }));
  const papers = await fetchPapers(client, {
    q: "  attention ",
    hasFulltext: true,
  });
  assert.deepEqual(papers, [item]);
  const url = calls[0]!.url;
  assert.equal(url.pathname, "/api/papers");
  assert.equal(url.searchParams.get("q"), "attention");
  assert.equal(url.searchParams.get("limit"), "20");
  assert.equal(url.searchParams.get("has_fulltext"), "true");

  const { client: plain, calls: plainCalls } = mocked(() =>
    Response.json({ papers: [] }),
  );
  await fetchPapers(plain, { q: "" });
  assert.equal(plainCalls[0]!.url.searchParams.has("has_fulltext"), false);
});

test("compare posts the ids and dimensions once and surfaces the API's message on failure", async () => {
  const { client, calls } = mocked(() => Response.json({ rows: [] }));
  await postCompare(client, { arxiv_ids: ["a", "b"], dimensions: ["task"] });
  assert.equal(calls.length, 1);
  assert.equal(calls[0]!.method, "POST");
  assert.equal(calls[0]!.url.pathname, "/api/compare");

  const failing = mocked(() =>
    Response.json({ detail: "unknown paper(s): 9999.99999" }, { status: 404 }),
  );
  await assert.rejects(
    postCompare(failing.client, {
      arxiv_ids: ["a", "9999.99999"],
      dimensions: ["task"],
    }),
    (err: unknown) =>
      err instanceof ApiError &&
      err.status === 404 &&
      err.message === "unknown paper(s): 9999.99999",
  );
  assert.equal(failing.calls.length, 1); // no retry: a retry would be a second paid call
});

test("timeline posts the topic and options once; a 503 keeps the server's sentence", async () => {
  const { client, calls } = mocked(() => Response.json({ points: [] }));
  await postTimeline(client, {
    topic: "adapters",
    max_papers: 30,
    key_papers: 8,
    categories: [],
  });
  assert.equal(calls[0]!.url.pathname, "/api/timeline");

  const down = mocked(() =>
    Response.json(
      { detail: "OPENAI_API_KEY is not set (put it in .env)" },
      { status: 503 },
    ),
  );
  await assert.rejects(
    postTimeline(down.client, {
      topic: "adapters",
      max_papers: 30,
      key_papers: 8,
    }),
    (err: unknown) =>
      err instanceof ApiError && /OPENAI_API_KEY/.test(err.message),
  );
});

// --- the library --------------------------------------------------------------------------------

test("collections: list, create, rename (the path is encoded) and delete", async () => {
  const { client, calls } = mocked((url) =>
    url.pathname.endsWith("/library/collections") &&
    calls.at(-1)?.method === "GET"
      ? Response.json([
          {
            name: "Saved",
            description: null,
            paper_count: 2,
            created_at: null,
          },
        ])
      : Response.json({
          name: "Thesis",
          description: null,
          paper_count: 0,
          created_at: "2026-10-20T00:00:00Z",
        }),
  );
  const list = await fetchCollections(client);
  assert.equal(list[0]!.name, "Saved");
  await createCollection(client, { name: "Thesis", description: "Ch. 2" });
  await updateCollection(client, "RL / agents", { name: "RL agents" });
  await deleteCollection(client, "Thesis");
  assert.deepEqual(
    calls.map((c) => `${c.method} ${c.url.pathname}`),
    [
      "GET /api/library/collections",
      "POST /api/library/collections",
      "PUT /api/library/collections/RL%20%2F%20agents",
      "DELETE /api/library/collections/Thesis",
    ],
  );
});

test("a duplicate collection name is an ApiError with the server's sentence", async () => {
  const { client } = mocked(() =>
    Response.json(
      { detail: "a collection named 'Thesis' exists already" },
      { status: 409 },
    ),
  );
  await assert.rejects(
    createCollection(client, { name: "Thesis" }),
    (err: unknown) =>
      err instanceof ApiError &&
      err.status === 409 &&
      /exists already/.test(err.message),
  );
});

test("items: the collection goes in the query; an old-style id is encoded in the path", async () => {
  const item = {
    arxiv_id: "hep-th/9901001",
    collection: "T",
    status: "read",
    notes: null,
    added_at: "2026-10-20T00:00:00Z",
    title: "Old",
    authors: [],
    published_at: "1999-01-01T00:00:00Z",
    evidence: "abstract_only",
    abs_url: "https://arxiv.org/abs/hep-th/9901001",
  };
  const { client, calls } = mocked(() => Response.json(item));
  await fetchItems(client, "My thesis");
  await updateItem(client, "hep-th/9901001", "My thesis", {
    status: "read",
    notes: null,
  });
  const [list, patch] = calls;
  assert.equal(list!.url.searchParams.get("collection"), "My thesis");
  assert.equal(patch!.method, "PATCH");
  assert.equal(patch!.url.pathname, "/api/library/items/hep-th%2F9901001");
  assert.equal(patch!.url.searchParams.get("collection"), "My thesis");
});

test("adding an unknown paper is a 404 whose message says to ingest it first", async () => {
  const { client } = mocked(() =>
    Response.json(
      {
        detail: "2610.99999 is not in the database yet. Ingest the paper first",
      },
      { status: 404 },
    ),
  );
  await assert.rejects(
    addItem(client, {
      reference: "2610.99999",
      collection: "T",
      status: "to_read",
    }),
    (err: unknown) =>
      err instanceof ApiError &&
      err.status === 404 &&
      /Ingest the paper first/.test(err.message),
  );
});

test("removing an item sends DELETE with the collection and expects no body", async () => {
  const { client, calls } = mocked(() => new Response(null, { status: 204 }));
  await removeItem(client, "2610.00001", "T");
  assert.equal(calls[0]!.method, "DELETE");
  assert.equal(calls[0]!.url.pathname, "/api/library/items/2610.00001");
  assert.equal(calls[0]!.url.searchParams.get("collection"), "T");
});

test("the .bib comes back as text with the suggested file name", async () => {
  const { client, calls } = mocked(
    () =>
      new Response("@misc{a2020b,\n}\n", {
        headers: {
          "content-type": "text/plain; charset=utf-8",
          "content-disposition": 'attachment; filename="my-thesis.bib"',
        },
      }),
  );
  const bib = await fetchBibtex(client, "My thesis");
  assert.equal(bib.text, "@misc{a2020b,\n}\n");
  assert.equal(bib.filename, "my-thesis.bib");
  assert.equal(calls[0]!.url.pathname, "/api/library/bibtex");
  const empty = mocked(
    () => new Response("", { headers: { "content-type": "text/plain" } }),
  );
  assert.equal((await fetchBibtex(empty.client, "Empty")).text, "");
});

test("the review is one POST with the request body; errors keep the server's sentence", async () => {
  const { client, calls } = mocked(() =>
    Response.json(
      { detail: "OPENAI_API_KEY is not set (put it in .env)" },
      { status: 503 },
    ),
  );
  await assert.rejects(
    postReview(client, { collection: "T", focus: null, max_papers: 30 }),
    (err: unknown) =>
      err instanceof ApiError &&
      err.status === 503 &&
      /OPENAI_API_KEY/.test(err.message),
  );
  assert.equal(calls.length, 1, "no retry of a paid call");
  assert.equal(calls[0]!.method, "POST");
  assert.equal(calls[0]!.url.pathname, "/api/library/review");
});

test("the upload sends the raw file once, with the metadata in the query", async () => {
  const sent: { url: string; init: RequestInit }[] = [];
  const fakeFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    sent.push({ url: String(input), init: init ?? {} });
    return Response.json(
      { arxiv_id: "local-0123456789ab", created: true },
      { status: 201 },
    );
  }) as typeof fetch;
  const file = new Blob(["%PDF-1.7 invented"], { type: "application/pdf" });
  const out = await uploadPdf(
    fakeFetch,
    "/api",
    { title: "A paper", year: 2021, collection: "T" },
    file,
  );
  assert.equal(out.arxiv_id, "local-0123456789ab");
  assert.equal(sent.length, 1);
  const url = new URL(sent[0]!.url, "http://x");
  assert.equal(url.pathname, "/api/library/upload");
  assert.deepEqual(Object.fromEntries(url.searchParams), {
    title: "A paper",
    year: "2021",
    collection: "T",
  });
  assert.equal(sent[0]!.init.method, "POST");
  assert.equal(
    sent[0]!.init.body,
    file,
    "the body is the file itself, not a form",
  );
  assert.deepEqual(sent[0]!.init.headers, {
    "Content-Type": "application/pdf",
  });
});

test("an upload error becomes an ApiError with the detail (413, 429, no JSON)", async () => {
  const failing = (status: number, body: BodyInit | null) =>
    (async () => new Response(body, { status })) as unknown as typeof fetch;
  const file = new Blob(["x"]);
  await assert.rejects(
    uploadPdf(
      failing(413, JSON.stringify({ detail: "the upload is over the limit" })),
      "/api",
      { title: "T" },
      file,
    ),
    (err: unknown) =>
      err instanceof ApiError &&
      err.status === 413 &&
      /over the limit/.test(err.message),
  );
  await assert.rejects(
    uploadPdf(
      failing(502, "<html>bad gateway</html>"),
      "/api",
      { title: "T" },
      file,
    ),
    (err: unknown) =>
      err instanceof ApiError &&
      err.status === 502 &&
      /HTTP 502/.test(err.message),
  );
});

test("deleting an upload is one DELETE", async () => {
  const { client, calls } = mocked(() =>
    Response.json({
      arxiv_id: "local-0123456789ab",
      chunks: 3,
      points: 3,
      library_items: 1,
    }),
  );
  const out = await deleteUpload(client, "local-0123456789ab");
  assert.equal(out.points, 3);
  assert.equal(calls[0]!.method, "DELETE");
  assert.equal(
    calls[0]!.url.pathname,
    "/api/library/uploads/local-0123456789ab",
  );
});

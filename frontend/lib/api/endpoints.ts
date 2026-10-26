// The read endpoints the pages use, written against an injected client so tests can give
// it a mocked fetch (no real request). lib/api/queries.ts binds them to the real client.
// Plain TypeScript apart from type-only imports.

import type { Client } from "openapi-fetch";

import type { SearchParams } from "../search/params.ts";
import { RESULT_LIMIT } from "../search/params.ts";
import { ApiError, apiErrorMessage, detailOf } from "./errors.ts";
import type { components, paths } from "./schema";

export type ApiClient = Client<paths>;
export type SearchResponse = components["schemas"]["SearchResponse"];
export type SearchHit = components["schemas"]["SearchHitOut"];
export type PaperOut = components["schemas"]["PaperOut"];
export type ChunkOut = components["schemas"]["ChunkOut"];
export type PaperChunksOut = components["schemas"]["PaperChunksOut"];
export type PaperDetailsOut = components["schemas"]["PaperDetailsOut"];
export type RelatedOut = components["schemas"]["RelatedOut"];
export type RelatedPaperOut = components["schemas"]["RelatedPaperOut"];

/** How many similar papers the paper page asks for. */
export const SIMILAR_LIMIT = 10;

/**
 * GET /search. `rerank=false` is sent explicitly: the cross-encoder takes about 20 s on a
 * CPU, and one search costs one embedding request, so nothing here repeats or reranks.
 */
export async function fetchSearch(
  client: ApiClient,
  params: SearchParams,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const { data, error, response } = await client.GET("/search", {
    params: {
      query: {
        q: params.q,
        limit: RESULT_LIMIT,
        category: params.categories.length > 0 ? params.categories : undefined,
        published_from: params.from ?? undefined,
        published_to: params.to ?? undefined,
        rerank: false,
      },
    },
    signal,
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "search"),
  );
}

/** GET /papers/{arxiv_id}: metadata and abstract, from Postgres only. */
export async function fetchPaper(
  client: ApiClient,
  arxivId: string,
): Promise<PaperOut> {
  const { data, error, response } = await client.GET("/papers/{arxiv_id}", {
    params: { path: { arxiv_id: arxivId } },
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper"),
  );
}

/** GET /papers/{arxiv_id}/chunks: the paper's passages in reading order, Postgres only. */
export async function fetchPaperChunks(
  client: ApiClient,
  arxivId: string,
): Promise<PaperChunksOut> {
  const { data, error, response } = await client.GET(
    "/papers/{arxiv_id}/chunks",
    {
      params: { path: { arxiv_id: arxivId } },
    },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper"),
  );
}

/**
 * GET /papers/{arxiv_id}/details: the analysis (or whether one is queued), repos and
 * Hugging Face models, datasets and Spaces. Postgres only: opening a page costs nothing.
 */
export async function fetchPaperDetails(
  client: ApiClient,
  arxivId: string,
): Promise<PaperDetailsOut> {
  const { data, error, response } = await client.GET(
    "/papers/{arxiv_id}/details",
    { params: { path: { arxiv_id: arxivId } } },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper"),
  );
}

/**
 * GET /papers/{arxiv_id}/related: similar papers (Qdrant neighbours of the stored abstract
 * vector, no embedding call), the papers it cites and the papers citing it.
 */
export async function fetchPaperRelated(
  client: ApiClient,
  arxivId: string,
): Promise<RelatedOut> {
  const { data, error, response } = await client.GET(
    "/papers/{arxiv_id}/related",
    {
      params: {
        path: { arxiv_id: arxivId },
        query: { limit: SIMILAR_LIMIT },
      },
    },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper"),
  );
}

/** GET /chunks/{chunk_id}: one passage (what a citation chip opens). */
export async function fetchChunk(
  client: ApiClient,
  chunkId: string,
): Promise<ChunkOut> {
  const { data, error, response } = await client.GET("/chunks/{chunk_id}", {
    params: { path: { chunk_id: chunkId } },
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "passage"),
  );
}

// --- the daily digest, interests and settings ---------------------------------------------
// None of these calls OpenAI. GET /digest only reads; Save, Dismiss and Undo change one
// entry and one interest weight on the server.

export type DigestOut = components["schemas"]["DigestOut"];
export type DigestEntryOut = components["schemas"]["DigestEntryOut"];
export type InterestOut = components["schemas"]["InterestOut"];
export type InterestIn = components["schemas"]["InterestIn"];
export type SettingsOut = components["schemas"]["SettingsOut"];
export type StatusOut = components["schemas"]["StatusResponse"];
export type RunOut = components["schemas"]["RunOut"];
export type SettingsIn = components["schemas"]["SettingsIn"];
export type EntryAction = "save" | "dismiss" | "undo";

/** GET /digest: the day's ranked papers (the latest digest if none exists for `date`). */
export async function fetchDigest(
  client: ApiClient,
  date?: string,
): Promise<DigestOut> {
  const { data, error, response } = await client.GET("/digest", {
    params: { query: { date } },
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "digest"),
  );
}

/** POST /digest/{id}/save|dismiss|undo for the digest of `date`; returns the updated entry. */
export async function actOnEntry(
  client: ApiClient,
  action: EntryAction,
  date: string,
  arxivId: string,
): Promise<DigestEntryOut> {
  const path = {
    save: "/digest/{arxiv_id}/save",
    dismiss: "/digest/{arxiv_id}/dismiss",
    undo: "/digest/{arxiv_id}/undo",
  } as const;
  const { data, error, response } = await client.POST(path[action], {
    params: { path: { arxiv_id: arxivId }, query: { date } },
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "digest entry"),
  );
}

/** GET /interests. */
export async function fetchInterests(
  client: ApiClient,
): Promise<InterestOut[]> {
  const { data, error, response } = await client.GET("/interests");
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "interests"),
  );
}

/** POST /interests (a duplicate name is a 409 with a readable message). */
export async function createInterest(
  client: ApiClient,
  body: InterestIn,
): Promise<InterestOut> {
  const { data, error, response } = await client.POST("/interests", { body });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "interest"),
  );
}

/** PUT /interests/{id}: replaces the editable fields. */
export async function updateInterest(
  client: ApiClient,
  id: number,
  body: InterestIn,
): Promise<InterestOut> {
  const { data, error, response } = await client.PUT(
    "/interests/{interest_id}",
    { params: { path: { interest_id: id } }, body },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "interest"),
  );
}

/** DELETE /interests/{id}. */
export async function deleteInterest(
  client: ApiClient,
  id: number,
): Promise<void> {
  const { error, response } = await client.DELETE("/interests/{interest_id}", {
    params: { path: { interest_id: id } },
  });
  if (response.ok) return;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "interest"),
  );
}

/** GET /settings: digest size, categories, full-text top N, budget cap, month-to-date spend. */
export async function fetchSettings(client: ApiClient): Promise<SettingsOut> {
  const { data, error, response } = await client.GET("/settings");
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "settings"),
  );
}

/** GET /status: stores, last pipeline runs, paper counts, queue, Qdrant points, budget. */
export async function fetchStatus(
  client: ApiClient,
  signal?: AbortSignal,
): Promise<StatusOut> {
  const { data, error, response } = await client.GET("/status", { signal });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "status"),
  );
}

/** PUT /settings. */
export async function saveSettings(
  client: ApiClient,
  body: SettingsIn,
): Promise<SettingsOut> {
  const { data, error, response } = await client.PUT("/settings", { body });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "settings"),
  );
}

// --- paper comparison and topic timeline -----------------------------------------------------
// fetchPapers is free (Postgres only). postCompare and postTimeline are PAID: each makes one
// embeddings request and one chat request on the server. They are only ever called from a
// button, never from a page load, a keystroke or a refetch.

export type PaperListItemOut = components["schemas"]["PaperListItemOut"];
export type CompareRequest = components["schemas"]["CompareRequest"];
export type CompareOut = components["schemas"]["CompareOut"];
export type TimelineBody = components["schemas"]["TimelineBody"];
export type TimelineOut = components["schemas"]["TimelineOut"];
export type TimelinePoint = components["schemas"]["TimelinePoint"];

/** How many papers one picker lookup shows. */
export const PICKER_LIMIT = 20;

/** GET /papers: newest first; `q` is title words or an exact arXiv id. */
export async function fetchPapers(
  client: ApiClient,
  params: { q: string; hasFulltext?: boolean; limit?: number },
  signal?: AbortSignal,
): Promise<PaperListItemOut[]> {
  const { data, error, response } = await client.GET("/papers", {
    params: {
      query: {
        q: params.q.trim(),
        limit: params.limit ?? PICKER_LIMIT,
        has_fulltext: params.hasFulltext,
      },
    },
    signal,
  });
  if (data) return data.papers;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper list"),
  );
}

/** POST /compare (paid). */
export async function postCompare(
  client: ApiClient,
  body: CompareRequest,
  signal?: AbortSignal,
): Promise<CompareOut> {
  const { data, error, response } = await client.POST("/compare", {
    body,
    signal,
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "comparison"),
  );
}

/** POST /timeline (paid). */
export async function postTimeline(
  client: ApiClient,
  body: TimelineBody,
  signal?: AbortSignal,
): Promise<TimelineOut> {
  const { data, error, response } = await client.POST("/timeline", {
    body,
    signal,
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "timeline"),
  );
}

// --- the library: collections, saved papers, the .bib file, the review and PDF upload ---------
// Everything here is free (Postgres only) EXCEPT postReview (chat calls) and uploadPdf
// (embeddings). Those two are only ever called from a button, never from a page load, a
// keystroke or a refetch.

export type CollectionOut = components["schemas"]["CollectionOut"];
export type CollectionCreate = components["schemas"]["CollectionCreate"];
export type CollectionUpdate = components["schemas"]["CollectionUpdate"];
export type ItemOut = components["schemas"]["ItemOut"];
export type ItemCreate = components["schemas"]["ItemCreate"];
export type ItemPatch = components["schemas"]["ItemPatch"];
export type ReviewRequest = components["schemas"]["ReviewRequest"];
export type ReviewOut = components["schemas"]["ReviewOut"];
export type UploadOut = components["schemas"]["UploadOut"];
export type UploadDeleted = components["schemas"]["UploadDeleted"];

/** GET /library/collections: every collection with its paper count. */
export async function fetchCollections(
  client: ApiClient,
): Promise<CollectionOut[]> {
  const { data, error, response } = await client.GET("/library/collections");
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "collections"),
  );
}

/** POST /library/collections (a duplicate name is a 409 with a readable message). */
export async function createCollection(
  client: ApiClient,
  body: CollectionCreate,
): Promise<CollectionOut> {
  const { data, error, response } = await client.POST("/library/collections", {
    body,
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "collection"),
  );
}

/** PUT /library/collections/{name}: rename (the items move with it) and/or describe. */
export async function updateCollection(
  client: ApiClient,
  name: string,
  body: CollectionUpdate,
): Promise<CollectionOut> {
  const { data, error, response } = await client.PUT(
    "/library/collections/{name}",
    { params: { path: { name } }, body },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "collection"),
  );
}

/** DELETE /library/collections/{name}: the collection and its items, never the papers. */
export async function deleteCollection(
  client: ApiClient,
  name: string,
): Promise<void> {
  const { error, response } = await client.DELETE(
    "/library/collections/{name}",
    { params: { path: { name } } },
  );
  if (response.ok) return;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "collection"),
  );
}

/** GET /library/items: the collection's papers. */
export async function fetchItems(
  client: ApiClient,
  collection: string,
): Promise<ItemOut[]> {
  const { data, error, response } = await client.GET("/library/items", {
    params: { query: { collection } },
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "collection"),
  );
}

/**
 * POST /library/items: add a paper that is already in the database (arXiv id or link).
 * An unknown paper is a 404 whose message says it has to be ingested first.
 */
export async function addItem(
  client: ApiClient,
  body: ItemCreate,
): Promise<ItemOut> {
  const { data, error, response } = await client.POST("/library/items", {
    body,
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper"),
  );
}

/** PATCH /library/items/{id}?collection=: status, notes, or a move to another collection. */
export async function updateItem(
  client: ApiClient,
  arxivId: string,
  collection: string,
  body: ItemPatch,
): Promise<ItemOut> {
  const { data, error, response } = await client.PATCH(
    "/library/items/{arxiv_id}",
    { params: { path: { arxiv_id: arxivId }, query: { collection } }, body },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper"),
  );
}

/** DELETE /library/items/{id}?collection=: the paper stays in the database. */
export async function removeItem(
  client: ApiClient,
  arxivId: string,
  collection: string,
): Promise<void> {
  const { error, response } = await client.DELETE("/library/items/{arxiv_id}", {
    params: { path: { arxiv_id: arxivId }, query: { collection } },
  });
  if (response.ok) return;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "paper"),
  );
}

/** GET /library/bibtex: the .bib text and the file name the server suggests. */
export async function fetchBibtex(
  client: ApiClient,
  collection: string,
): Promise<{ text: string; filename: string | null }> {
  const { data, error, response } = await client.GET("/library/bibtex", {
    params: { query: { collection } },
    parseAs: "text",
  });
  if (typeof data === "string")
    return {
      text: data,
      filename:
        /filename="([^"\\/]+)"/.exec(
          response.headers.get("content-disposition") ?? "",
        )?.[1] ?? null,
    };
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "collection"),
  );
}

/** POST /library/review (PAID: one chat call per group of eight papers above eight, plus one). */
export async function postReview(
  client: ApiClient,
  body: ReviewRequest,
  signal?: AbortSignal,
): Promise<ReviewOut> {
  const { data, error, response } = await client.POST("/library/review", {
    body,
    signal,
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "collection"),
  );
}

export interface UploadParams {
  title: string;
  year?: number | null;
  collection?: string | null;
  abstract?: string | null;
}

/**
 * POST /library/upload (PAID: the embeddings of the PDF's chunks). The body is the raw PDF,
 * not a form; this is a plain fetch because the OpenAPI schema does not describe a raw body.
 */
export async function uploadPdf(
  fetchFn: typeof fetch,
  baseUrl: string,
  params: UploadParams,
  file: Blob,
  signal?: AbortSignal,
): Promise<UploadOut> {
  const query = new URLSearchParams({ title: params.title });
  if (params.year != null) query.set("year", String(params.year));
  if (params.collection) query.set("collection", params.collection);
  if (params.abstract) query.set("abstract", params.abstract);
  const response = await fetchFn(`${baseUrl}/library/upload?${query}`, {
    method: "POST",
    headers: { "Content-Type": "application/pdf" },
    body: file,
    signal,
  });
  if (response.ok) return (await response.json()) as UploadOut;
  const body: unknown = await response.json().catch(() => undefined);
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(body), "upload"),
  );
}

/** DELETE /library/uploads/{id}: an uploaded document and everything indexed from it. */
export async function deleteUpload(
  client: ApiClient,
  arxivId: string,
): Promise<UploadDeleted> {
  const { data, error, response } = await client.DELETE(
    "/library/uploads/{arxiv_id}",
    { params: { path: { arxiv_id: arxivId } } },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "upload"),
  );
}

// --- evaluation runs (the /evals page) ---------------------------------------------------------
// Read-only and free: the API reads Postgres only (no Qdrant, no model call, nothing written).

export type EvalRunSummaryOut = components["schemas"]["EvalRunSummaryOut"];
export type EvalRunOut = components["schemas"]["EvalRunOut"];
export type EvalResultsOut = components["schemas"]["EvalResultsOut"];
export type EvalResultRowOut = components["schemas"]["EvalResultRowOut"];
export type EvalResultDetailOut = components["schemas"]["EvalResultDetailOut"];
export type RetrievedChunkOut = components["schemas"]["RetrievedChunkOut"];

/** GET /evals/runs: every run, newest first, with the headline numbers of its "all" split. */
export async function fetchEvalRuns(
  client: ApiClient,
): Promise<EvalRunSummaryOut[]> {
  const { data, error, response } = await client.GET("/evals/runs");
  if (data) return data.runs;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "evaluation run list"),
  );
}

/** GET /evals/runs/{id}: one run with its configuration and every stored metric (splits too). */
export async function fetchEvalRun(
  client: ApiClient,
  runId: number,
): Promise<EvalRunOut> {
  const { data, error, response } = await client.GET("/evals/runs/{run_id}", {
    params: { path: { run_id: runId } },
  });
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "evaluation run"),
  );
}

/** GET /evals/runs/{id}/results: one row of scores per question (none for a selection). */
export async function fetchEvalResults(
  client: ApiClient,
  runId: number,
): Promise<EvalResultsOut> {
  const { data, error, response } = await client.GET(
    "/evals/runs/{run_id}/results",
    { params: { path: { run_id: runId } } },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "evaluation run"),
  );
}

/** GET /evals/runs/{id}/results/{item}: one question with its chunks, answer and verdicts. */
export async function fetchEvalResult(
  client: ApiClient,
  runId: number,
  itemId: number,
): Promise<EvalResultDetailOut> {
  const { data, error, response } = await client.GET(
    "/evals/runs/{run_id}/results/{item_id}",
    { params: { path: { run_id: runId, item_id: itemId } } },
  );
  if (data) return data;
  throw new ApiError(
    response.status,
    apiErrorMessage(response.status, detailOf(error), "question result"),
  );
}

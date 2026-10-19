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

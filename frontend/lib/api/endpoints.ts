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

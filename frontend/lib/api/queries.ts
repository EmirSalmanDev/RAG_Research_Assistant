// The endpoint functions bound to the real client (same-origin /api, proxied to FastAPI).
// Components call these from TanStack Query; tests call lib/api/endpoints.ts directly.

import { api } from "./client";
import * as endpoints from "./endpoints";
import type { SearchParams } from "../search/params";

export type {
  ChunkOut,
  PaperChunksOut,
  PaperDetailsOut,
  PaperOut,
  RelatedOut,
  RelatedPaperOut,
  SearchHit,
  SearchResponse,
} from "./endpoints";

export const fetchChunk = (chunkId: string) =>
  endpoints.fetchChunk(api, chunkId);
export const fetchPaper = (arxivId: string) =>
  endpoints.fetchPaper(api, arxivId);
export const fetchPaperChunks = (arxivId: string) =>
  endpoints.fetchPaperChunks(api, arxivId);
export const fetchPaperDetails = (arxivId: string) =>
  endpoints.fetchPaperDetails(api, arxivId);
export const fetchPaperRelated = (arxivId: string) =>
  endpoints.fetchPaperRelated(api, arxivId);
export const fetchSearch = (params: SearchParams, signal?: AbortSignal) =>
  endpoints.fetchSearch(api, params, signal);

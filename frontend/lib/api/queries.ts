// The endpoint functions bound to the real client (same-origin /api, proxied to FastAPI).
// Components call these from TanStack Query; tests call lib/api/endpoints.ts directly.

import { api } from "./client";
import * as endpoints from "./endpoints";
import type { SearchParams } from "../search/params";

export type {
  ChunkOut,
  CompareOut,
  CompareRequest,
  DigestEntryOut,
  DigestOut,
  EntryAction,
  InterestIn,
  InterestOut,
  SettingsIn,
  SettingsOut,
  PaperChunksOut,
  PaperDetailsOut,
  PaperListItemOut,
  PaperOut,
  RelatedOut,
  RelatedPaperOut,
  SearchHit,
  SearchResponse,
  TimelineBody,
  TimelineOut,
  TimelinePoint,
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
export const fetchDigest = (date?: string) => endpoints.fetchDigest(api, date);
export const actOnEntry = (
  action: endpoints.EntryAction,
  date: string,
  arxivId: string,
) => endpoints.actOnEntry(api, action, date, arxivId);
export const fetchInterests = () => endpoints.fetchInterests(api);
export const createInterest = (body: endpoints.InterestIn) =>
  endpoints.createInterest(api, body);
export const updateInterest = (id: number, body: endpoints.InterestIn) =>
  endpoints.updateInterest(api, id, body);
export const deleteInterest = (id: number) => endpoints.deleteInterest(api, id);
export const fetchSettings = () => endpoints.fetchSettings(api);
export const saveSettings = (body: endpoints.SettingsIn) =>
  endpoints.saveSettings(api, body);
export const fetchPapers = (
  params: { q: string; hasFulltext?: boolean; limit?: number },
  signal?: AbortSignal,
) => endpoints.fetchPapers(api, params, signal);
// The two paid calls: only ever started by a button.
export const postCompare = (
  body: endpoints.CompareRequest,
  signal?: AbortSignal,
) => endpoints.postCompare(api, body, signal);
export const postTimeline = (
  body: endpoints.TimelineBody,
  signal?: AbortSignal,
) => endpoints.postTimeline(api, body, signal);

// The endpoint functions bound to the real client (same-origin /api, proxied to FastAPI).
// Components call these from TanStack Query; tests call lib/api/endpoints.ts directly.

import { api } from "./client";
import * as endpoints from "./endpoints";
import type { SearchParams } from "../search/params";

export type {
  ChunkOut,
  CollectionCreate,
  CollectionOut,
  CollectionUpdate,
  CompareOut,
  CompareRequest,
  DigestEntryOut,
  DigestOut,
  EntryAction,
  InterestIn,
  InterestOut,
  ItemCreate,
  ItemOut,
  ItemPatch,
  ReviewOut,
  ReviewRequest,
  UploadDeleted,
  UploadOut,
  UploadParams,
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

// The library. Free (Postgres only), except the review and the upload.
export const fetchCollections = () => endpoints.fetchCollections(api);
export const createCollection = (body: endpoints.CollectionCreate) =>
  endpoints.createCollection(api, body);
export const updateCollection = (
  name: string,
  body: endpoints.CollectionUpdate,
) => endpoints.updateCollection(api, name, body);
export const deleteCollection = (name: string) =>
  endpoints.deleteCollection(api, name);
export const fetchItems = (collection: string) =>
  endpoints.fetchItems(api, collection);
export const addItem = (body: endpoints.ItemCreate) =>
  endpoints.addItem(api, body);
export const updateItem = (
  arxivId: string,
  collection: string,
  body: endpoints.ItemPatch,
) => endpoints.updateItem(api, arxivId, collection, body);
export const removeItem = (arxivId: string, collection: string) =>
  endpoints.removeItem(api, arxivId, collection);
export const fetchBibtex = (collection: string) =>
  endpoints.fetchBibtex(api, collection);
export const deleteUpload = (arxivId: string) =>
  endpoints.deleteUpload(api, arxivId);
// The two paid calls of the library: only ever started by a button.
export const postReview = (
  body: endpoints.ReviewRequest,
  signal?: AbortSignal,
) => endpoints.postReview(api, body, signal);
export const uploadPdf = (
  params: endpoints.UploadParams,
  file: Blob,
  signal?: AbortSignal,
) =>
  endpoints.uploadPdf(
    (input, init) => fetch(input, init),
    "/api",
    params,
    file,
    signal,
  );

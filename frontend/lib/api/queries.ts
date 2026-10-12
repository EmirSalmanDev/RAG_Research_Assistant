import { api } from "./client";

/** The text of one chunk (what a citation chip opens). Throws a message fit for the UI. */
export async function fetchChunk(chunkId: string) {
  const { data, response } = await api.GET("/chunks/{chunk_id}", {
    params: { path: { chunk_id: chunkId } },
  });
  if (data) return data;
  throw new Error(
    response.status === 404
      ? "This passage is no longer in the index."
      : "The passage could not be loaded. Is the API running?",
  );
}

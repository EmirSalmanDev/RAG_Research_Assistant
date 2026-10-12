// Turning a failed API response into a sentence for the reader. Plain TypeScript.

/** The API answered, but not with the data (404, 503, 422, ...). */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * `detail` is what FastAPI puts in an error body: a string for HTTPException (written by
 * our routes, e.g. "OPENAI_API_KEY is not set (put it in .env)"), a list for a 422.
 * `what` names the thing asked for ("paper", "passage", "search").
 */
export function apiErrorMessage(
  status: number,
  detail: unknown,
  what: string,
): string {
  if (typeof detail === "string" && detail.trim() !== "") return detail;
  if (status === 404) return `This ${what} was not found.`;
  if (status === 422)
    return "The request was not accepted. Check the question and the filters.";
  if (status >= 500)
    return `The API had a problem (HTTP ${status}). Check its log.`;
  return `The request failed (HTTP ${status}).`;
}

/** The `detail` of an error body, if there is one. */
export function detailOf(error: unknown): unknown {
  return typeof error === "object" && error !== null && "detail" in error
    ? (error as { detail: unknown }).detail
    : undefined;
}

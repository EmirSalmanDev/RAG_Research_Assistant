// The /evals fetchers against a MOCKED fetch: the paths they ask for and how an error
// becomes an ApiError. Nothing here touches a network or a real API.
import { strict as assert } from "node:assert";
import { test } from "node:test";

import createClient from "openapi-fetch";

import {
  fetchEvalResult,
  fetchEvalResults,
  fetchEvalRun,
  fetchEvalRuns,
} from "../api/endpoints.ts";
import { ApiError } from "../api/errors.ts";
import type { paths } from "../api/schema";

function mocked(respond: (url: URL) => Response) {
  const calls: { method: string; url: URL }[] = [];
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

test("the four reads are GETs on the /evals paths", async () => {
  const { client, calls } = mocked((url) => {
    if (url.pathname === "/api/evals/runs") return Response.json({ runs: [] });
    if (url.pathname.endsWith("/results"))
      return Response.json({ run_id: 3, kind: "answers", results: [] });
    return Response.json({ id: 3 });
  });
  assert.deepEqual(await fetchEvalRuns(client), []);
  await fetchEvalRun(client, 3);
  assert.equal((await fetchEvalResults(client, 3)).run_id, 3);
  await fetchEvalResult(client, 3, 41);
  assert.deepEqual(
    calls.map((c) => `${c.method} ${c.url.pathname}`),
    [
      "GET /api/evals/runs",
      "GET /api/evals/runs/3",
      "GET /api/evals/runs/3/results",
      "GET /api/evals/runs/3/results/41",
    ],
  );
});

test("a 404 becomes an ApiError with the API's own message", async () => {
  const { client } = mocked(() =>
    Response.json({ detail: "unknown evaluation run" }, { status: 404 }),
  );
  await assert.rejects(fetchEvalRun(client, 999), (err: unknown) => {
    assert.ok(err instanceof ApiError);
    assert.equal(err.status, 404);
    assert.equal(err.message, "unknown evaluation run");
    return true;
  });
  const { client: down } = mocked(() => new Response("", { status: 503 }));
  await assert.rejects(fetchEvalRuns(down), /HTTP 503/);
});

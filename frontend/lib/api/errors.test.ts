import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  ApiError,
  apiErrorMessage,
  describeError,
  detailOf,
} from "./errors.ts";

test("a message written by the API is shown as is", () => {
  assert.equal(
    apiErrorMessage(
      503,
      "OPENAI_API_KEY is not set (put it in .env)",
      "search",
    ),
    "OPENAI_API_KEY is not set (put it in .env)",
  );
  assert.equal(apiErrorMessage(404, "unknown paper", "paper"), "unknown paper");
});

test("without a message, the status decides", () => {
  assert.equal(
    apiErrorMessage(404, undefined, "paper"),
    "This paper was not found.",
  );
  assert.match(apiErrorMessage(422, [{ msg: "x" }], "search"), /not accepted/);
  assert.match(apiErrorMessage(500, undefined, "search"), /HTTP 500/);
  assert.match(apiErrorMessage(418, "  ", "search"), /HTTP 418/);
});

test("detailOf reads FastAPI error bodies and nothing else", () => {
  assert.equal(detailOf({ detail: "x" }), "x");
  assert.equal(detailOf({}), undefined);
  assert.equal(detailOf(null), undefined);
  assert.equal(detailOf("text"), undefined);
});

test("ApiError carries the status", () => {
  const err = new ApiError(404, "gone");
  assert.equal(err.status, 404);
  assert.equal(err.name, "ApiError");
});

test("describeError explains a network failure and passes other messages through", () => {
  assert.match(
    describeError(new TypeError("Failed to fetch")),
    /Could not reach the API/,
  );
  assert.match(
    describeError(new TypeError("NetworkError when attempting")),
    /make up/,
  );
  assert.match(
    describeError(new TypeError("Load failed")),
    /Could not reach the API/,
  );
  // a TypeError that is a bug in our code must not be disguised as "API down"
  assert.equal(
    describeError(new TypeError("x is not a function")),
    "x is not a function",
  );
  assert.equal(describeError(new ApiError(404, "gone")), "gone");
  assert.equal(describeError(new Error("boom")), "boom");
  assert.equal(describeError(new Error("  ")), "Something went wrong.");
  assert.equal(describeError("text"), "Something went wrong.");
  assert.equal(describeError(undefined), "Something went wrong.");
});

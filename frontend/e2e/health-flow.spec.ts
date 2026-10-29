// The /health page, the budget banner, and the loading and error states of the views.
//
// WHAT THIS PROVES, AND WHAT IT DOES NOT. The API behind /api is the mock in
// e2e/mock-api.ts (invented numbers), reached through the real Next.js /api rewrite; each
// test replaces single answers with page.route (503 with a message, a dropped connection,
// a slow answer, a status with a store down). So this proves what the page shows for each
// of those answers. It does not prove that the real API computes them (backend/tests/
// unit/test_status_api.py and integration/test_status_db.py do) or that Docker's
// healthchecks see a healthy container.
//
// CONTAINMENT. Same as the other browser tests: every non-local request is aborted and
// recorded, and the test fails if one was attempted.
import { expect, test } from "@playwright/test";
import type { BrowserContext } from "@playwright/test";

import { isAllowedUrl } from "./containment.ts";

async function contain(context: BrowserContext, blocked: string[]) {
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (isAllowedUrl(url)) return route.continue();
    blocked.push(url);
    return route.abort("blockedbyclient");
  });
}

const STATUS = "**/api/status";
const DIGEST = "**/api/digest**";

function run(over: Record<string, unknown> = {}) {
  return {
    source: "embed-paper",
    started_at: new Date(Date.now() - 3 * 3600_000).toISOString(),
    finished_at: new Date(Date.now() - 3 * 3600_000 + 60_000).toISOString(),
    ok: true,
    error: null,
    skipped: null,
    abandoned: false,
    retrying: false,
    note: null,
    ...over,
  };
}

function status(
  over: {
    stores?: Record<string, string>;
    budget?: Record<string, unknown>;
    runs?: unknown[];
    byStatus?: Record<string, number>;
  } = {},
) {
  return {
    stores: { postgres: "up", qdrant: "up", redis: "up", ...over.stores },
    database: {
      papers_total: 12,
      papers_by_fulltext_status: over.byStatus ?? { none: 9, indexed: 3 },
      last_runs: over.runs ?? [run()],
      budget: {
        cap_usd: 5,
        spent_usd: 1.25,
        paused: false,
        month_start: "2026-10-01T00:00:00Z",
        ...over.budget,
      },
    },
    queue_length: 2,
    qdrant_points: 8912,
    errors: {},
  };
}

test("health: a healthy system, then a store down, a failed step and a paused budget", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  const summary = page.getByTestId("health-summary");

  await test.step("a healthy system", async () => {
    await page.route(STATUS, (route) => route.fulfill({ json: status() }));
    await page.goto("/health", { waitUntil: "domcontentloaded" });
    await expect(
      page
        .getByRole("navigation", { name: "Main" })
        .getByRole("link", { name: "Health" }),
    ).toHaveAttribute("aria-current", "page");
    await expect(summary).toHaveAttribute("data-level", "ok");
    await expect(summary).toContainText("Everything is running.");
    await expect(page.getByTestId("health-spend")).toHaveText(
      "Spent $1.2500 of $5.0000",
    );
    await expect(page.getByTestId("health-budget-state")).toContainText(
      "running",
    );
    await expect(page.getByTestId("health-points")).toContainText("8,912");
    await expect(page.getByTestId("health-queue")).toContainText("2");
    await expect(page.getByText("Full text indexed")).toBeVisible();
    const row = page.getByRole("row", { name: /embed-paper/ });
    await expect(row).toContainText("3 h ago");
    await expect(row).toContainText("ok");
    await expect(page.getByTestId("budget-banner")).toHaveCount(0);
  });

  await test.step("a store down is named, the page stays readable", async () => {
    await page.unroute(STATUS);
    await page.route(STATUS, (route) =>
      route.fulfill({ json: status({ stores: { qdrant: "down" } }) }),
    );
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(summary).toHaveAttribute("data-level", "down");
    await expect(summary).toContainText("Qdrant is not reachable.");
    await expect(page.getByTestId("health-spend")).toBeVisible();
  });

  await test.step("a failed step and a skipped step are told apart", async () => {
    await page.unroute(STATUS);
    await page.route(STATUS, (route) =>
      route.fulfill({
        json: status({
          runs: [
            run({
              source: "index-abstracts",
              ok: false,
              error: "RuntimeError: Invented failure",
            }),
            run({ source: "embed-paper", skipped: "budget cap reached" }),
            run({
              source: "arxiv-daily",
              note: "window capped: papers updated 2026-09-28 to 2026-10-19 skipped (make backfill)",
            }),
          ],
        }),
      }),
    );
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(summary).toHaveAttribute("data-level", "warning");
    await expect(summary).toContainText(
      "A step failed in its last run: index-abstracts.",
    );
    // a capped arXiv window is a note on its row, not a problem of the system
    await expect(page.getByRole("row", { name: /arxiv-daily/ })).toContainText(
      "window capped: papers updated 2026-09-28 to 2026-10-19 skipped",
    );
    await expect(
      page.getByRole("row", { name: /index-abstracts/ }),
    ).toContainText("RuntimeError: Invented failure");
    await expect(page.getByRole("row", { name: /embed-paper/ })).toContainText(
      "skipped: budget cap reached",
    );
  });

  await test.step("a paused budget shows a banner on every page", async () => {
    await page.unroute(STATUS);
    await page.route(STATUS, (route) =>
      route.fulfill({
        json: status({ budget: { paused: true, spent_usd: 5.2 } }),
      }),
    );
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("health-budget-state")).toContainText(
      "paused",
    );
    const banner = page.getByTestId("budget-banner");
    await expect(banner).toContainText("$5.2000 of $5.0000");
    await expect(banner).toContainText("chat and search still work");
    await page.goto("/settings", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("budget-banner")).toBeVisible();
    await page
      .getByTestId("budget-banner")
      .getByRole("link", { name: "Change the cap" })
      .click();
    await expect(page).toHaveURL(/\/settings$/);
  });

  expect(pageErrors, "no uncaught page error").toEqual([]);
  expect(blocked, "no request left the machine").toEqual([]);
});

test("health: an unreadable status is an error with a retry, not an empty page", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  await page.route(STATUS, (route) =>
    route.fulfill({
      status: 503,
      json: { detail: "Invented: the API is starting." },
    }),
  );
  await page.goto("/health", { waitUntil: "domcontentloaded" });
  const alert = page.getByTestId("error-state");
  await expect(alert).toContainText(
    "The system status could not be loaded: Invented: the API is starting.",
  );
  await page.unroute(STATUS);
  await page.route(STATUS, (route) => route.fulfill({ json: status() }));
  await alert.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByTestId("health-summary")).toHaveAttribute(
    "data-level",
    "ok",
  );
  expect(blocked, "no request left the machine").toEqual([]);
});

test("states: loading while the API is slow, the API's message on an error, retry, an unreachable API, an unknown page", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));

  await test.step("a slow answer shows the loading state, then the failure", async () => {
    await page.route(DIGEST, async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 700));
      await route.fulfill({
        status: 503,
        json: { detail: "Invented: the database is not reachable." },
      });
    });
    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(
      page.getByRole("status").filter({ hasText: "Loading the digest" }),
    ).toBeVisible();
    await expect(page.getByTestId("error-state")).toContainText(
      "The digest could not be loaded: Invented: the database is not reachable.",
    );
  });

  await test.step("Try again asks again and shows the page", async () => {
    await page.unroute(DIGEST);
    await page
      .getByRole("alert")
      .getByRole("button", { name: "Try again" })
      .click();
    await expect(
      page.getByRole("heading", { level: 1, name: "Mon, Oct 19, 2026" }),
    ).toBeVisible();
    await expect(page.getByTestId("error-state")).toHaveCount(0);
  });

  await test.step("an API that cannot be reached is explained", async () => {
    await page.route(DIGEST, (route) => route.abort("failed"));
    await page.goto("/?date=2026-10-18", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("error-state")).toContainText(
      "Could not reach the API",
    );
    await page.unroute(DIGEST);
  });

  await test.step("an address with no page", async () => {
    await page.goto("/no-such-page", { waitUntil: "domcontentloaded" });
    await expect(
      page.getByRole("heading", { name: "Page not found" }),
    ).toBeVisible();
    await page.getByRole("link", { name: "Back to the digest" }).click();
    await expect(page).toHaveURL(/\/$/);
  });

  expect(pageErrors, "no uncaught page error").toEqual([]);
  expect(blocked, "no request left the machine").toEqual([]);
});

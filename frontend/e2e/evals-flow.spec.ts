// The /evals page: metric trends, a stopped run, two runs compared, one question opened.
//
// WHAT THIS PROVES, AND WHAT IT DOES NOT. The API behind /api is the mock in
// e2e/mock-api.ts (invented runs, questions and texts), reached through the real Next.js
// /api rewrite. So this proves the page: the chart's series and its metric switch, that a
// stopped run is marked and never drawn as a complete one, the comparison with B - A, the
// split filter, and the question panel with its chunks, answer, verdicts and links. It
// does not prove that the real API returns these shapes (backend/tests/unit/
// test_evals_api.py does, and the shapes here follow the generated schema).
//
// CONTAINMENT. Same as the other browser tests: every non-local request is aborted and
// recorded, and links are checked by their href only, never opened.
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

test("evals: trends, a stopped run, two runs compared, one question opened", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  // The page only reads: anything but a GET to the API would be a bug.
  const writes: string[] = [];
  page.on("request", (request) => {
    if (request.method() !== "GET" && request.url().includes("/api/"))
      writes.push(`${request.method()} ${request.url()}`);
  });

  await page.goto("/evals", { waitUntil: "domcontentloaded" });
  const chart = page.getByTestId("trend-chart");
  const runRow = (id: number) =>
    page.locator(`[data-testid="run-row"][data-run="${id}"]`);

  await test.step("the site nav links the page", async () => {
    await expect(
      page
        .getByRole("navigation", { name: "Main" })
        .getByRole("link", { name: "Evals" }),
    ).toHaveAttribute("aria-current", "page");
  });

  await test.step("the trend chart draws two retrieval series", async () => {
    await expect(chart).toBeVisible();
    await expect(chart).toHaveAttribute("data-metric", "recall@8");
    await expect(chart.getByTestId("trend-series")).toHaveCount(2);
    await expect(chart.getByTestId("trend-line")).toHaveCount(2);
    await expect(chart.getByTestId("trend-point")).toHaveCount(4);
    const legend = page.getByRole("list", { name: "Series" });
    await expect(legend.getByRole("button")).toHaveCount(2);
    // a legend button switches its line off and on again
    const dense = legend.getByRole("button", { name: /dense/ });
    await expect(dense).toHaveAttribute("aria-pressed", "true");
    await dense.click();
    await expect(chart.getByTestId("trend-series")).toHaveCount(1);
    await dense.click();
    await expect(chart.getByTestId("trend-series")).toHaveCount(2);
  });

  await test.step("the stopped run is marked as stopped, never as complete", async () => {
    const status = runRow(8).getByTestId("run-status");
    await expect(status).toHaveAttribute("data-state", "stopped");
    await expect(status).toContainText("stopped");
    await expect(status).toContainText("spend cap");
    await expect(status).not.toContainText("complete");
    await expect(runRow(7).getByTestId("run-status")).toHaveText("complete");
    await expect(runRow(9)).toContainText("no per-question results"); // the selection
  });

  await test.step("switching the metric draws the answer runs; the stopped run is a hollow point off the line", async () => {
    await page
      .getByLabel("Metric", { exact: true })
      .selectOption("faithfulness");
    await expect(chart).toHaveAttribute("data-metric", "faithfulness");
    await expect(chart.getByTestId("trend-series")).toHaveCount(3);
    const stopped = chart.locator('[data-testid="trend-point"][data-run="8"]');
    await expect(stopped).toHaveAttribute("data-state", "stopped");
    expect(await stopped.locator("title").textContent()).toContain(
      "not a complete run",
    );
    // its series has the point but no line: a stopped run is never joined to a trend
    const hybrid = chart.locator(
      '[data-testid="trend-series"][data-series="answers|ablation|a3-hybrid"]',
    );
    await expect(hybrid.getByTestId("trend-point")).toHaveCount(1);
    await expect(hybrid.getByTestId("trend-line")).toHaveCount(0);
    // the runs table is the chart's table view: it shows the chosen metric
    await expect(runRow(7)).toContainText("0.800");
  });

  await test.step("pick two runs: the headline numbers side by side with B − A", async () => {
    await page.getByLabel("Run A", { exact: true }).selectOption("7");
    await page.getByLabel("Run B", { exact: true }).selectOption("8");
    const headline = page.getByTestId("headline-table");
    const faithfulness = headline.locator('tr[data-metric="faithfulness"]');
    await expect(faithfulness).toContainText("0.800");
    await expect(faithfulness).toContainText("0.700");
    await expect(faithfulness.getByTestId("diff")).toHaveText(/−0\.100/);
    await expect(faithfulness.getByTestId("diff")).toHaveAttribute(
      "data-direction",
      "worse",
    );
    const recall = headline.locator('tr[data-metric="context_recall"]');
    await expect(recall.getByTestId("diff")).toHaveText(/\+0\.250/);
    await expect(recall.getByTestId("diff")).toHaveAttribute(
      "data-direction",
      "better",
    );
    // lower latency is better
    const latency = headline.locator('tr[data-metric="latency_ms_p50"]');
    await expect(latency.getByTestId("diff")).toHaveAttribute(
      "data-direction",
      "worse",
    );
  });

  const table = page.getByTestId("questions-table");
  const rows = table.getByTestId("question-row");

  await test.step("the questions joined on their id, with the split filter", async () => {
    await expect(rows).toHaveCount(3);
    await expect(page.getByTestId("join-counts")).toContainText(
      "3 questions: 2 in both runs, 1 only in A, 0 only in B.",
    );
    // run 8 stopped before its third question
    await expect(table.locator('[data-question="gen-002"]')).toContainText(
      "not in this run",
    );
    const q01 = table.locator('[data-question="q01"]');
    await expect(q01.getByTestId("diff")).toHaveText(/\+0\.250/); // 0.5 -> 0.75
    await expect(page.getByTestId("heldout-note")).toContainText(
      "never used to choose settings",
    );
    await page.getByRole("button", { name: "Held-out (golden)" }).click();
    await expect(rows).toHaveCount(1);
    await expect(rows.first()).toHaveAttribute("data-question", "q01");
    // the headline numbers follow the split (from GET /evals/runs/{id})
    await expect(
      page
        .getByTestId("headline-table")
        .locator('tr[data-metric="faithfulness"]'),
    ).toContainText("0.500");
    await page.getByRole("button", { name: "All questions" }).click();
    await expect(rows).toHaveCount(3);
  });

  await test.step("a question opens beside the tables: chunks, answer, verdicts, arXiv link", async () => {
    await table
      .getByRole("button", {
        name: "Does the invented adapter freeze the base model?",
      })
      .click();
    const panel = page.getByRole("dialog");
    await expect(panel).toBeVisible();
    // it opens on run A and switches to run B
    await expect(
      panel.getByRole("button", { name: "Run A #7" }),
    ).toHaveAttribute("aria-pressed", "true");
    await expect(panel.getByTestId("eval-answer")).toContainText(
      "Invented dense answer",
    );
    await panel.getByRole("button", { name: "Run B #8" }).click();
    await expect(panel.getByTestId("eval-answer")).toContainText(
      "Invented hybrid answer: the invented base model stays frozen [1].",
    );
    await expect(
      panel.getByText("Yes: the invented base model stays frozen."),
    ).toBeVisible(); // the reference answer

    const chunks = panel.getByTestId("retrieved-chunk");
    await expect(chunks).toHaveCount(2);
    const first = chunks.first();
    await expect(first.getByTestId("chunk-text")).toHaveText(
      "Invented hybrid chunk text: the adapter is trained while the base model stays frozen.",
    );
    await expect(first.getByTestId("reference-badge")).toBeVisible();
    await expect(chunks.nth(1).getByTestId("reference-badge")).toHaveCount(0);
    await expect(
      first.getByRole("link", { name: /arXiv 2612\.00001/ }),
    ).toHaveAttribute("href", "https://arxiv.org/abs/2612.00001");
    await expect(chunks.nth(1).getByRole("link")).toHaveCount(0); // an upload: no arXiv page

    const faithfulness = panel.locator(
      '[data-testid="judge-section"][data-metric="faithfulness"]',
    );
    await expect(faithfulness).toContainText(
      "The invented base model stays frozen.",
    );
    await expect(faithfulness.getByRole("row").nth(1)).toContainText("yes");
    await expect(faithfulness.getByRole("row").nth(2)).toContainText("no");
    await expect(
      panel.locator(
        '[data-testid="judge-section"][data-metric="citation_accuracy"]',
      ),
    ).toContainText("invalid_citations");
    await expect(panel.getByTestId("judge-errors")).toContainText(
      "an invented reason",
    );

    // Links are checked by href only: no PDF anywhere, and every external one is an
    // arXiv abstract page.
    const hrefs = await page
      .locator("a[href]")
      .evaluateAll((links) => links.map((a) => a.getAttribute("href") ?? ""));
    expect(hrefs.filter((h) => /\.pdf|\/pdf\//i.test(h))).toEqual([]);
    const external = hrefs.filter((h) => /^https?:/.test(h));
    expect(external.length).toBeGreaterThan(0);
    expect(external.every((h) => h.startsWith("https://arxiv.org/abs/"))).toBe(
      true,
    );

    await page.keyboard.press("Escape");
    await expect(panel).toBeHidden();
  });

  expect(writes).toEqual([]);
  expect(pageErrors).toEqual([]);
  expect(blocked).toEqual([]);
});

test("evals: no runs yet, an API error, and a selection without results", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  // page.route wins over the context's route: these answers replace the mock's.
  const RUNS = "**/api/evals/runs";

  await test.step("no runs stored yet", async () => {
    await page.route(RUNS, (route) => route.fulfill({ json: { runs: [] } }));
    await page.goto("/evals", { waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("no-runs")).toBeVisible();
    await expect(page.getByTestId("trend-chart")).toHaveCount(0);
    await page.unroute(RUNS);
  });

  await test.step("an API error is shown as the API's own message", async () => {
    await page.route(RUNS, (route) =>
      route.fulfill({
        status: 503,
        json: { detail: "Invented: the database is not reachable." },
      }),
    );
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(
      page.getByText("Invented: the database is not reachable."),
    ).toBeVisible();
    await page.unroute(RUNS);
  });

  await test.step("a selection picked as a run has no per-question results", async () => {
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(page.getByTestId("trend-chart")).toBeVisible();
    await page.getByLabel("Run A", { exact: true }).selectOption("9");
    await expect(page.getByTestId("no-results")).toContainText(
      "is a selection",
    );
    await expect(page.getByTestId("questions-table")).toHaveCount(0);
  });

  expect(pageErrors).toEqual([]);
  expect(blocked).toEqual([]);
});

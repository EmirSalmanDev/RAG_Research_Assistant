// Compare three papers and run a timeline.
//
// WHAT THIS PROVES, AND WHAT IT DOES NOT. The API behind /api is the mock in
// e2e/mock-api.ts (invented papers, no model), reached through the real Next.js /api
// rewrite. So this proves the pages: the picker and its limits, that nothing paid is sent
// until a button is pressed, the table with cited and "not stated" cells, that a citation
// chip opens the sources panel with the passage, the markdown export, and the timeline's
// chart and list. It does not prove that the real model cites correctly (the backend tests
// and a separate run against the real API do), and the passage text is invented.
//
// CONTAINMENT. Same as the other browser tests: every non-local request is aborted and
// recorded, and links are checked by their href only, never opened.
import { readFile } from "node:fs/promises";

import { expect, test } from "@playwright/test";

import { isAllowedUrl } from "./containment.ts";

test.beforeEach(async ({ context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
});

async function contain(
  context: import("@playwright/test").BrowserContext,
  blocked: string[],
) {
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (isAllowedUrl(url)) return route.continue();
    blocked.push(url);
    return route.abort("blockedbyclient");
  });
}

test("compare three papers: pick, compare, open a citation, export", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  const paid: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") paid.push(request.url());
  });

  await page.goto("/compare", { waitUntil: "domcontentloaded" });
  const compare = page.getByRole("button", { name: "Compare", exact: true });
  const picked = page.getByRole("list", { name: "Picked papers" });

  await test.step("the quick list shows the full-text papers; Compare is off", async () => {
    await expect(
      page.getByRole("button", { name: /Invented Explore Paper 1/ }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Invented Explore Paper 3/ }),
    ).toHaveCount(0); // abstract-only papers are not in the quick list
    await expect(compare).toBeDisabled();
    await expect(
      page.getByText("Pick at least 2 papers (0 picked)"),
    ).toBeVisible();
  });

  await test.step("typing a search finds papers and sends nothing paid", async () => {
    await page.getByLabel("Find a paper by title or arXiv id").fill("explore");
    const results = page.getByRole("group", { name: "Search results" });
    await expect(
      results.getByRole("button", { name: /Invented Explore Paper 6/ }),
    ).toBeVisible();
    await expect(results.getByRole("button")).toHaveCount(6);
    expect(paid).toEqual([]);
  });

  await test.step("pick three papers (one with full text, two abstract-only); limit of five", async () => {
    const results = page.getByRole("group", { name: "Search results" });
    await results.getByRole("button", { name: /Explore Paper 1/ }).click();
    await expect(compare).toBeDisabled(); // one is not enough
    await results.getByRole("button", { name: /Explore Paper 3/ }).click();
    await results.getByRole("button", { name: /Explore Paper 4/ }).click();
    await expect(picked.getByRole("listitem")).toHaveCount(3);
    await expect(compare).toBeEnabled();
    // two more fill the list; a sixth is refused with a message
    await results.getByRole("button", { name: /Explore Paper 5/ }).click();
    await results.getByRole("button", { name: /Explore Paper 6/ }).click();
    await expect(picked.getByRole("listitem")).toHaveCount(5);
    await results.getByRole("button", { name: /Explore Paper 2/ }).click();
    await expect(picked.getByRole("listitem")).toHaveCount(5);
    await expect(
      page.getByText("You can compare at most 5 papers"),
    ).toBeVisible();
    // back to three
    await page
      .getByRole("button", { name: "Remove Invented Explore Paper 6" })
      .click();
    await page
      .getByRole("button", { name: "Remove Invented Explore Paper 5" })
      .click();
    await expect(picked.getByRole("listitem")).toHaveCount(3);
    expect(paid).toEqual([]);
  });

  await test.step("dimensions: all six on; turn one off and on again", async () => {
    const compute = page.getByRole("button", { name: "Compute", exact: true });
    await expect(compute).toHaveAttribute("aria-pressed", "true");
    await compute.click();
    await expect(compute).toHaveAttribute("aria-pressed", "false");
    await compute.click();
    await expect(compute).toHaveAttribute("aria-pressed", "true");
  });

  await test.step("Compare sends exactly one request and shows the table", async () => {
    await compare.click();
    const table = page.getByRole("table");
    await expect(table).toBeVisible();
    expect(paid).toHaveLength(1);
    expect(paid[0]).toContain("/api/compare");
    await expect(table.getByRole("columnheader")).toHaveCount(4); // Dimension + 3 papers
    await expect(table.getByText("abstract only")).toHaveCount(2); // papers 3 and 4
    await expect(table.getByText("full text")).toHaveCount(1);
    const data = table.getByRole("row", { name: /^Data/ });
    await expect(data.locator('td[data-status="not_stated"]')).toHaveCount(2);
    await expect(data.locator('td[data-status="cited"]')).toHaveCount(1);
    await expect(
      table.locator('td[data-status="not_stated"]').first(),
    ).toHaveText("Not stated in the sources.");
    // 6 rows x 3 papers = 18 cells; the two abstract-only papers say nothing about data
    await expect(page.getByTestId("cell-counts")).toContainText(
      "16 of 18 cells have a claim with a source. 2 say",
    );
    // every cited cell has at least one chip
    const cited = table.locator('td[data-status="cited"]');
    expect(await cited.count()).toBe(16);
    for (let i = 0; i < 16; i++)
      expect(await cited.nth(i).getByRole("button").count()).toBeGreaterThan(0);
  });

  await test.step("a chip on a full-text paper opens the passage beside the table", async () => {
    const panel = page.getByRole("dialog");
    await page
      .getByRole("row", { name: /^Method/ })
      .getByRole("button", { name: /^Source 1: / })
      .click();
    await expect(panel).toBeVisible();
    await expect(panel.getByTestId("cited-passage")).toHaveText(
      "Invented section text of Invented Explore Paper 1.",
    );
    await expect(panel.getByText("Full-text passage")).toBeVisible();
    await expect(
      panel.getByRole("link", { name: "Read the abstract on arXiv" }),
    ).toHaveAttribute("href", "https://arxiv.org/abs/2611.00001");
  });

  await test.step("a chip on an abstract-only paper says it rests on the abstract", async () => {
    const panel = page.getByRole("dialog");
    // The panel is non-modal: the table behind it stays clickable.
    await page
      .getByRole("row", { name: /^Method/ })
      .getByRole("button", { name: /^Source 2: / })
      .click();
    await expect(panel.getByTestId("cited-passage")).toHaveText(
      "Invented abstract text of Invented Explore Paper 3.",
    );
    await expect(
      panel.getByText("Abstract only", { exact: true }),
    ).toBeVisible();
    await expect(panel.getByText(/rests on the summary/)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(panel).toBeHidden();
  });

  await test.step("export: download and copy give the same markdown", async () => {
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download as markdown" }).click(),
    ]);
    expect(download.suggestedFilename()).toBe(
      "comparison-2611.00001_2611.00003_2611.00004.md",
    );
    const file = await download.path();
    const markdown = await readFile(file, "utf8");
    expect(markdown).toContain(
      "| Dimension | Invented Explore Paper 1 (arXiv:2611.00001) |",
    );
    expect(markdown).toContain("(arXiv:2611.00003, abstract only)");
    expect(markdown).toContain("_Not stated in the sources._");
    expect(markdown).toContain("Invented method claim about paper 1. [1]");
    expect(markdown).toContain(
      "1. Invented Explore Paper 1 (arXiv:2611.00001), 3 Method. https://arxiv.org/abs/2611.00001",
    );
    expect(markdown).not.toContain(".pdf");

    await page.getByRole("button", { name: "Copy markdown" }).click();
    await expect(page.getByText("Copied to the clipboard.")).toBeVisible();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
      markdown,
    );
  });

  expect(paid).toHaveLength(1); // the whole flow made one paid call
  expect(pageErrors).toEqual([]);
  expect(blocked).toEqual([]);
});

test("timeline: typing sends nothing; Run draws the papers and the key lines", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  const paid: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") paid.push(request.url());
  });

  await page.goto("/timeline", { waitUntil: "domcontentloaded" });
  const topic = page.getByLabel("Topic");
  const run = page.getByRole("button", { name: "Run", exact: true });

  await test.step("typing a topic sends nothing; a too short topic is refused by the form", async () => {
    await expect(run).toBeDisabled();
    // Typing before React has hydrated the page is lost, so retry until it sticks.
    await expect(async () => {
      await topic.fill(""); // React ignores a fill that repeats the value it already has
      await topic.fill("ad");
      await expect(run).toBeEnabled({ timeout: 500 });
    }).toPass();
    await run.click();
    await expect(
      page.getByText("The topic needs at least 3 characters"),
    ).toBeVisible();
    expect(paid).toEqual([]);
  });

  await test.step("Run sends one request and draws six dots, two of them key papers", async () => {
    await topic.fill("adapter methods");
    await run.click();
    const chart = page.getByTestId("timeline-chart");
    await expect(chart).toBeVisible();
    expect(paid).toHaveLength(1);
    expect(paid[0]).toContain("/api/timeline");
    await expect(chart.getByTestId("timeline-dot")).toHaveCount(6);
    await expect(chart.locator('[data-key="true"]')).toHaveCount(2);
    await expect(chart).toHaveAttribute("data-axis-unit", "day"); // a narrow range
  });

  await test.step("the list: dates, the one-line summaries, labels and links", async () => {
    const entries = page
      .getByRole("list", { name: "Papers by date" })
      .getByRole("listitem");
    await expect(entries).toHaveCount(6);
    const first = entries.first();
    await expect(first.getByRole("heading")).toHaveText(
      "Invented Explore Paper 1",
    );
    await expect(first.getByText("Key paper")).toBeVisible();
    await expect(
      first.getByText("Invented line: introduced the first adapter."),
    ).toBeVisible();
    await expect(
      first.getByText("from the abstract", { exact: true }),
    ).toBeVisible();
    await expect(first.getByText("cited by 120")).toBeVisible();
    await expect(
      first.getByRole("link", { name: "Open the paper page" }),
    ).toHaveAttribute("href", "/paper/2611.00001");
    await expect(
      first.getByRole("link", { name: "arXiv 2611.00001" }),
    ).toHaveAttribute("href", "https://arxiv.org/abs/2611.00001");
    const fallback = entries.nth(3);
    await expect(
      fallback.getByText("from the abstract (first sentence)"),
    ).toBeVisible();
    await expect(entries.nth(1).getByText("from the abstract")).toHaveCount(0); // not a key paper
    await expect(
      entries.nth(1).getByText("citation count unknown"),
    ).toBeVisible();
  });

  await test.step("the notes say how thin the data is", async () => {
    const notes = page.getByTestId("timeline-notes");
    await expect(notes).toContainText("6 papers, 2 marked as key.");
    await expect(notes).toContainText("known for 1 of 6 papers");
    await expect(notes).toContainText("only 10 days");
    await expect(notes).toContainText(
      "1 line is the first sentence of the abstract",
    );
  });

  expect(paid).toHaveLength(1);
  expect(pageErrors).toEqual([]);
  expect(blocked).toEqual([]);
});

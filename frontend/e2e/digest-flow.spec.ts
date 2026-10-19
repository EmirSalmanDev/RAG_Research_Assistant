// The digest and the settings page: open the digest, save one card, dismiss one, undo,
// then add, edit and delete an interest and save the settings.
//
// WHAT THIS PROVES, AND WHAT IT DOES NOT. The API behind /api is the mock in
// e2e/mock-api.ts (invented papers, in memory), reached through the real Next.js /api
// rewrite. So this proves the pages, the optimistic buttons and the forms; it does not
// prove the scoring or the weight changes (the backend tests cover those) and it says
// nothing about real data (a separate run against the real API does).
//
// CONTAINMENT. Same as the other browser tests: every non-local request is aborted and
// recorded, and links are checked by their href only, never opened.
import { expect, test } from "@playwright/test";

import { isAllowedUrl } from "./containment.ts";

test("the digest: save, dismiss and undo; settings: add an interest", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (isAllowedUrl(url)) return route.continue();
    blocked.push(url);
    return route.abort("blockedbyclient");
  });
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  const posts: string[] = [];
  page.on("request", (request) => {
    if (request.method() !== "GET")
      posts.push(`${request.method()} ${request.url()}`);
  });

  await page.goto("/", { waitUntil: "domcontentloaded" });
  await expect(
    page.getByRole("heading", { level: 1, name: "Mon, Oct 19, 2026" }),
  ).toBeVisible();

  await test.step("the cards, in rank order, with their facts", async () => {
    const cards = page.getByRole("article");
    await expect(cards).toHaveCount(3);
    await expect(cards.first().getByRole("heading")).toHaveText(
      "Invented Digest Paper 1",
    );
    await expect(cards.first().getByText("Trending")).toBeVisible();
    await expect(
      cards.first().getByText("Ada Example, Bo Sample, Cy Test and 2 more"),
    ).toBeVisible();
    await expect(
      cards.first().getByText("Invented one-line summary number 1."),
    ).toBeVisible();
    await expect(
      cards.first().getByRole("link", { name: "Open paper page" }),
    ).toHaveAttribute("href", "/paper/2610.00001");
    await expect(
      cards.first().getByRole("link", { name: "arXiv 2610.00001" }),
    ).toHaveAttribute("href", "https://arxiv.org/abs/2610.00001");
    await expect(cards.nth(1).getByText("Trending")).toHaveCount(0);
  });

  await test.step("no interests yet: the page says so and links to settings", async () => {
    const note = page.getByRole("note");
    await expect(note).toContainText("Add interests to rank this list.");
    await expect(
      note.getByRole("link", { name: "Add an interest" }),
    ).toHaveAttribute("href", "/settings");
  });

  await test.step("save one card, dismiss another, undo the dismissal", async () => {
    const first = page.getByRole("article").nth(0);
    const second = page.getByRole("article").nth(1);
    await first.getByRole("button", { name: /^Save:/ }).click();
    await expect(first).toHaveAttribute("data-status", "saved");
    await expect(first.getByText("Saved to your library.")).toBeVisible();
    await expect(first.getByRole("button", { name: /^Save:/ })).toHaveCount(0);

    await second.getByRole("button", { name: /^Dismiss:/ }).click();
    await expect(second).toHaveAttribute("data-status", "dismissed");
    await expect(
      second.getByText("It will not come back.", { exact: false }),
    ).toBeVisible();

    await second.getByRole("button", { name: /^Undo:/ }).click();
    await expect(second).toHaveAttribute("data-status", "new");
    await expect(second.getByRole("button", { name: /^Save:/ })).toBeVisible();
    await expect(
      second.getByRole("button", { name: /^Dismiss:/ }),
    ).toBeVisible();

    // the server kept the saved state: a fresh load shows it
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(page.getByRole("article").nth(0)).toHaveAttribute(
      "data-status",
      "saved",
    );
    await expect(page.getByRole("article").nth(1)).toHaveAttribute(
      "data-status",
      "new",
    );
    expect(posts.filter((p) => p.includes("/api/digest/"))).toEqual([
      expect.stringContaining(
        "POST http://127.0.0.1:3999/api/digest/2610.00001/save?date=2026-10-19",
      ),
      expect.stringContaining(
        "POST http://127.0.0.1:3999/api/digest/2610.00002/dismiss?date=2026-10-19",
      ),
      expect.stringContaining(
        "POST http://127.0.0.1:3999/api/digest/2610.00002/undo?date=2026-10-19",
      ),
    ]);
  });

  await test.step("settings: add an interest", async () => {
    await page.getByRole("link", { name: "Settings" }).click();
    await expect(
      page.getByRole("heading", { level: 1, name: "Settings" }),
    ).toBeVisible();
    await expect(page.getByText("No interests yet.")).toBeVisible();

    const form = page.getByRole("form", { name: "New interest" });
    await form.getByLabel("Name").fill("Retrieval-augmented generation");
    await form.getByLabel("Description").fill("Papers about RAG.");
    await form.getByLabel("Keywords").fill("rag, retrieval");
    await form.getByLabel("arXiv categories").fill("cs.CL, nope!");
    await form.getByRole("button", { name: "Add interest" }).click();
    await expect(form.getByRole("alert")).toContainText(
      "not an arXiv category",
    );
    await form.getByLabel("arXiv categories").fill("cs.CL, cs.IR");
    await form.getByRole("button", { name: "Add interest" }).click();

    const item = page
      .getByRole("listitem")
      .filter({ hasText: "Retrieval-augmented generation" });
    await expect(item.getByText("weight 1.00")).toBeVisible();
    await expect(
      item.getByText("waiting for the next nightly embedding"),
    ).toBeVisible();
    await expect(item.getByText("rag", { exact: true })).toBeVisible();
    await expect(form.getByLabel("Name")).toHaveValue(""); // emptied after the add

    // a second interest with the same name: the server's sentence, nothing lost
    await form.getByLabel("Name").fill("Retrieval-augmented generation");
    await form.getByRole("button", { name: "Add interest" }).click();
    await expect(form.getByRole("alert")).toHaveText(
      "an interest with this name exists",
    );
    await expect(form.getByLabel("Name")).toHaveValue(
      "Retrieval-augmented generation",
    );
    await form.getByLabel("Name").fill("LLM agents");
    await form.getByRole("button", { name: "Add interest" }).click();
    await expect(
      page.getByRole("heading", { level: 3, name: "LLM agents" }),
    ).toBeVisible();
  });

  await test.step("edit one, delete the other (with a confirmation)", async () => {
    await page.getByRole("button", { name: "Edit LLM agents" }).click();
    const edit = page.getByRole("form", { name: "Edit LLM agents" });
    await edit.getByLabel("Keywords").fill("agent, tool use");
    await edit.getByLabel("Active").uncheck();
    await edit.getByRole("button", { name: "Save changes" }).click();
    const item = page.getByRole("listitem").filter({ hasText: "LLM agents" });
    await expect(item.getByText("Inactive")).toBeVisible();
    await expect(item.getByText("tool use", { exact: true })).toBeVisible();

    await page.getByRole("button", { name: "Delete LLM agents" }).click();
    await expect(page.getByText("Delete this interest?")).toBeVisible();
    await page.getByRole("button", { name: "Keep" }).click();
    await expect(
      page.getByRole("heading", { level: 3, name: "LLM agents" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Delete LLM agents" }).click();
    await page
      .getByRole("button", { name: "Confirm delete LLM agents" })
      .click();
    await expect(
      page.getByRole("heading", { level: 3, name: "LLM agents" }),
    ).toHaveCount(0);
  });

  await test.step("settings: digest size, categories, budget cap and the month's spending", async () => {
    await expect(page.getByTestId("month-spend")).toHaveText("$0.0450");
    const form = page.getByRole("form", { name: "Digest settings" });
    await form.getByLabel("Digest size").fill("0");
    await form.getByRole("button", { name: "Save settings" }).click();
    await expect(form.getByRole("alert")).toContainText(
      "Digest size must be a whole number",
    );
    await form.getByLabel("Digest size").fill("12");
    await form.getByLabel("arXiv categories to fetch").fill("cs.CV, stat.ML");
    await form.getByLabel("Monthly budget cap (USD)").fill("7.50");
    await form.getByRole("button", { name: "Save settings" }).click();
    await expect(form.getByText("Settings saved.")).toBeVisible();

    await page.reload({ waitUntil: "domcontentloaded" });
    const reloaded = page.getByRole("form", { name: "Digest settings" });
    await expect(reloaded.getByLabel("Digest size")).toHaveValue("12");
    await expect(reloaded.getByLabel("arXiv categories to fetch")).toHaveValue(
      "cs.CV, stat.ML",
    );
    await expect(reloaded.getByLabel("Monthly budget cap (USD)")).toHaveValue(
      "7.5",
    );
  });

  await test.step("back on the digest, the interests note is gone", async () => {
    // one interest is left and it is active, so the digest no longer asks for interests
    await page.getByRole("link", { name: "Digest" }).click();
    await expect(page.getByRole("article")).toHaveCount(3);
    await expect(page.getByRole("note")).toHaveCount(0);
  });

  expect(pageErrors).toEqual([]);
  expect(blocked).toEqual([]);
});

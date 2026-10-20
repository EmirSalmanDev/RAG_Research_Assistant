// The library: collections, papers, status, notes, the .bib file, the review and a PDF upload.
//
// WHAT THIS PROVES, AND WHAT IT DOES NOT. The API behind /api is the mock in
// e2e/mock-api.ts (invented papers, in memory, no model), reached through the real Next.js
// /api rewrite. So this proves the page: that typing and choosing send nothing paid, that the
// two paid buttons send exactly one request, the table and its edits, the .bib download, the
// draft with its evidence labels, that a citation chip opens the sources panel with the
// passage, the markdown export, and that a raw PDF body reaches the API intact through the
// rewrite (also above 10 MB). It does not prove that the real model cites correctly or that
// a real PDF parses (the backend tests and a separate run against the real API do).
//
// CONTAINMENT. Same as the other browser tests: every non-local request is aborted and
// recorded, and links are checked by their href only, never opened.
import { readFile } from "node:fs/promises";

import { expect, test } from "@playwright/test";
import type { BrowserContext, Page } from "@playwright/test";

import { API_PORT, isAllowedUrl } from "./containment.ts";

test.beforeEach(async ({ context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
});

async function contain(context: BrowserContext, blocked: string[]) {
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (isAllowedUrl(url)) return route.continue();
    blocked.push(url);
    return route.abort("blockedbyclient");
  });
}

const PAID = /\/api\/library\/(review|upload)/;

function watch(page: Page) {
  const paid: string[] = [];
  const pageErrors: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && PAID.test(request.url()))
      paid.push(request.url());
  });
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  return { paid, pageErrors };
}

async function createCollection(page: Page, name: string) {
  await page.getByLabel("New collection").fill(name);
  await page.getByRole("button", { name: "Create", exact: true }).click();
  await expect(page.getByRole("heading", { name, level: 2 })).toBeVisible();
}

test("library: collection, papers, status, note, .bib, review, chip, export", async ({
  page,
  context,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const { paid, pageErrors } = watch(page);

  await page.goto("/library", { waitUntil: "domcontentloaded" });
  const generate = page.getByRole("button", { name: "Generate review" });

  await test.step("an empty library: nothing to review, nothing sent", async () => {
    await expect(page.getByText("No collection yet.")).toBeVisible();
    await expect(page.getByText("Create a collection to start.")).toBeVisible();
    await expect(generate).toBeDisabled();
    await expect(page.getByText("Pick a collection.")).toBeVisible();
  });

  await test.step("create a collection (it starts empty)", async () => {
    await createCollection(page, "Thesis ch. 2");
    await expect(
      page.getByText("0 papers", { exact: true }).first(),
    ).toBeVisible();
    await expect(page.getByText("This collection is empty.")).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Download .bib" }),
    ).toBeDisabled();
    await expect(generate).toBeDisabled(); // below two papers
    await expect(
      page.getByText(/needs at least 2 papers \(Thesis ch\. 2 has 0\)/),
    ).toBeVisible();
  });

  await test.step("add papers by link and by id; an unknown paper gets the hint", async () => {
    const box = page.getByLabel("Add a paper by arXiv id or link");
    const add = page.getByRole("button", { name: "Add paper" });
    await expect(add).toBeDisabled();
    await box.fill("https://arxiv.org/abs/2611.00001v2");
    await add.click();
    await expect(
      page.getByRole("link", { name: "Invented Explore Paper 1" }),
    ).toBeVisible();
    await box.fill("2611.00003");
    await add.click();
    await expect(
      page.getByRole("link", { name: "Invented Explore Paper 3" }),
    ).toBeVisible();
    const table = page.getByRole("table");
    await expect(table.getByText("full text", { exact: true })).toHaveCount(1);
    await expect(table.getByText("abstract only", { exact: true })).toHaveCount(
      1,
    );
    await box.fill("2611.00099");
    await add.click();
    await expect(
      page.getByRole("alert").filter({ hasText: "Ingest the paper first" }),
    ).toBeVisible();
    await expect(table.getByRole("row")).toHaveCount(3); // header + two papers
    // typing and choosing sent nothing paid
    expect(paid).toEqual([]);
  });

  await test.step("change a status; it survives a reload (the server holds it)", async () => {
    const status = page.getByLabel("Status of Invented Explore Paper 1");
    await expect(status).toHaveValue("to_read");
    await status.selectOption("reading");
    await expect(status).toHaveValue("reading");
    await page.reload({ waitUntil: "domcontentloaded" });
    await expect(
      page.getByLabel("Status of Invented Explore Paper 1"),
    ).toHaveValue("reading");
    await expect(
      page.getByLabel("Status of Invented Explore Paper 3"),
    ).toHaveValue("to_read");
  });

  await test.step("edit a note", async () => {
    await page
      .getByRole("button", { name: "Add note for Invented Explore Paper 1" })
      .click();
    await page
      .getByLabel("Note for Invented Explore Paper 1")
      .fill("Check the ablation in section 4.");
    await page.getByRole("button", { name: "Save note" }).click();
    await expect(
      page.getByText("Check the ablation in section 4."),
    ).toBeVisible();
    await expect(
      page.getByRole("button", {
        name: "Edit note for Invented Explore Paper 1",
      }),
    ).toBeVisible();
  });

  let bib = "";
  await test.step("download the .bib file", async () => {
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download .bib" }).click(),
    ]);
    expect(download.suggestedFilename()).toBe("thesis-ch-2.bib");
    bib = await readFile(await download.path(), "utf8");
    expect(bib.match(/^@misc\{/gm)).toHaveLength(2);
    expect(bib).toContain("url = {https://arxiv.org/abs/2611.00001}");
    await page.getByRole("button", { name: "Copy BibTeX" }).click();
    await expect(page.getByText("Copied to the clipboard.")).toBeVisible();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(bib);
  });

  await test.step("Generate review: focus typed sends nothing, the button sends one request", async () => {
    await page.getByLabel("Focus (optional)").fill("small adapters");
    await expect(generate).toBeEnabled();
    expect(paid).toEqual([]);
    await generate.click();
    const draft = page.getByRole("region", { name: "Related-work draft" });
    await expect(draft).toBeVisible();
    expect(paid).toHaveLength(1);
    expect(paid[0]).toContain("/api/library/review");
    // the honest label, the removed claims, the evidence levels
    await expect(draft.getByTestId("evidence-note")).toContainText(
      "Built from model-generated analyses and abstracts",
    );
    await expect(draft.getByTestId("removed-note")).toContainText(
      "2 claims removed because they had no valid citation.",
    );
    await expect(
      draft.getByRole("button", {
        name: /^Source 1: .*\(generated analysis\)$/,
      }),
    ).toBeVisible();
    await expect(
      draft.getByRole("button", { name: /^Source 2: .*\(abstract only\)$/ }),
    ).toBeVisible();
    // references carry the BibTeX keys of the .bib file
    const keys = [...bib.matchAll(/^@misc\{([^,]+),/gm)].map((m) => m[1]);
    for (const key of keys)
      await expect(draft.getByText(key, { exact: true })).toBeVisible();
    await expect(draft.getByTestId("reference-1")).toContainText(
      "generated analysis",
    );
    await expect(draft.getByTestId("reference-2")).toContainText(
      "abstract only",
    );
  });

  await test.step("a chip on an analysis opens the passage beside the draft", async () => {
    const panel = page.getByRole("dialog");
    await page.getByRole("button", { name: /^Source 1: / }).click();
    await expect(panel).toBeVisible();
    await expect(panel.getByTestId("cited-passage")).toHaveText(
      "Invented analysis text of Invented Explore Paper 1.",
    );
    await expect(
      panel.getByText("Generated analysis", { exact: true }),
    ).toBeVisible();
    await expect(
      panel.getByText(/model-generated analysis of the paper/),
    ).toBeVisible();
    await expect(
      panel.getByRole("link", { name: "Read the abstract on arXiv" }),
    ).toHaveAttribute("href", "https://arxiv.org/abs/2611.00001");
  });

  await test.step("a chip on an abstract says it rests on the abstract", async () => {
    const panel = page.getByRole("dialog");
    await page.getByRole("button", { name: /^Source 2: / }).click(); // the panel is non-modal
    await expect(panel.getByTestId("cited-passage")).toHaveText(
      "Invented abstract text of Invented Explore Paper 3.",
    );
    await expect(
      panel.getByText("Abstract only", { exact: true }),
    ).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(panel).toBeHidden();
  });

  await test.step("export the draft: download and copy give the same markdown", async () => {
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download as markdown" }).click(),
    ]);
    expect(download.suggestedFilename()).toBe("related-work-thesis-ch-2.md");
    const markdown = await readFile(await download.path(), "utf8");
    expect(markdown).toContain("# Related work: Thesis ch. 2");
    expect(markdown).toContain("Focus: small adapters");
    expect(markdown).toMatch(/studied invented explore paper 1\.\[1\]/);
    expect(markdown).toMatch(/studied invented explore paper 3\.\[2\]/);
    for (const key of [...bib.matchAll(/^@misc\{([^,]+),/gm)].map((m) => m[1]))
      expect(markdown).toContain(`\`${key}\``);
    expect(markdown).toContain("https://arxiv.org/abs/2611.00003");
    expect(markdown).toContain(
      "2 sentences removed for lack of a valid citation",
    );
    expect(markdown).not.toContain(".pdf");
    await page.getByRole("button", { name: "Copy markdown" }).click();
    await expect(
      page.getByText("Copied to the clipboard.").last(),
    ).toBeVisible();
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
      markdown,
    );
  });

  await test.step("move a paper, remove a paper, rename and delete the collection", async () => {
    await createCollection(page, "Reading list");
    await page.getByRole("button", { name: /^Thesis ch\. 2/ }).click();
    await page
      .getByLabel("Move Invented Explore Paper 3 to")
      .selectOption("Reading list");
    await expect(
      page.getByRole("link", { name: "Invented Explore Paper 3" }),
    ).toHaveCount(0);
    await page
      .getByRole("button", { name: "Remove Invented Explore Paper 1" })
      .click();
    await expect(page.getByText("This collection is empty.")).toBeVisible();
    await page.getByRole("button", { name: /^Reading list/ }).click();
    await expect(
      page.getByRole("link", { name: "Invented Explore Paper 3" }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Rename", exact: true }).click();
    await page.getByLabel(/^Rename .* to$/).fill("To read later");
    await page.getByRole("button", { name: "Save name" }).click();
    await expect(
      page.getByRole("heading", { name: "To read later", level: 2 }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Delete collection" }).click();
    const confirm = page.getByRole("alertdialog", { name: "Confirm delete" });
    await expect(confirm).toContainText(
      "The papers themselves stay in the database",
    );
    await confirm.getByRole("button", { name: "Keep it" }).click();
    await expect(confirm).toBeHidden();
    await page.getByRole("button", { name: "Delete collection" }).click();
    await page.getByRole("button", { name: "Yes, delete it" }).click();
    await expect(
      page.getByRole("heading", { name: "To read later", level: 2 }),
    ).toHaveCount(0);
    // leave the mock as it was found: the next test starts from an empty library
    await page.getByRole("button", { name: /^Thesis ch\. 2/ }).click();
    await page.getByRole("button", { name: "Delete collection" }).click();
    await page.getByRole("button", { name: "Yes, delete it" }).click();
    await expect(page.getByText("Create a collection to start.")).toBeVisible();
  });

  expect(paid).toHaveLength(1); // the whole flow made one paid call
  expect(pageErrors).toEqual([]);
  expect(blocked).toEqual([]);
});

test("library: a PDF upload sends one request, repeats cost nothing, big files arrive whole", async ({
  page,
  context,
  request,
}) => {
  const blocked: string[] = [];
  await contain(context, blocked);
  const { paid, pageErrors } = watch(page);

  await page.goto("/library", { waitUntil: "domcontentloaded" });
  // The collections have loaded, so the page has hydrated and typing reaches React state.
  await expect(page.getByText("Loading the library")).toHaveCount(0);
  await createCollection(page, "Uploads");
  const upload = page.getByRole("button", { name: "Upload", exact: true });
  const chooser = page.getByLabel("PDF file");
  const card = page.getByRole("region", { name: "Upload a PDF" });

  await test.step("nothing is sent by choosing a file; the cost is stated", async () => {
    await expect(upload).toBeDisabled();
    await expect(card.getByText("Choose a PDF file.")).toBeVisible();
    await expect(card).toContainText(
      "about $0.0002 per 10 pages of embeddings",
    );
    await chooser.setInputFiles({
      name: "my_journal-paper.pdf",
      mimeType: "application/pdf",
      buffer: Buffer.from("%PDF-1.7\nan invented small document\n"),
    });
    await expect(page.getByLabel("Title")).toHaveValue("my journal paper");
    await expect(page.getByTestId("file-size")).toHaveText(
      "my_journal-paper.pdf, 36 B",
    );
    await expect(upload).toBeEnabled();
    expect(paid).toEqual([]);
  });

  await test.step("a title is required", async () => {
    await page.getByLabel("Title").fill("   ");
    await expect(upload).toBeDisabled();
    await expect(card.getByText("The title is required.")).toBeVisible();
    await page.getByLabel("Title").fill("An Invented Journal Paper");
    await page.getByLabel("Year").fill("2019");
  });

  await test.step("Upload sends one request and the document appears as uploaded", async () => {
    await upload.click();
    await expect(
      card.getByText(/Added "An Invented Journal Paper"/),
    ).toBeVisible();
    expect(paid).toHaveLength(1);
    expect(paid[0]).toContain(
      "/api/library/upload?title=An+Invented+Journal+Paper",
    );
    const row = page.getByRole("row", { name: /An Invented Journal Paper/ });
    await expect(row.getByText("uploaded", { exact: true })).toBeVisible();
    await expect(row.getByText("uploaded document")).toBeVisible();
    await expect(row.getByRole("link", { name: /^arXiv / })).toHaveCount(0); // no arXiv link
    await expect(
      page.getByRole("button", { name: "Download .bib" }),
    ).toBeEnabled();
  });

  await test.step("the same bytes again change nothing", async () => {
    await chooser.setInputFiles({
      name: "my_journal-paper.pdf",
      mimeType: "application/pdf",
      buffer: Buffer.from("%PDF-1.7\nan invented small document\n"),
    });
    await page.getByLabel("Title").fill("A different title");
    await upload.click();
    await expect(
      card.getByText(/was uploaded before. Nothing changed/),
    ).toBeVisible();
    expect(paid).toHaveLength(2);
  });

  await test.step("a 12 MB file reaches the API whole through the proxy; 26 MB is refused here", async () => {
    const size = 12 * 1024 * 1024;
    const big = Buffer.alloc(size, 0x20);
    big.write("%PDF-1.7\n", 0);
    await chooser.setInputFiles({
      name: "big.pdf",
      mimeType: "application/pdf",
      buffer: big,
    });
    await expect(page.getByTestId("file-size")).toHaveText("big.pdf, 12.0 MB");
    await page.getByLabel("Title").fill("A big invented document");
    await upload.click();
    await expect(
      card.getByText(/Added "A big invented document"/),
    ).toBeVisible();
    const log = await (
      await request.get(`http://127.0.0.1:${API_PORT}/__uploads`)
    ).json();
    expect(log.at(-1)).toEqual({
      bytes: size,
      title: "A big invented document",
    });

    const tooBig = Buffer.alloc(26 * 1024 * 1024, 0x20);
    await chooser.setInputFiles({
      name: "huge.pdf",
      mimeType: "application/pdf",
      buffer: tooBig,
    });
    await page.getByLabel("Title").fill("Too big");
    await expect(upload).toBeDisabled();
    await expect(card.getByText(/limit is 25\.0 MB/)).toBeVisible();
    expect(paid).toHaveLength(3);
  });

  await test.step("the .bib marks an uploaded document and has no link for it", async () => {
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download .bib" }).click(),
    ]);
    const bib = await readFile(await download.path(), "utf8");
    expect(bib).toContain("note = {Uploaded document}");
    expect(bib).not.toContain("eprint = {local-");
  });

  expect(pageErrors).toEqual([]);
  expect(blocked).toEqual([]);
});

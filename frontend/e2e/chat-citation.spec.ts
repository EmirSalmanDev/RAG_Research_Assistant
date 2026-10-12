// The one browser test: ask a question, watch the answer stream in, open a citation, read
// the cited passage beside the answer.
//
// WHAT THIS PROVES, AND WHAT IT DOES NOT. The API behind /api is the mock in
// e2e/mock-api.ts (invented data), reached through the real Next.js /api rewrite. So this
// proves the UI and that the rewrite passes the event stream through unbuffered. It does
// not prove anything about the real backend, the real retrieval or real answers; the event
// shapes are checked against the backend by backend/tests/unit/test_sse_fixtures.py and
// lib/chat/recorded.test.ts, and a run against the real API is a separate, paid step.
//
// CONTAINMENT. The test never navigates to or clicks an external URL: the arXiv link is
// checked by its href only. Every non-local request is aborted by the route handler below,
// and the test makes one external request on purpose (to a .invalid name, which resolves
// nowhere) to prove it. Chrome and the servers run inside the sandbox-exec profile
// e2e/e2e.sb with Chrome's resolver rules (see e2e/containment.ts).
import { expect, test } from "@playwright/test";

import { API_PORT, isAllowedUrl } from "./containment.ts";

const QUESTION = "How do adapters reduce the number of trained parameters?";
const FIRST_TITLE = "An Invented Paper About Adapters"; // source [1], an abstract
const SECOND_TITLE = "A Second Invented Paper"; // source [2], a full-text passage

const diagnostics: string[] = [];

test.afterEach(async () => {
  const info = test.info();
  if (info.status !== info.expectedStatus) {
    console.log(
      "browser diagnostics:\n  " +
        (diagnostics.length ? diagnostics.join("\n  ") : "(none)"),
    );
  }
});

test("ask, watch the answer stream, open a citation and read the cited passage", async ({
  page,
  context,
  request,
  browser,
}) => {
  test
    .info()
    .annotations.push({ type: "chrome", description: browser.version() });

  // --- containment: abort everything that is not local, and remember it ---
  const blocked: string[] = [];
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (isAllowedUrl(url)) return route.continue();
    blocked.push(url);
    return route.abort("blockedbyclient");
  });
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  // what the browser says; printed when the test fails (it is the first thing to read)
  page.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warning")
      diagnostics.push(`console.${msg.type()}: ${msg.text()}`);
  });
  page.on("requestfailed", (req) =>
    diagnostics.push(
      `request failed: ${req.url()} (${req.failure()?.errorText})`,
    ),
  );
  page.on("response", (res) => {
    if (res.status() >= 400)
      diagnostics.push(`HTTP ${res.status()}: ${res.url()}`);
  });
  // the page records when each stream event arrives (lib/chat/debug.ts)
  await page.addInitScript(() =>
    window.localStorage.setItem("chat-debug", "1"),
  );

  const composer = page.getByRole("textbox", { name: "Your question" });
  const panel = page.getByRole("dialog");
  const chip1 = page.getByRole("button", {
    name: /^Source 1: .*\(abstract only\)$/,
  });
  const chip2 = page.getByRole("button", { name: /^Source 2: / });
  const scrollY = () => page.evaluate(() => window.scrollY);

  await test.step("ask a question", async () => {
    await page.goto("/chat", { waitUntil: "domcontentloaded" });
    const ask = page.getByRole("button", { name: "Ask", exact: true });
    // The page is server-rendered first, and text typed before React has hydrated is not
    // seen by the app: after hydration React still thinks the box has the old value, so
    // filling the same text again fires no change. Clear it first, then fill, until the app
    // has accepted the text and enabled Ask.
    await expect(async () => {
      await composer.fill("");
      await composer.fill(QUESTION);
      await expect(ask).toBeEnabled({ timeout: 500 });
    }).toPass({ timeout: 15_000 });
    await ask.click();
    await expect(page.getByRole("heading", { name: QUESTION })).toBeVisible();
  });

  await test.step("the answer streams in while the turn is still running", async () => {
    // The mock holds the stream open after the first tokens until we release it.
    await expect(
      page.getByText("Adapters train a small module beside the frozen model"),
    ).toBeVisible();
    await expect(page.getByRole("button", { name: "Stop" })).toBeVisible(); // still running
    await expect(page.getByText(/tokens, \$/)).toHaveCount(0); // no cost line: not done
    await expect(page.getByText(/^All \d+ sources$/)).toHaveCount(0);
    await expect(page.getByRole("button", { name: /^Source \d/ })).toHaveCount(
      0,
    ); // no citations yet

    const released = await request.post(
      `http://127.0.0.1:${API_PORT}/__release`,
    );
    expect(released.ok()).toBe(true);

    await expect(page.getByText(/tokens, \$/)).toBeVisible(); // done
    await expect(
      page.getByText("Invented point 14 about adapters."),
    ).toBeVisible();

    // arrival times seen by the page: the tokens came as separate events, before `done`
    const timeline = await page.evaluate(() => window.__chatTimeline ?? []);
    const tokens = timeline.filter((entry) => entry.event === "token");
    const done = timeline.find((entry) => entry.event === "done");
    expect(tokens.length).toBeGreaterThanOrEqual(4);
    expect(done).toBeDefined();
    expect(tokens[0]!.ms).toBeLessThan(done!.ms);
    console.log(
      "chat timeline (event, ms after the request):",
      JSON.stringify(timeline.map((e) => [e.event, e.ms])),
    );
  });

  await test.step("citations are chips; the abstract source carries the badge", async () => {
    await expect(chip1).toBeVisible();
    await expect(chip1).toContainText("abstract"); // the badge
    await expect(chip2).toBeVisible();
    await expect(chip2).not.toContainText("abstract");
    await expect(chip2).toHaveAttribute(
      "aria-label",
      new RegExp(`^Source 2: ${SECOND_TITLE}$`),
    );
  });

  await test.step("a citation opens the cited passage in a side panel", async () => {
    await chip1.click();
    await expect(panel).toBeVisible();
    await expect(
      panel.getByRole("heading", { name: FIRST_TITLE }),
    ).toBeVisible();
    await expect(panel.getByTestId("cited-passage")).toHaveText(
      `Invented passage text for ${FIRST_TITLE}.`,
    );
    await expect(
      panel.getByText("Abstract only", { exact: true }),
    ).toBeVisible();

    // The arXiv link is checked by its href and never clicked or followed.
    const link = panel.getByRole("link", {
      name: "Read the abstract on arXiv",
    });
    await expect(link).toHaveAttribute("href", /^https:\/\/arxiv\.org\/abs\//);
    await expect(link).toHaveAttribute("target", "_blank");
    await expect(link).toHaveAttribute("rel", /noopener/);
  });

  await test.step("the chat stays usable beside the open panel", async () => {
    // type in the composer
    await composer.click();
    await page.keyboard.type("a follow-up typed beside the panel");
    await expect(composer).toHaveValue("a follow-up typed beside the panel");

    // scroll the page: it must be taller than the window; go to a known place (the bottom),
    // then use the mouse wheel to scroll up. (Playwright's own click above scrolled the
    // page to bring the chip into view, so the position before this is not a given.)
    const tallerThanWindow = await page.evaluate(
      () => document.documentElement.scrollHeight > window.innerHeight,
    );
    expect(tallerThanWindow).toBe(true);
    await page.evaluate(() =>
      window.scrollTo(0, document.documentElement.scrollHeight),
    );
    const before = await scrollY();
    expect(before).toBeGreaterThan(0);
    await page.mouse.move(300, 200);
    await page.mouse.wheel(0, -200);
    await expect.poll(scrollY).toBeLessThan(before);

    // click another citation: the panel switches to it and stays open
    await chip2.click();
    await expect(panel).toBeVisible();
    await expect(panel.getByTestId("cited-passage")).toHaveText(
      `Invented passage text for ${SECOND_TITLE}.`,
    );
    await expect(panel.getByText("Full-text passage")).toBeVisible();
  });

  await test.step("an outside click does not close the panel; Escape and the close button do", async () => {
    await page.getByRole("heading", { name: QUESTION }).click();
    await expect(panel).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(panel).toBeHidden();

    await chip1.click();
    await expect(panel).toBeVisible();
    await panel.getByRole("button", { name: "Close" }).click();
    await expect(panel).toBeHidden();
  });

  await test.step("nothing left the machine", async () => {
    // one deliberate external request, to a name that resolves nowhere: it must be blocked
    const outcome = await page.evaluate(() =>
      fetch("https://blocked.invalid/probe").then(
        () => "reached",
        () => "blocked",
      ),
    );
    expect(outcome).toBe("blocked");
    expect(blocked).toEqual(["https://blocked.invalid/probe"]);
    expect(pageErrors).toEqual([]);
  });
});

import { defineConfig } from "@playwright/test";

import {
  API_PORT,
  CHROME_CONTAINMENT_ARGS,
  WEB_PORT,
} from "./e2e/containment.ts";

// The browser tests (e2e/*.spec.ts), run by `make e2e`. They are not part of
// `make test`. It drives the Google Chrome that is already installed (channel "chrome"),
// so no browser is downloaded. See README, "Browser test", for how traffic is contained.
export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.spec.ts", // the *.test.ts files in e2e/ belong to Node's runner
  outputDir: "./test-results",
  fullyParallel: false,
  workers: 1,
  retries: 0,
  forbidOnly: true,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    channel: "chrome",
    headless: true,
    viewport: { width: 1280, height: 480 }, // wide enough for the side-by-side layout, short enough to need scrolling
    serviceWorkers: "block",
    launchOptions: { args: CHROME_CONTAINMENT_ARGS },
    screenshot: "only-on-failure",
    trace: "off",
    video: "off",
  },
  webServer: [
    {
      // The mock API (invented data); the Next rewrite sends /api/* here.
      command: "node e2e/mock-api.ts",
      url: `http://127.0.0.1:${API_PORT}/health`,
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      // --webpack: Turbopack cannot bind its worker port inside the sandbox.
      command: `pnpm exec next dev --webpack -H 127.0.0.1 -p ${WEB_PORT}`,
      url: `http://127.0.0.1:${WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        API_ORIGIN: `http://127.0.0.1:${API_PORT}`,
        NEXT_TELEMETRY_DISABLED: "1",
        NEXT_IGNORE_INCORRECT_LOCKFILE: "1",
      },
    },
  ],
});

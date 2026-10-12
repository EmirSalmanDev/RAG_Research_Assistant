// What the browser test is allowed to reach, in one place. Plain TypeScript with no
// imports: the Playwright config, the test and Node's unit-test runner all use it.
//
// The e2e test must never touch the internet. Four layers hold that line (see README,
// "Browser test"): the sandbox-exec profile (e2e.sb) allows TCP only to the two ports
// below; Chrome is started with the arguments below, so no hostname resolves and any
// connection would go to a dead proxy; the test aborts every request that is not local
// (isAllowedUrl); and the test itself tries an external fetch and expects it to be blocked.

export const WEB_PORT = 3999; // the Next dev server
export const API_PORT = 18000; // the mock API (e2e/mock-api.ts)

export const CHROME_CONTAINMENT_ARGS = [
  // No name resolves except loopback: even an unblocked request has nowhere to go.
  "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1 , EXCLUDE localhost",
  // Anything that would leave goes to a dead local port (Chrome skips the proxy for loopback).
  "--proxy-server=socks5://127.0.0.1:1",
  // Chrome's own background traffic (updates, sync, telemetry) off.
  "--disable-background-networking",
  "--disable-component-update",
  "--disable-sync",
  "--disable-default-apps",
  "--disable-breakpad",
  "--metrics-recording-only",
  "--no-first-run",
  "--no-default-browser-check",
];

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]"]);
const ALLOWED_PORTS = new Set([String(WEB_PORT), String(API_PORT)]);
// These never reach the network.
const LOCAL_SCHEMES = new Set(["data:", "about:", "blob:"]);
const NETWORK_SCHEMES = new Set(["http:", "https:", "ws:", "wss:"]);

/** True for a request the test may let through: loopback on one of the two ports, or a local scheme. */
export function isAllowedUrl(url: string): boolean {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (LOCAL_SCHEMES.has(parsed.protocol)) return true;
  if (!NETWORK_SCHEMES.has(parsed.protocol)) return false;
  if (parsed.username || parsed.password) return false;
  return LOOPBACK_HOSTS.has(parsed.hostname) && ALLOWED_PORTS.has(parsed.port);
}

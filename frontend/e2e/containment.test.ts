import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  API_PORT,
  CHROME_CONTAINMENT_ARGS,
  WEB_PORT,
  isAllowedUrl,
} from "./containment.ts";

test("loopback on the two test ports is allowed, in every scheme the app uses", () => {
  for (const url of [
    `http://127.0.0.1:${WEB_PORT}/chat`,
    `http://localhost:${WEB_PORT}/_next/static/chunks/app.js`,
    `http://127.0.0.1:${API_PORT}/chat`,
    `ws://127.0.0.1:${WEB_PORT}/_next/webpack-hmr`,
    `http://[::1]:${WEB_PORT}/`,
  ]) {
    assert.equal(isAllowedUrl(url), true, url);
  }
});

test("local schemes that never reach the network are allowed", () => {
  for (const url of [
    "about:blank",
    "data:text/plain,hello",
    "blob:http://127.0.0.1:3999/abc",
  ]) {
    assert.equal(isAllowedUrl(url), true, url);
  }
});

test("everything else is refused: other hosts, other ports, lookalikes, credentials, other schemes", () => {
  for (const url of [
    "https://blocked.invalid/x",
    "https://arxiv.org/abs/2609.00001",
    "http://127.0.0.1:8000/chat", // loopback, but not a port of the test
    "http://127.0.0.1/",
    "http://127.0.0.1.evil.example:3999/",
    "http://localhost.evil.example:3999/",
    "http://127.0.0.1:3999@evil.example/", // the host here is evil.example
    "http://user:pw@127.0.0.1:3999/",
    "http://0.0.0.0:3999/",
    "http://192.168.1.10:3999/",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "ftp://127.0.0.1:3999/",
    "not a url",
    "",
  ]) {
    assert.equal(isAllowedUrl(url), false, url);
  }
});

test("Chrome is started with the resolver rule, the dead proxy and no background traffic", () => {
  const args = CHROME_CONTAINMENT_ARGS;
  assert.ok(
    args.some(
      (a) =>
        a.startsWith("--host-resolver-rules=MAP * ~NOTFOUND") &&
        a.includes("EXCLUDE 127.0.0.1"),
    ),
  );
  assert.ok(args.includes("--proxy-server=socks5://127.0.0.1:1"));
  for (const flag of [
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
  ]) {
    assert.ok(args.includes(flag), flag);
  }
  assert.ok(
    !args.some((a) => a.includes("no-proxy") || a.includes("proxy-bypass")),
    "nothing may bypass the proxy",
  );
});

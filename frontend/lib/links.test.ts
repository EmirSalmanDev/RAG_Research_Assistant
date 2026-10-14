import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  arxivAbsUrl,
  decodeRouteParam,
  paperPath,
  safeHttpUrl,
  siteUrl,
} from "./links.ts";

test("arxivAbsUrl accepts only https://arxiv.org/abs/ pages", () => {
  assert.equal(
    arxivAbsUrl("https://arxiv.org/abs/2609.00001"),
    "https://arxiv.org/abs/2609.00001",
  );
  assert.equal(
    arxivAbsUrl("https://arxiv.org/abs/hep-th/9901001"),
    "https://arxiv.org/abs/hep-th/9901001",
  );
});

test("arxivAbsUrl refuses PDFs, other hosts, other schemes and tricks", () => {
  for (const bad of [
    "https://arxiv.org/pdf/2609.00001",
    "https://arxiv.org/pdf/2609.00001.pdf",
    "https://arxiv.org/abs/../pdf/2609.00001", // normalises to /pdf/
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/2609.00001",
    "https://arxiv.org.evil.example/abs/2609.00001",
    "https://arxiv.org@evil.example/abs/2609.00001",
    "https://user:pw@arxiv.org/abs/2609.00001",
    "javascript:alert(1)",
    "//arxiv.org/abs/2609.00001",
    "https://evil.example/https://arxiv.org/abs/2609.00001",
    "",
  ]) {
    assert.equal(arxivAbsUrl(bad), null, bad);
  }
  assert.equal(arxivAbsUrl(null), null);
  assert.equal(arxivAbsUrl(undefined), null);
});

test("safeHttpUrl allows http(s) and nothing else", () => {
  assert.equal(
    safeHttpUrl("https://github.com/example/repo"),
    "https://github.com/example/repo",
  );
  assert.equal(
    safeHttpUrl("http://example.org/page"),
    "http://example.org/page",
  );
  for (const bad of [
    "javascript:alert(1)",
    "data:text/html,x",
    "ftp://example.org",
    "not a url",
    "https://u:p@example.org",
    "",
  ]) {
    assert.equal(safeHttpUrl(bad), null, bad);
  }
  assert.equal(safeHttpUrl(null), null);
});

test("paperPath encodes the id and decodeRouteParam reverses it", () => {
  assert.equal(paperPath("2609.00001"), "/paper/2609.00001");
  assert.equal(paperPath("hep-th/9901001"), "/paper/hep-th%2F9901001");
  assert.equal(decodeRouteParam("hep-th%2F9901001"), "hep-th/9901001");
  assert.equal(decodeRouteParam("2609.00001"), "2609.00001");
  assert.equal(decodeRouteParam("100%"), "100%"); // malformed escape: kept as written
});

test("siteUrl accepts only https pages on the named site", () => {
  const gh = "https://github.com";
  const hf = "https://huggingface.co";
  assert.equal(siteUrl("https://github.com/a/b", gh), "https://github.com/a/b");
  assert.equal(
    siteUrl("https://huggingface.co/datasets/org/x", hf),
    "https://huggingface.co/datasets/org/x",
  );
  assert.equal(siteUrl("http://github.com/a/b", gh), null);
  assert.equal(siteUrl("https://github.com.evil.test/a", gh), null);
  assert.equal(siteUrl("https://user:pw@github.com/a/b", gh), null);
  assert.equal(siteUrl("https://github.com/", gh), null);
  assert.equal(siteUrl("https://huggingface.co/org/m", gh), null);
  assert.equal(siteUrl("javascript:alert(1)", gh), null);
  assert.equal(siteUrl(null, hf), null);
});

import { strict as assert } from "node:assert";
import { test } from "node:test";

import {
  MAX_UPLOAD_BYTES,
  formatBytes,
  titleFromFileName,
  uploadProblem,
} from "./upload.ts";

const pdf = (size: number, name = "paper.pdf", type = "application/pdf") => ({
  name,
  size,
  type,
});

test("sizes are shown in B, KB and MB", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(812), "812 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(200 * 1024), "200 KB");
  assert.equal(formatBytes(1.4 * 1024 * 1024), "1.4 MB");
  assert.equal(formatBytes(-1), "");
});

test("a title is suggested from the file name", () => {
  assert.equal(titleFromFileName("my_paper-v2.PDF"), "my paper v2");
  assert.equal(titleFromFileName("  a  b .pdf"), "a b");
});

test("Upload is off without a file, a PDF, a size within the limit and a title", () => {
  assert.match(uploadProblem(null, "T") ?? "", /Choose a PDF/);
  assert.match(
    uploadProblem(pdf(10, "notes.txt", "text/plain"), "T") ?? "",
    /Only PDF/,
  );
  assert.match(uploadProblem(pdf(0), "T") ?? "", /empty/);
  assert.match(
    uploadProblem(pdf(MAX_UPLOAD_BYTES + 1), "T") ?? "",
    /limit is 25\.0 MB/,
  );
  assert.match(uploadProblem(pdf(1000), "  ") ?? "", /title is required/);
  assert.match(uploadProblem(pdf(1000), "x".repeat(301)) ?? "", /300/);
  assert.equal(uploadProblem(pdf(1000), "A title"), null);
  assert.equal(uploadProblem(pdf(MAX_UPLOAD_BYTES), "A title"), null);
});

test("a file counts as a PDF by its type or its extension", () => {
  assert.equal(uploadProblem(pdf(10, "scan", "application/pdf"), "T"), null);
  assert.equal(uploadProblem(pdf(10, "Paper.PDF", ""), "T"), null);
});

test("the limit can be given (the server's setting may differ)", () => {
  assert.match(uploadProblem(pdf(2000), "T", 1000) ?? "", /limit is 1000 B/);
});

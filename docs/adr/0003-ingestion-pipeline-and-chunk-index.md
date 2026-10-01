# 0003: Ingestion pipeline and the chunk index

## Context
The assistant answers questions from arXiv papers. Every day it needs the new papers in
its categories, searchable by abstract, and the full text of the papers that matter
(library papers and papers requested by hand), split into chunks and indexed for hybrid
search (dense OpenAI embeddings + BM25 sparse vectors in Qdrant). It runs on an 8 GB
Mac, unattended at 07:00, and must be safe to re-run after any crash.

## Decision

**Source of truth.** Postgres holds papers, full texts and chunk rows; Qdrant only holds
what search needs and is derived from the `chunks` rows. Every step reads its input from
Postgres and writes its output there, so each step (and each Celery task) can be re-run
on its own. Celery chains use immutable signatures and pass arXiv ids, never data.

**Chunk ids and ords.** A chunk id is `uuid5(NAMESPACE, "<arxiv_id>v<version>#<ord>")`
with a fixed project namespace, and it is also the Qdrant point id. ord 0 is the abstract
chunk, section chunks are numbered from 1, and analysis chunks (a later milestone) start
at 10000. Re-indexing the same paper version overwrites the same ids; a new version gets
new ids. Stale points and rows (older versions, ords no longer produced) are deleted
after the new points are upserted, and only for chunk types `abstract` and `section`:
analysis chunks are never removed by ingestion. Unchanged points are not embedded again
(only rows whose embedded text changed); a change to other payload fields only updates
the payload.

**The arXiv window.** The daily run fetches papers whose latest version is dated between
the last successful run's cursor minus 4 days and now, so new versions (v2, v3) are
caught, not only new submissions (`submittedDate` is the v1 date). The overlap covers
arXiv's announcement delay and weekends; repeats are free because papers are upserted and
abstracts already indexed are skipped. arXiv's `lastUpdatedDate:[...]` query filter does
not restrict results to the requested window. Checked on 2026-10-01 with
`search_query=cat:cs.CL AND lastUpdatedDate:[202609200000 TO 202609210000]`,
`sortBy=lastUpdatedDate`, `sortOrder=descending`, `max_results=5`: arXiv answered
`totalResults=50`, but the five entries' `<updated>` values were 2026-09-30T05:42:02Z,
2026-09-27T08:13:41Z, 2026-09-23T07:05:51Z, 2026-09-22T10:17:56Z and 2026-09-20T23:47:30Z,
so four of five lie outside the one-day window (the order did follow `<updated>`). The
client therefore sorts one combined category query by `lastUpdatedDate` descending, keeps
entries whose `updated` lies in the window, and stops paging at the first older entry. Every arXiv request,
page and redirect hop goes through a Redis limiter (one request per 3 s across all
processes). A backfill uses the same query for a past window and also sets the cursor.

**Full text.** The chain is Hugging Face markdown, then arXiv HTML (LaTeXML converted to
markdown), then the PDF (PyMuPDF4LLM, parsed in memory, never stored or re-served). A
result shorter than 2,000 characters counts as missing (e.g. an abstract-only page) and
the next source is tried. Limit: a genuinely short paper (a 1-page note) is treated as
having no full text and stays abstract-only. A source error also moves on; if no source
succeeds the step fails and is retried, and only "every source missing" marks the paper
`unavailable`. After the last retry a paper is marked `failed`; a manual request resets
it. The raw text and the cleaned text are both stored (`paper_texts.raw_markdown`,
`paper_texts.markdown`), so cleaning and chunking can be redone without a download.

**Chunking.** Sections first, then windows of about 512 tokens with up to 64 tokens of
overlap on sentence boundaries. A section shorter than 64 tokens is merged into the
section right after it, once only: merges do not chain, so a chunk is never labelled with
a section far from its text. Tables, code and display math are never split below a hard
cap of 2048 tokens; a larger table is split by rows with its caption and header repeated
in every piece, code and math by lines inside their fences. The embedding text is
prefixed with `Title > Section path`.

**BM25 without a model download.** FastEmbed's `Qdrant/bm25` reads only one file, the
English stopword list. FastEmbed 0.8.1's cache check requires `mock.file` (and
`tamil.txt`), which the Hugging Face repo does not have, so it goes online on every load
and fails offline. The stopword list (Apache-2.0, from the `Qdrant/bm25` repo, source and
hash in its header) is committed in `app/ingest/data/` and passed through
`specific_model_path`: no Hub access at runtime, in Docker or in CI.

**No LlamaIndex in M2.** Points are written with qdrant-client directly (deterministic
ids, exact payload). The payload can be extended later without re-embedding, so
LlamaIndex read compatibility is decided with retrieval.

**New columns** (migration 0002, non-destructive): `papers.primary_category`,
`papers.project_url`, a CHECK on `papers.fulltext_status`, and `paper_texts.version`,
`raw_markdown` and `references`.

**Schedule.** Celery Beat starts the chain at 07:00 Europe/Istanbul. The worker runs one
process, takes one task at a time and acknowledges a task only after it finishes, so a
task killed mid-run is run again (safe, since every task is idempotent).

## Consequences
- Qdrant can always be rebuilt from Postgres; re-running any step or the whole chain
  costs no extra embeddings for unchanged papers.
- The 4-day overlap re-reads a few hundred papers a day from arXiv (metadata only).
- Beat does not catch up a run missed while the Mac sleeps; the next run's overlap
  covers the gap.
- Parsing a PDF adds about 360 MB to the worker while it runs.
- A rare native crash at interpreter exit (after all work finished) has been seen twice
  on macOS since onnxruntime and PyMuPDF were added. It is accepted and watched: one more
  occurrence, or any in CI or the worker container, replaces FastEmbed's BM25 with an own
  encoder (same tokenizer and stemmer) that does not load onnxruntime.

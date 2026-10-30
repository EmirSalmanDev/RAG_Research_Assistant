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
abstracts already indexed are skipped. The daily window never starts more than `arxiv_daily_max_window_days` (7) before
now, so a run after weeks without one fetches about a week of papers instead of tens of
thousands; the skipped days are recorded on the run (`window_capped`, `skipped_from`,
`skipped_to`, shown on the health page) and fetched only by a deliberate `make backfill`. arXiv's `lastUpdatedDate:[...]` query filter does
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

**Schedule.** Celery Beat starts the chain at 07:00 Europe/Istanbul (and again at the
catch-up times when that run did not get past arXiv, see below). The worker runs one
process, takes one task at a time and acknowledges a task only after it finishes, so a
task killed mid-run is run again (safe, since every task is idempotent).

**Retries of the two daily fetch steps** (added 2026-10-28). On 2026-10-28 arXiv failed
from the first page on for about five minutes: 20 requests, read timeouts of 20 s mixed
with fast failures, one shown as HTTP 429. The chain's tasks then retried after 0-1, 0-2 and
0-4 s (Celery's backoff with full jitter), so all four attempts were used within 4 min
46 s, and the rest of the chain was lost for that day. The two first steps of the chain,
`fetch_arxiv` and `fetch_hf_daily`, therefore wait minutes between task attempts:
- 3 retries (4 attempts), the first after 2 minutes, each next one 3 times longer: 2, 6
  and 18 minutes, each multiplied by a random factor between 0.8 and 1.2 (+/-20 %), so
  retries from several runs do not line up. Settings `DAILY_FETCH_RETRIES`,
  `DAILY_FETCH_RETRY_FIRST_DELAY_MIN`, `DAILY_FETCH_RETRY_FACTOR`, `DAILY_FETCH_RETRY_JITTER`.
- The longest possible delay (18 x 1.2 = 21.6 min) stays below the Redis broker's
  visibility timeout (set to 3600 s, kombu's default, in the Celery config). A retry
  is a message with an ETA that the worker holds unacknowledged until it is due. A
  delay longer than the visibility timeout would make Redis deliver it a second time.
  Settings that break this are refused when they are loaded.
- The ingest_runs row of a failed attempt that will be retried starts its error with
  "will retry in N min (attempt k of 4)"; the last attempt's row starts with "last attempt
  (4 of 4), no retry". The note comes first because the health page cuts errors at 300
  characters. A ValueError is not retried and gets no note.
- On the health page (`GET /status`) such a row is `retrying` when it is the latest row of
  its source and its error starts with the "will retry" note (one constant in
  `app/ingest/retry.py` writes it and reads it). The page's summary lists it as "A step is
  waiting to retry", not as a failed step. The last attempt's failure stays "failed".
- arXiv's 429 reaches us without its `Retry-After` header: the `arxiv` package's HTTPError
  keeps only the URL and the status. So the schedule above is used as it is, and the
  package is not patched.
- Unchanged: the 3-second arXiv limiter, the 5 inner HTTP tries per request with their
  1-9 s backoff, the 20 s timeout, the 7-day window cap, and "the cursor moves only on
  success". Every attempt fetches the whole window again from the first page (a failed
  attempt stores nothing). The final failure still stops the chain. The other chain
  tasks keep the seconds-scale backoff.

**Same-day catch-up of the daily chain** (added 2026-10-29). On 2026-10-28 and 2026-10-29
export.arxiv.org refused the 07:00 run (04:00 UTC) from its first request on, with HTTP 429
"Rate exceeded" and read timeouts, through all four attempts (about 29 minutes). Runs started
by hand later in the day (18:25, 08:27 and 12:23 UTC on other days) were answered. arXiv's
staff describe "Rate exceeded" as a limit of their servers' capacity that is met by trying
again later. So Beat tries the same chain again later the same day:
- Beat entries at the times in `DAILY_CATCHUP_TIMES` (default `13:00,19:00`, read in
  Europe/Istanbul like every Beat entry; empty = no catch-up). Each time is `HH:MM`, later
  than 07:00, and given once. Anything else is refused when the settings are loaded, so the
  API, the worker and Beat do not start with it.
- Each entry sends `ingest.daily_catchup`. It sends the same daily chain (the same tasks
  and the same retry rule) only if both hold:
  1. no `arxiv-daily` run finished without an error at or after the latest 07:00 Istanbul
     (for every allowed time that is today's 07:00), and
  2. no daily chain is still in progress: no unfinished `arxiv-daily` run that started
     less than 2 hours ago, and no waiting fetch_arxiv retry, i.e. the latest `arxiv-daily`
     row is not a "will retry" row that finished less than 2 hours ago. Two hours is longer
     than one attempt and longer than any retry delay (both stay below the 1-hour
     visibility timeout). An unfinished row older than that was left by a stopped worker.
  Otherwise it sends nothing and makes no external request.
- Every catch-up writes one `ingest_runs` row with source `daily-catchup` and counts
  `{"action": "started" | "skipped", "reason": ...}`. The health page shows it as its own
  step, with the action and the reason as its note. The chain it starts writes the usual
  `arxiv-daily`, `hf-daily`, ... rows, so the `arxiv-daily` line still shows the real latest
  arXiv outcome.
- The catch-up task itself is not retried. It only reads Postgres and sends one message,
  and the next entry (or the next morning) decides again.
- Beat sends a missed entry as soon as it starts (docs/adr/0013). For a catch-up that is
  safe: the same check decides, so it starts a chain only when the morning run has not
  succeeded and nothing is in progress.
- The digest: papers fetched by a catch-up reach the next day's digest at 08:00, whose
  7-day window still holds them. Today's digest is not rebuilt; it was built at 08:00 from
  the papers stored then. Their full text is fetched by the catch-up chain the same day.
- Unchanged: the 07:00 chain, the 08:00 digest, the retry rule above, the 3-second arXiv
  limiter, the inner tries and the timeout.

## Consequences
- Qdrant can always be rebuilt from Postgres; re-running any step or the whole chain
  costs no extra embeddings for unchanged papers.
- The 4-day overlap re-reads a few hundred papers a day from arXiv (metadata only).
- Beat does not catch up a run missed while the Mac sleeps; the next run's overlap
  covers the gap.
- An arXiv outage of up to about half an hour no longer costs the day. The price is
  time and requests. Waits alone add 20.8-31.2 minutes when every attempt fails. One
  failing day can send up to pages x 5 inner tries x 4 attempts arXiv requests (33 pages
  for a 7-day window: 660, all at least 3 s apart). A chain that succeeds only on its last
  retry can finish after the 08:00 digest, which is then built from the papers already
  stored.
- A day on which arXiv refuses the morning run but answers by the last catch-up is no
  longer lost; its papers reach the next morning's digest instead of today's. The price is
  requests: up to three chain attempts a day, so on a day that fails every time up to
  3 x 660 arXiv requests for a 7-day window (3 x 20 in an outage at the first page), all at
  least 3 s apart. Money is spent once: only a chain whose arXiv fetch succeeded embeds,
  and that success stops the later catch-ups.
- Parsing a PDF adds about 360 MB to the worker while it runs.
- A rare native crash at interpreter exit (after all work finished) has been seen twice
  on macOS since onnxruntime and PyMuPDF were added. It is accepted and watched: one more
  occurrence, or any in CI or the worker container, replaces FastEmbed's BM25 with an own
  encoder (same tokenizer and stemmer) that does not load onnxruntime.

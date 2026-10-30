# Technical decisions

A short record of the technical decisions behind the AI Research Copilot, in their current
state. The full reasoning for most of them is in the ADRs under `docs/adr/`; the ADR number
follows the title where one covers the decision.

## Architecture

- **Services** (Docker Compose): `postgres` 16, `qdrant` v1.19.1, `redis` 7, `api` (FastAPI), `worker` (Celery, concurrency 1) and `frontend` (Next.js), all started by `make up`; `beat` (Celery Beat) is in the `scheduler` profile, so `make up` does not start it. Published ports are bound to 127.0.0.1.
- **Ingest path**: Beat sends tasks through Redis to the worker, in Europe/Istanbul time: the daily ingest chain at 07:00, the digest at 08:00, the paper-analysis submission at 09:00 (only when `analysis_auto_submit` is on), the analysis collector every 30 minutes and a catch-up at 13:00 and 19:00. The worker reads arXiv, Hugging Face, Semantic Scholar and GitHub and writes Postgres (the source of truth) and Qdrant (derived from it).
- **Query path**: FastAPI reads Postgres and Qdrant. A chat turn (`POST /chat`) passes six stages, `condense`, `plan`, `search`, `rerank`, `generate` and `verify`, and streams the answer as server-sent events.
- **External services**: the public sources above, OpenAI (embeddings, chat, Batch API) and, optionally, Langfuse Cloud.
- **Frontend**: Next.js (App Router), started in compose by `make up` (native `pnpm dev` for development); the browser calls `/api/*` and Next rewrites it to the API.
- **Traced** (only when the Langfuse keys are set): every chat turn (trace `chat-turn`, one span per stage), compare, timeline and the literature review. Celery tasks, `GET /search`, the PDF upload and evaluation runs are not traced.

## Architecture and data

### Two paths that meet in one index (ADR 0002, ADR 0003)
Decision made: The Celery worker writes papers into Postgres and Qdrant on a schedule and FastAPI reads them to answer; the Qdrant collection schema is the contract between the two paths. The data layer and the source clients are synchronous, and only `/health` keeps its own async engine (ADR 0002).
Trade-offs / other options: Not recorded.
Why this decision was taken: Neither path waits on the other, so a slow ingest never slows a chat; ingestion runs in Celery tasks and a CLI, which are synchronous.

### Postgres is the source of truth (ADR 0003)
Decision made: Postgres holds everything, including the raw and the cleaned text of each paper and the text of every chunk; Qdrant holds only what search needs and is derived from the `chunks` rows. The schema is managed with SQLAlchemy models and Alembic migrations.
Trade-offs / other options: Not recorded.
Why this decision was taken: Cleaning and chunking can be redone without a download, and the exact cited passage can always be shown.

### Deterministic chunk ids and re-runnable steps (ADR 0003)
Decision made: A chunk id is a uuid5 of the arXiv id, version and ord, and is also the Qdrant point id (ord 0 is the abstract, sections start at 1, analysis chunks at 10000). Every ingestion step reads its input from Postgres and writes its output there, Celery chains pass arXiv ids, never data, and unchanged points are not embedded again.
Trade-offs / other options: Reference chunk ids depend on the chunking settings and the paper version, so a chunking change or a new arXiv version of a referenced paper needs new reference ids in the evaluation set.
Why this decision was taken: Each step and each Celery task can be re-run on its own; re-indexing the same version overwrites the same ids.

## Ingestion

### Daily arXiv window with an overlap and a cap (ADR 0003)
Decision made: The daily run fetches papers whose latest version is dated between the last successful run's cursor minus 4 days and now, with one combined category query sorted by `lastUpdatedDate` and the window applied in the client. The window never starts more than `arxiv_daily_max_window_days` (7) before now; skipped days are recorded on the run and fetched only by a deliberate `make backfill`.
Trade-offs / other options: The overlap re-reads metadata of papers already fetched every day, and days cut by the cap are not fetched by the daily run.
Why this decision was taken: New versions are caught as well as new submissions, the overlap covers arXiv's announcement delay and weekends, and arXiv's `lastUpdatedDate:[...]` query filter does not restrict results to the requested window.

### One rate limiter for all processes (ADR 0002)
Decision made: Rate limits are enforced for all processes by one Redis limiter (a lock with a TTL), at 1 request per 3 s for arXiv and 1 request/s for Semantic Scholar; every arXiv page and redirect hop goes through it, and the `arxiv` package runs with `num_retries=0` while the project pages manually.
Trade-offs / other options: Not recorded.
Why this decision was taken: The TTL means a crashed worker cannot block others, and manual paging makes every page one limiter slot.

### Recovering from a failed daily fetch (ADR 0003, ADR 0013)
Decision made: The `fetch_arxiv` and `fetch_hf_daily` tasks retry 3 times with delays of 2, 6 and 18 minutes, each multiplied by a random factor between 0.8 and 1.2. Beat also sends `ingest.daily_catchup` at `DAILY_CATCHUP_TIMES` (default `13:00,19:00`), which starts the daily chain only if no `arxiv-daily` run finished without an error since the latest 07:00 Istanbul and no daily chain is still in progress.
Trade-offs / other options: Waits alone add 20.8-31.2 minutes when every attempt fails, and a day that fails every time can make up to three chain attempts. arXiv's `Retry-After` header is not read, because the `arxiv` package's error keeps only the URL and the status.
Why this decision was taken: The earlier second-scale retries used all four attempts within 4 min 46 s during a five-minute arXiv failure, and arXiv's staff describe "Rate exceeded" as a limit that is met by trying again later.

### Full-text sources and the AGPL licence (ADR 0003, ADR 0004)
Decision made: Full text comes from Hugging Face markdown, then arXiv HTML converted to markdown, then the PDF via PyMuPDF4LLM, parsed in memory and never stored or re-served; a result shorter than 2,000 characters counts as missing. PyMuPDF4LLM is kept, so the project is licensed AGPL-3.0.
Trade-offs / other options: A genuinely short paper stays abstract-only. Permissive alternatives (pdfminer-based parsers) produce plain text without reliable headings or tables, and AGPL obliges anyone distributing it or offering it as a network service with changes to make the source available.
Why this decision was taken: The alternatives would weaken section-aware chunking for papers that are only available as PDF.

### Section-aware chunking (ADR 0003, ADR 0012)
Decision made: Chunking splits by sections first, then into windows of about 512 tokens with up to 64 tokens of overlap on sentence boundaries; a section shorter than 64 tokens is merged once into the next one, and tables, code and display math are never split below a hard cap of 2048 tokens. Chunks stayed at 512 tokens after the chunk-size ablation.
Trade-offs / other options: Context recall favours larger chunks by construction.
Why this decision was taken: A short section is merged once only, so a chunk is never labelled with a section far from its text.

### Enrichment is best-effort (ADR 0002, ADR 0008)
Decision made: The Hugging Face papers API replaces Papers with Code; GitHub enrichment reads only the repo already linked to a paper and fetches no README; Semantic Scholar failures return a typed `S2Unavailable` instead of raising. A rate-limit refusal stops that source for the rest of the run, and "cited by" is read from the references of papers the project holds.
Trade-offs / other options: Hub artifacts rate-limited out of a run are picked up again only by `make enrich-one`; fetching Semantic Scholar's citing papers (one extra request per paper) was rejected, so "cited by" stays short for a recent corpus.
Why this decision was taken: Papers with Code shut down in July 2025; references often point at papers the project does not hold, and `citations.citing_id` references `papers`.

### Uploaded PDFs are papers rows (ADR 0011)
Decision made: `POST /library/upload` takes the raw PDF request body, parses it in the API process one upload at a time, and stores a `papers` row with the id `local-<first 12 hex of sha256 of the PDF bytes>`; the PDF itself is never stored or served, and the same bytes give the same id.
Trade-offs / other options: A scanned PDF with no text layer is refused because there is no OCR, and PyMuPDF4LLM peaks at about 430 MiB for a long paper inside the API container.
Why this decision was taken: Parsing runs in the API because the worker was not started in the project when this was decided.

## Retrieval

### Hybrid search in one Qdrant call (ADR 0014, ADR 0005, ADR 0012)
Decision made: A search is one `query_points` call with two prefetches of 50 candidates each (dense and BM25), fused by Reciprocal Rank Fusion with k = 60, with filters applied inside each prefetch; `search_mode` can switch to `dense` or `bm25` for the evaluation. Equal scores are ordered by chunk id, also at the cut, and retrieval calls qdrant-client directly, with no LlamaIndex and no second search engine.
Trade-offs / other options: A weighted sum of normalised scores needs a weight and a normalisation tuned on the data it would be judged on.
Why this decision was taken: Some questions are about meaning and others hinge on an exact token that an embedding blurs; RRF uses ranks only, k = 60 was fixed before any measurement, and the tie order makes the same question give the same list every time.

### Vectors and the embedding model (ADR 0003, ADR 0005, ADR 0014)
Decision made: Every chunk is a Qdrant point with two named vectors, `text-dense` (OpenAI `text-embedding-3-small`, cosine) and `text-sparse` (BM25 term weights, IDF applied by Qdrant at query time). The full 1536 dimensions are kept, with int8 quantization in RAM and the original vectors on disk, and the BM25 stopword list is committed so nothing is fetched from the Hub at runtime.
Trade-offs / other options: Changing the embedding model or its dimensions means embedding everything again; the evaluation found no gain from `text-embedding-3-large` over `text-embedding-3-small`.
Why this decision was taken: The dimension count is the one choice that forces a full re-embed if changed, while quantization can be switched on or off at any time; FastEmbed's cache check would otherwise go online on every load.

### Per-paper cap, with analysis chunks counted apart (ADR 0005, ADR 0012)
Decision made: At most 3 chunks of a paper's own text are kept per paper (`max_chunks_per_paper`), and generated analysis chunks are counted apart, at most `max_analysis_chunks_per_paper` (default 1) per paper.
Trade-offs / other options: Leaving analysis chunks out altogether contradicts the design's evidence levels and hides the paper analysis from chat; a down-weighting factor would be a constant with nothing to tune it on except the golden set.
Why this decision was taken: One paper cannot crowd out the others, and model output must not push the paper's own words out of the answer's sources.

### The reranker is optional and off (ADR 0015)
Decision made: The reranker is a FastEmbed `TextCrossEncoder` (default `BAAI/bge-reranker-base`) and is off by default (`rerank_enabled = false`); v1.0 ships with reranking off. If it is turned on, the recommended model is `Xenova/ms-marco-MiniLM-L-12-v2`, and the app never downloads the model at runtime; the API loads it lazily behind a lock, and the worker never loads it.
Trade-offs / other options: Dropping the reranker altogether is the simplest option; a hosted reranking API was rejected because it would send paper text to a third party on every question.
Why this decision was taken: With `bge-reranker-base` the generated and held-out questions disagree (on the held-out golden questions it scored R@8 0.794 against hybrid's 0.912), and it is over the memory and time bar.

### Digest ranking and the nightly full-text choice (ADR 0009)
Decision made: A paper's score for an interest is `s_i = w_i * (0.8 * cos + 0.2 * kw_i) + 0.05 * category_i`, using the abstract vector read from Qdrant with no embedding request. Every new paper is indexed by its abstract, and the 07:00 chain gives full text to library and requested papers plus the top `fulltext_top_n` (20) papers of the window by relevance.
Trade-offs / other options: The constants come from the original design and were not tuned; top N is read as the N best papers of the window, so a paper that stays in the top N for a week is fetched once.
Why this decision was taken: It keeps compute and cost small, and a tie is never decided by the order the database returns rows in.

## Generation

### Six stages as separate functions (ADR 0006)
Decision made: A chat turn runs six stages (condense, plan, search, rerank, generate, verify), each a typed function in `app/rag/` with unit tests; condense, plan, rerank and verify can each be switched off by a setting, and search runs in the mode `search_mode` sets. Condense, plan and verify fall back on failure, while search, rerank and generate end the stream with an `error` event.
Trade-offs / other options: A turn costs up to four chat requests and one question embedding per search query, each visible in `llm_usage`.
Why this decision was taken: The evaluation can switch a stage off and measure what it adds; rerank cannot fall back because a configured reranker that is missing is a setup problem.

### Condense and plan (ADR 0006)
Decision made: Condense rewrites a follow-up into a standalone question from the last `chat_history_messages` (6) messages, with no running summary. Plan makes one Structured Outputs call for the intent, two rephrasings and filters, and a filter is kept only if an arXiv id is written in the question and a date is not in the future; there is no author filter.
Trade-offs / other options: The check only verifies that a category exists, so the planner can still add categories the user did not name.
Why this decision was taken: A wrong filter hides every right answer, and chunk payloads have no author field.

### Numbered citations, refusal and verification (ADR 0006)
Decision made: Generate numbers the sources [1]..[8] with their evidence level and asks for a citation after every factual sentence. With no chunks left it returns a fixed refusal without a model call, and when the sources do not answer the model replies `INSUFFICIENT_SOURCES`, which never reaches the user. Verify checks every `[n]` in code and then judges each cited sentence in one call, flagging unsupported sentences, never removing them.
Trade-offs / other options: Holding back the start of the answer delays the first visible token; verification roughly doubles the input tokens of a turn and can be switched off.
Why this decision was taken: Every claim in an answer can be checked against a paper, and a question the index cannot answer must get a refusal, not a guess.

### Prompts are versioned files; third-party text sits in delimited blocks (ADR 0006)
Decision made: Prompts are files in `app/rag/prompts/` named `<name>_v<N>.txt`, a change means a new version, and the versions a turn used go into the trace and the message meta. Sources, the question and the conversation go in the user message inside delimited blocks, every prompt says to ignore instructions inside them, and the project's own tag names are neutralised inside third-party text.
Trade-offs / other options: Changing a prompt means a new file and version, and comparisons across versions use the version recorded on each message.
Why this decision was taken: Third-party text is untrusted input and a chunk must not be able to close its block early; a stable system message lets OpenAI's prompt caching reuse it.

### Streaming, and saving a turn whole (ADR 0006, ADR 0007)
Decision made: `POST /chat` returns server-sent events through FastAPI's own `EventSourceResponse`, and the browser reads them with fetch and a small parser. A stream is retried only while it is being opened, and a turn is saved in one transaction at its end, with its metadata in one nullable jsonb column (`messages.meta`) checked by a versioned pydantic model.
Trade-offs / other options: With the jsonb column, type checking moves from the database to the application.
Why this decision was taken: A retry after tokens were shown would repeat the answer, `/chat` takes a POST body that EventSource cannot send, there is never half a turn, and the metadata shape was expected to change while the chat features were built.

### Paper analyses through the Batch API (ADR 0016, ADR 0008)
Decision made: Paper analyses use the Batch API and nothing interactive does; each analysis becomes up to four `analysis` chunks labelled as generated analysis. Creating a batch is sent once and never retried, and the daily submission is off by default (`analysis_auto_submit = false`).
Trade-offs / other options: Ordinary requests at ingestion cost twice the price, while Batch results arrive later; an unknown outcome of a batch creation needs a person to check OpenAI's batch list, a rare manual check that is cheaper than a double charge.
Why this decision was taken: Nobody waits for an analysis, a retried batch creation would bill the same papers twice, and a started Beat must not spend money unasked.

### Compare, timeline and review: code checks the citations (ADR 0010, ADR 0011)
Decision made: Comparison and timeline run only when a button is pressed and store nothing. In a comparison a cited number must belong to a source of the cell's own paper, and in a literature review a claim with no valid citation is removed and counted.
Trade-offs / other options: Pressing the button twice costs twice; code guarantees that a shown claim has a citation to a real passage, not that the passage supports it.
Why this decision was taken: Both features cost money, and the check is the guarantee behind "every cell has a working citation".

## Evaluation

### Retrieval metrics on cached question embeddings (ADR 0005)
Decision made: Retrieval metrics use binary relevance and are Recall@k, MRR over the top 10 and nDCG@k. Question embeddings are cached, `make eval-retrieval` never calls OpenAI and stops with the missing count and cost, and `EMBED=1` embeds only the missing ones.
Trade-offs / other options: A new paper version or re-chunking changes chunk ids, so reference ids can go stale, and a run stops if a reference chunk id is missing from Postgres.
Why this decision was taken: Dense and hybrid search need each question embedded by OpenAI, which costs money, so repeated runs are free.

### Answer metrics follow the RAGAS definitions, computed by the project's own judge (ADR 0012)
Decision made: Context precision, context recall, faithfulness and response relevancy follow the RAGAS definitions, but the verdicts come from the project's own Structured Outputs judge calls and the `ragas` library is not a dependency; the project adds citation accuracy and refusal accuracy (reported with the false refusal rate).
Trade-offs / other options: The judge and the generator use the same model family as the chat, which can favour answers in its own style; a dated snapshot can be pinned in `EVAL_MODEL`.
Why this decision was taken: Every model call must be a versioned prompt file, show third-party text only inside delimited blocks and be logged to `llm_usage`, while the RAGAS prompts are inline strings; refusing everything would score 1.0 on refusal accuracy alone.

### Held-out golden questions and generated questions (ADR 0012, ADR 0005)
Decision made: The 20 golden questions are AI-drafted and verified against stored chunk text, and are a held-out check never tuned against. The generated questions come from the project's own generator seeded from full-text papers, are AI-generated and AI-reviewed, and every choice between settings uses them only; papers a golden question refers to are left out of the generation pool.
Trade-offs / other options: The generated split rests on two models, so a wrong question that both let through can move a selection; RAGAS test-set generation was not used.
Why this decision was taken: A slip could otherwise turn a held-out question into a tuning question, and seeding from the project's own chunks gives reference chunk ids for free.

### A selection rule fixed before any result (ADR 0012)
Decision made: Only generated questions are used and runs are paired; the primary metric is context recall, and a challenger is better only if the paired bootstrap interval of its mean difference lies above 0 (Bonferroni-corrected), while a challenger significantly worse on faithfulness is disqualified.
Trade-offs / other options: The highest mean of a composite score was not taken because its weights would be another unvalidated constant and noise would pick the winner.
Why this decision was taken: With about 30 scored generated questions one question moves a mean by about 0.03, so "no significant difference, keep what runs" is the honest usual outcome.

### Evaluation runs the pipeline the API serves (ADR 0012)
Decision made: Each evaluation question runs through `run_chat_turn` exactly as `POST /chat` runs it; settings overrides are applied per run, read back and compared, and a mismatch stops the run. Chunking and embedding variants search their own Qdrant collections built from Postgres.
Trade-offs / other options: Calling the stages one by one in the evaluation, and re-indexing into the app's collection, were not taken.
Why this decision was taken: What is measured must be what the API serves, and an ablation must not move its own yardstick.

### Defaults unchanged after the ablations (ADR 0012)
Decision made: Query rewriting stays on, chunks stay at 512 tokens and `text-embedding-3-small` at 1536 stays; the app keeps reranking off.
Trade-offs / other options: On the held-out golden questions hybrid + `bge-reranker-base` had R@8 0.794 against hybrid's 0.912; context recall was at its ceiling (0.987) on the generated questions, so no later challenger could show a gain.
Why this decision was taken: Turning the reranker on is the owner's decision, and no challenger for rewriting, chunking or embeddings was significantly better on the primary metric.

## Safety and cost

### A budget guard for background jobs (ADR 0013)
Decision made: Background tasks that spend money check the monthly budget before any paid work and skip when it is reached; the guard is paused when a cap is stored and this month's `llm_usage` spend is at or above it. Chat, search, compare, timeline and the literature review are not paused; a banner links to the settings. This replaced the earlier rule that the cap was only stored and shown.
Trade-offs / other options: Blocking interactive requests at the cap was rejected; the check is made before a task starts, so one run can cross the cap by its own cost.
Why this decision was taken: A cap that locked the owner out of reading papers would be worse than the overspend it prevents, and the OpenAI account's own monthly limit is the hard backstop.

### Spend caps on paid evaluation runs, enforced by an egress guard (ADR 0012)
Decision made: The egress guard refuses outgoing requests before they are sent, through hooks on the HTTP client libraries, and is off unless `EGRESS_GUARD_POLICY` is set. Every paid evaluation run needs `--max-usd`; the evaluation workflow uses $1.00 for its pull-request smoke run and $3.50 by default for a manual run.
Trade-offs / other options: The bound takes every request at its output limit and every body byte as a token, so a cap is about ten times the real spend.
Why this decision was taken: A guard that counted requests afterwards let parallel requests through past the approved cap.

### Every model call is logged (ADR 0006)
Decision made: Each request writes its own committed `llm_usage` row, and a stream that breaks before its final usage chunk gets an estimated row counted with the local `cl100k_base` tokenizer.
Trade-offs / other options: The `cl100k_base` count is only an approximation for gpt-4o-mini, and the output count is a lower bound.
Why this decision was taken: A failed turn's spend is kept.

## Observability

### Tracing behind a small interface (ADR 0006, ADR 0010)
Decision made: The pipeline uses a small `Tracer` interface; Langfuse Cloud is used only when both Langfuse keys are set, and otherwise, in tests and when the SDK cannot start the tracer is a no-op. A chat turn sends exactly six stage spans in one trace named `chat-turn` with the conversation id as session, a stage that did not run is ended as skipped, and a failure inside the SDK never fails a turn.
Trade-offs / other options: Self-hosted Langfuse was not used because it needs ClickHouse and is sized for 16 GiB machines.
Why this decision was taken: Every trace has the same shape, and a private tracer provider keeps the SDK from receiving other libraries' spans.

## Infrastructure

### Beat behind a profile, one worker process (ADR 0013, ADR 0003)
Decision made: Beat is a compose profile (`scheduler`): `make up` starts the stores, the API, the worker and the frontend, and `make up-scheduler` adds Beat. The worker runs one process, takes one task at a time and acknowledges a task only after it finishes.
Trade-offs / other options: Whoever wants the nightly pipeline runs `make up-scheduler` and should know that Beat sends an overdue run on start.
Why this decision was taken: Beat found the 07:00 run overdue after restarts and sent it at once, so the worker called arXiv, Hugging Face and OpenAI without anyone asking; a task killed mid-run is run again, which is safe since every task is idempotent.

### The frontend starts with `make up` (ADR 0017, ADR 0001, ADR 0007)
Decision made: The frontend has no compose profile: `make up` builds it and starts it once the API is healthy, on 127.0.0.1:3000. Native `pnpm dev` stays for development, after `docker compose stop frontend` (both use port 3000). The browser calls `/api/*` and `next.config.ts` rewrites it to the API; the compose image sets `API_ORIGIN` as a build argument defaulting to `http://api:8000`.
Trade-offs / other options: The running stack uses about 41-44 MiB more memory and the first `make up` also builds the frontend image; a different `FRONTEND_API_ORIGIN` needs a rebuild, because setting `API_ORIGIN` at runtime was tested and did not help. Until ADR 0017 the frontend sat in a `frontend` profile and ran natively day to day, to save memory.
Why this decision was taken: The profile existed only because of memory on an 8 GB Mac; the development Mac now has 24 GB (Docker VM 7.65 GiB) and the idle stack measured about 1.5 GiB with the frontend. The same-origin proxy means the chat stream needs no CORS, and `API_ORIGIN` is read at build time.

### Pinned images and local-only ports
Decision made: The Python base image and the uv image are pinned by digest, both stages of the frontend Dockerfile pin `node:24-slim` by digest, Qdrant is pinned to v1.19.1, and every published port is bound to 127.0.0.1.
Trade-offs / other options: Not recorded.
Why this decision was taken: With the base image pinned by tag only, a moved tag or a pruned build cache turned a code-only rebuild into a full one.

## CI and release

### Tests never touch the network (ADR 0002, ADR 0007)
Decision made: Tests replay HTTP cassettes with `pytest-recording` (credentials stripped, Hugging Face full text replaced with invented text) and run under a session-wide DNS guard, and OpenAI is mocked. `make e2e` runs the Playwright browser tests against a mock API inside `sandbox-exec`.
Trade-offs / other options: The browser tests need Google Chrome installed, and `sandbox-exec` exists only on macOS (`E2E_UNSANDBOXED=1` runs them without it).
Why this decision was taken: `--block-network` covers each test only, not module-scoped fixtures, and nothing may leave the machine.

### Three workflows, and images as the deployment
Decision made: `ci.yml` runs on every push and pull request (lint, types, tests with Postgres, Qdrant and Redis service containers, a frontend build; the browser test on `main` only), `eval.yml` runs the evaluation on demand or for pull requests labelled `retrieval`, and `release.yml` builds arm64 and amd64 images on a version tag and pushes them to GitHub Container Registry.
Trade-offs / other options: The arm64 image is built under QEMU, which is slow.
Why this decision was taken: The app runs locally, so "deploy" means CI publishes versioned images.

## Frontend

### A typed API client that cannot drift (ADR 0007)
Decision made: The API schema is exported from the route definitions to `frontend/openapi.json`, turned into types with `openapi-typescript` and wrapped with `openapi-fetch`, and `make lint` fails if either committed file is stale; the `/chat` stream's event types are written by hand.
Trade-offs / other options: OpenAPI cannot describe server-sent events, so the `/chat` stream is the exception.
Why this decision was taken: The frontend needs a typed API client that cannot drift from the server.

### Citation chips and a non-modal sources panel (ADR 0007)
Decision made: A citation marker becomes a chip only when the number is one of the answer's sources, abstract chips are marked "abstract", and the sources panel is non-modal, with no backdrop and no focus trap.
Trade-offs / other options: Not recorded.
Why this decision was taken: No chip ever opens nothing, and the answer stays readable, scrollable and clickable beside the panel.

### Few dependencies, no hidden paid calls (ADR 0007, ADR 0010, ADR 0012)
Decision made: TanStack Query queries are never retried or refetched on window focus; answers are shown as pre-wrapped text with no markdown library, charts are plain SVG instead of Recharts, the app uses system fonts, and the frontend's runtime dependencies are pinned to exact versions.
Trade-offs / other options: A list or emphasis in an answer appears as the characters the model wrote.
Why this decision was taken: `/search` embeds the question (a paid OpenAI request) and must not repeat behind the user's back; rendering markdown safely is a separate decision, and a new dependency needs network approval.

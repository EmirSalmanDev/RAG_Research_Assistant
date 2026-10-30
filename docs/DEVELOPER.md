# Developer guide

This guide is for a developer who has cloned the repository. Part A shows how to run the
app. Part B shows how to see what it is doing: the health page, the logs, the queue, the
database, the budget guard, the traces and the evaluation runs. The README covers what the
app does. The design decisions are in `docs/adr/`.

Every command below is a `make` target from the `Makefile`, a plain `docker compose`
command against `docker-compose.yml`, or a read-only query. Commands that cost money or call
outside services say so.

## Part A: running the app

### Prerequisites
- **Docker Desktop** with Compose v2, and **make**. That is enough to run the backend: the
  images pin their own Python 3.12 and uv.
- For native work (migrations, tests, linting, the evaluation and the frontend dev server):
  - **uv 0.12.21**, the version the backend image and CI use. uv installs **Python 3.12**
    for you (`requires-python = "==3.12.*"` in `backend/pyproject.toml`).
  - **Node 24** with **pnpm 9.15.9**: run `corepack enable` and corepack picks the pnpm
    version from `"packageManager"` in `frontend/package.json`.
- An **OpenAI API key** for anything that embeds or answers (see the next section).

`make migrate` runs natively through uv, so even a backend-only setup needs uv once.

### The `.env` file
Copy `.env.example` to `.env` in the repository root. Docker Compose reads it, and so does
the backend when it runs natively from `backend/` (`app/config.py` looks in `.env` and
`../.env`). **Never commit `.env`.** An empty value (`S2_API_KEY=`) counts as "not set".

| Key | Needed for | Without it |
| --- | --- | --- |
| `OPENAI_API_KEY` | embeddings (ingest, `dense` and `hybrid` search, interests, PDF upload) and every model call (chat, compare, timeline, review, analyses, evaluation) | the stack starts. `POST /chat` and the compare and timeline routes answer 503 "OPENAI_API_KEY is not set (put it in .env)", and an embedding step fails with the same message |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (+ `LANGFUSE_HOST`, default `https://cloud.langfuse.com`) | traces in Langfuse (see "Langfuse traces") | tracing is a no-op and everything else works the same |
| `GITHUB_TOKEN` | GitHub repo stats in enrichment | calls go out without credentials. GitHub then allows 60 requests per hour, which is enough for `make ingest-one` |
| `S2_API_KEY` | Semantic Scholar citations | the shared public pool is used. HTTP 429 is common there: the client gives up after its retries and marks the citation data as missing, and the enrichment step stops calling that source for the rest of its run |

The store URLs in `.env.example` (`DATABASE_URL`, `QDRANT_URL`, `REDIS_URL`) point at
`localhost` for native runs. Inside compose the backend services override them with the
service names `postgres`, `qdrant` and `redis`. Every other setting in `.env.example` is
optional and shows its default.

Also set a monthly spending limit in your OpenAI account. It is the hard backstop behind the
app's own budget cap.

### Start, migrate, stop
```sh
make up        # docker compose up -d --build --wait
make migrate   # cd backend && uv run alembic upgrade head
make down      # docker compose --profile '*' down
```

- `make up` builds the backend image and starts **postgres, qdrant, redis, api and worker**.
  It waits until every healthcheck passes. It does **not** start **beat** (the scheduler) or
  the **frontend**: both are in compose profiles. The first build downloads the Python
  packages and the store images.
- Every port is bound to `127.0.0.1` only: the API on 8000, Postgres on 5432, Qdrant on
  6333, Redis on 6379 (and the compose frontend on 3000).
- `make migrate` creates or updates the schema in the Postgres from `make up`. Until it has
  run, `GET /status` shows `database: null` and the reason under `errors`.
- `make down` stops and removes the containers of every profile (beat and frontend included).
  The named volumes `postgres_data` and `qdrant_data` stay. Redis has no named volume, so
  count queued tasks as lost once its container is removed.
- **`docker compose down -v` deletes the volumes, and with them the whole corpus**: papers,
  text, chunks, the search index, your library, the usage log and the evaluation runs. Do not
  use `-v` unless that is what you want.
- `make logs` follows the logs of the `make up` services (see "Logs" in Part B).

### The scheduler (Beat)
```sh
make up-scheduler   # docker compose --profile scheduler up -d --build --wait
```
Beat sends these jobs (`app/worker/schedule.py`; times are Europe/Istanbul):

| Time | Task | What it does |
| --- | --- | --- |
| 07:00 | `ingest.daily_pipeline` | queues the daily chain: fetch_arxiv, fetch_hf_daily, index_abstracts, enrich_new, score_relevance, select_fulltext (then one full-text chain per selected paper) |
| 08:00 | `digest.build` | builds the day's digest from stored papers |
| 09:00 | `analysis.submit` | submits a Batch API job of paper analyses, but only when `ANALYSIS_AUTO_SUBMIT` is on (off by default) |
| every 30 min | `analysis.collect` | collects finished Batch jobs. It makes no API call while no batch is open |
| 13:00, 19:00 | `ingest.daily_catchup` | the same-day catch-up: it sends the daily chain again only if no `arxiv-daily` run has succeeded since 07:00 and none is running or waiting for a retry. It always writes one `daily-catchup` run, "started" or "skipped", with the reason |

The catch-up times come from `DAILY_CATCHUP_TIMES`: `HH:MM` values, comma-separated, each
later than 07:00. An empty value switches the catch-up off. Settings that break these rules
are refused when the app loads its settings, so the api, the worker and beat do not start.

**Beat's schedule file.** Beat writes the last run of every entry to its schedule file
(Celery's default `celerybeat-schedule`, in the container's working directory `/app`). The
file lives inside the beat container, not in a volume. When a **stopped** beat container is
started again, Beat finds the entries that came due while it was stopped and sends them at
once, so it can begin a daily run (arXiv, Hugging Face, OpenAI) right away. To avoid that,
**recreate** the container instead of starting the old one. A new container has no schedule
file, so it waits for the next scheduled time (Celery's default for an entry with no
recorded last run):

```sh
docker compose --profile scheduler rm -s -f beat   # remove the old container (and its schedule file)
make up-scheduler                                  # creates a fresh beat container
```

`make down` already removes the beat container. Note that `make up-scheduler` on top of an
existing, stopped beat container only starts it again (Compose recreates a container only
when its configuration or image changed). Avoid `docker compose start beat` and restarting
the container from Docker Desktop for the same reason. Read the README's "Known limitations"
before you rely on the scheduler: it says which parts of the daily pipeline have been seen
working.

### The frontend
Two ways, both on http://localhost:3000. The browser calls `/api/*` on the same origin, and
Next rewrites it to `API_ORIGIN` (`frontend/next.config.ts`), so the API needs no CORS.

- **Native (development):** `cd frontend && pnpm install && pnpm dev`. `API_ORIGIN` is read
  when the dev server starts. The default is `http://127.0.0.1:8000`, the API from `make up`.
  For another address, start it with `API_ORIGIN=http://host:port pnpm dev`.
- **In compose:** `docker compose --profile frontend up -d --wait frontend`. Next fixes
  `API_ORIGIN` **when the image is built**. The compose build uses
  `FRONTEND_API_ORIGIN` from `.env`, default `http://api:8000`. Setting it when the container
  starts has no effect. After a change, rebuild: `docker compose --profile frontend build frontend`.

### Getting data in
All of these call outside services, and most also call OpenAI, which costs money. **Set a
monthly budget cap on `/settings` first** (see "The budget guard"). They need `make up` and
`make migrate`.

- `make ingest-one ID=1706.03762 [SOURCE=hf|html|pdf] [METADATA_ONLY=1]`: one paper end to
  end, inside the worker container. It fetches metadata, the abstract and the full text
  (Hugging Face markdown, else arXiv HTML, else the PDF), then chunks and embeds them (a
  fraction of a cent). `METADATA_ONLY=1` fetches metadata and enrichment only, with no
  OpenAI call.
- `make backfill DAYS=7`: the last N days of the configured arXiv categories, abstracts only,
  inside the worker container, in the foreground. More than 7 days is refused unless you add
  `CONFIRM=1`, because a longer window costs more.
- **PDF upload** for a paper that is not on arXiv: the upload box on `/library`
  (`POST /library/upload`), or `make ingest-pdf FILE=paper.pdf TITLE="..." [ESTIMATE=1]`
  natively. It costs about $0.0002 per 10 pages of embeddings. `ESTIMATE=1` parses in memory
  and prints the cost bound with no network call. The PDF itself is never stored.
- `make enrich-one ID=...` refreshes the GitHub, Hugging Face and Semantic Scholar data of a
  paper you already hold (5 free calls, no OpenAI call).

arXiv calls are limited to one every 3 seconds across all processes (a Redis lock), and
Semantic Scholar calls to one per second.

### Tests and checks
| Command | What it runs | Needs |
| --- | --- | --- |
| `make test` | backend pytest (network blocked: only localhost is allowed, cassettes are replayed and never recorded), then frontend typecheck, eslint and unit tests | Postgres, Qdrant and Redis reachable at `DATABASE_URL`, `QDRANT_URL` and `REDIS_URL` (the `make up` ports by default). `docker compose up -d --wait postgres qdrant redis` starts just these three. The tests create and drop their own `copilot_test_<random>` database and their own Qdrant collections |
| `make lint` | ruff, ruff format, mypy, the OpenAPI staleness check, eslint, prettier | no stores |
| `make e2e` | the Playwright browser tests against a **mock API** with invented data (`frontend/e2e/mock-api.ts`), in a macOS `sandbox-exec` profile that allows only local traffic | Google Chrome installed. It makes no OpenAI call and proves nothing about the real backend |
| `make openapi` | regenerates `frontend/openapi.json` and `frontend/lib/api/schema.d.ts` from the route definitions (no network, no keys) | run it after changing an API model; `make lint` fails when either file is stale |

The chunking tests need tiktoken's `cl100k_base` file, and pytest blocks the network. CI
fetches the file once before pytest with
`uv run python -c "from app.ingest.tokens import get_encoding; print(get_encoding().name)"`
(run in `backend/`). Run the same command once on a new machine.

Evaluation commands (run natively against the stores; details in the README's "Evaluation"):
- `make eval-retrieval`: Recall@8, MRR and nDCG@10 on the golden questions for four search
  configurations. It is **free when the question embeddings are cached** in
  `backend/eval/.cache/`. Otherwise it stops and prints what embedding them would cost, and
  `EMBED=1` embeds them, which is paid. It needs the reranker model (`make fetch-reranker`
  downloads about 1 GB from Hugging Face once).
- `make eval-smoke` and `make eval` are **paid**: a chat turn plus judge calls per question.
  `ESTIMATE=1` prints the requests and the cost without sending anything. A real run needs
  `MAX_USD=...`, a hard cap enforced before every request.

### Memory on a small machine
- The worker runs with `--concurrency=1`. The frontend is left out of `make up` to save
  memory.
- The cross-encoder reranker is **off by default** (`RERANK_ENABLED=false`). The API loads
  it on the first reranked search (`GET /search?rerank=true`) and keeps it in memory. That
  can take well over 1 GB in that process (`docs/adr/0005`, `docs/adr/0015`). Load it in
  **one process only**: the retrieval evaluation runs natively and loads its own copy, so do
  not send reranked searches to the API while an evaluation runs. Restart the api container
  to free the memory.

## Part B: following what it does

### The health page and `GET /status`
`/health` in the web app reads `GET /status` every 30 seconds. `GET /health` is the plain
probe the API's Docker healthcheck uses: `{"postgres": "up"|"down", "qdrant": ..., "redis": ...}`,
with HTTP 503 when any store is down.

```sh
curl -s localhost:8000/status | python3 -m json.tool
```

`GET /status` always answers 200 and never writes or calls a paid or external API. Each part
is read on its own: a part that cannot be read is `null`, and the reason is in `errors`.

| Field | Meaning |
| --- | --- |
| `stores.postgres` / `.qdrant` / `.redis` | `"up"` or `"down"` (2 s probe each) |
| `database.papers_total` | all papers |
| `database.papers_by_fulltext_status` | papers per `fulltext_status`: `none` (abstract only), `requested`, `fetched` (text stored, not indexed yet), `indexed`, `unavailable`, `failed` |
| `database.last_runs[]` | the latest `ingest_runs` row of each source, newest first (fields below) |
| `database.budget.cap_usd` | the monthly cap from `/settings`, `null` = no cap |
| `database.budget.spent_usd` | month-to-date spend: the sum of `llm_usage.cost_usd` since the 1st of the month, 00:00 Europe/Istanbul |
| `database.budget.paused` | `true` when a cap is set and the spend has reached it |
| `database.budget.month_start` | that month start, in UTC |
| `queue_length` | tasks waiting in the Redis list `celery` (not the ones a worker already holds) |
| `qdrant_points` | points in the app's collection (`QDRANT_COLLECTION`, default `chunks`), 0 before the first ingest |
| `errors` | part name (`database`, `queue_length`, `qdrant_points`) mapped to why it could not be read (3 s timeout each) |

Each entry of `last_runs`:

| Field | Meaning |
| --- | --- |
| `source` | the step, e.g. `arxiv-daily`, `hf-daily`, `index-abstracts`, `enrich-daily`, `score-relevance`, `select-fulltext`, `fetch-fulltext`, `embed-paper`, `build-digest`, `daily-catchup`, `analysis-submit`, `ingest-paper` |
| `started_at`, `finished_at` | `finished_at` is `null` while the run is going |
| `ok` | finished and no error |
| `error` | the error, cut to 300 characters. For the two fetch steps it starts with a retry note: `will retry in N min (attempt k of 4)`, or `last attempt (4 of 4), no retry`. An error without a note is bad input (a `ValueError`), which is never retried |
| `skipped` | why the run did nothing, e.g. `budget cap reached` |
| `abandoned` | never finished and started more than 6 h ago: the worker stopped during the run. Its `error` then says `abandoned: started N h ago and never finished (the worker stopped during the run)` |
| `note` | the catch-up's decision (`started: ...` / `skipped: ...` with the reason), or the arXiv window cap: `window capped: papers updated <from> to <to> skipped (make backfill)` |
| `retrying` | this failed attempt will be retried. The step is waiting, not failed for good |

The summary at the top of `/health` (`frontend/lib/health/status.ts`) shows one sentence per
problem, most serious first:

| Sentence | Level | Meaning |
| --- | --- | --- |
| "Postgres / Qdrant / Redis is not reachable." | down | that store did not answer its probe |
| "The database numbers could not be read." | down | `database` is `null`, for example before `make migrate` |
| "The monthly budget cap is reached: background jobs that spend money are paused." | warning | see "The budget guard" |
| "A step never finished (abandoned, ...)" | warning | a run with `abandoned: true`. The worker was stopped or crashed during it |
| "A step is waiting to retry: ..." | warning | a fetch step failed and its retry is scheduled. Nothing to do yet |
| "A step failed in its last run: ..." | warning | the latest run of that source failed and will not be retried. For `arxiv-daily` it means the daily chain stopped there, and the rest of the chain did not run |
| "N papers could not get full text." | warning | papers with `fulltext_status = failed` (a per-paper task failed after its retries) |
| "<part> could not be read: <reason>" | warning | `queue_length` or `qdrant_points` failed to read |

### Logs
```sh
make logs                                                   # the make up services
docker compose logs -f api worker                           # just these two
docker compose --profile scheduler logs -f api worker beat  # with Beat
```

Celery logs `Task <name>[<task id>] received` and `Task <name>[<task id>] succeeded in <s>s: <return value>`
for every task. Most pipeline tasks return their `ingest_runs` counts, so the success line
shows what a step did. Lines from this project's code worth knowing:

| Line (from the code) | Where | Meaning |
| --- | --- | --- |
| `ingest.fetch_arxiv failed (will retry in 6 min (attempt 1 of 4)): <error>` | worker, `app/ingest/tasks.py` | a fetch step (`ingest.fetch_arxiv` or `ingest.fetch_hf_daily`) failed and is retried after the stated wait (2, 6 and 18 min ±20 % by default). Celery adds its own `Task ...[<id>] retry: ...` line |
| (no line of our own) last attempt | worker | the fourth failure is raised: Celery logs `Task ingest.fetch_arxiv[<id>] raised unexpected: <error>`, and the run's `error` in `ingest_runs` starts with `last attempt (4 of 4), no retry` |
| `task ingest.embed_paper failed for <arxiv_id> after retries: <error>` (any per-paper step) | worker | a full-text step gave up. The paper is set to `failed` |
| `full text <arxiv_id>: source <source> failed: <error>` | worker | one full-text source failed. The next one in the chain is tried |
| `enrich <arxiv_id>: step <name> failed: <error>` | worker | one enrichment call failed. The others go on |
| `enrich: <source> is rate limiting us; stopping it for this run` | worker | that source answered 429. It is not called again in this run |
| `langfuse could not start; tracing is off` | api | the Langfuse SDK failed to start. Chat works without traces |
| `langfuse: <action> failed; the chat turn goes on` | api | one tracing call failed and was dropped |
| `egress-guard: installed; ...`, `egress-guard: ALLOW #n ...`, `egress-guard: REFUSED ...` | api (stderr), only with `EGRESS_GUARD_POLICY` set | see "The egress guard" |

The catch-up and the budget guard write **no log line of their own**. Their decisions are in
`ingest_runs`, on the health page and in Celery's success line:
- `ingest.daily_catchup` returns `{'action': 'started' | 'skipped', 'reason': '...'}` (plus
  `chain_id` when it started the chain).
- A step paused by the budget returns `skipped: 'budget cap reached'`.
  `digest.score_relevance` returns `embed_skipped: 'budget cap reached'` and still scores with
  the stored vectors. `analysis.collect` returns `skipped` but writes no `ingest_runs` row,
  because it runs every 30 minutes.

### Celery and Redis (read-only)
Inside the worker container (the same app name the healthcheck uses):
```sh
docker compose exec worker celery -A app.worker.celery_app.celery_app inspect active     # running now
docker compose exec worker celery -A app.worker.celery_app.celery_app inspect reserved   # taken, not started
docker compose exec worker celery -A app.worker.celery_app.celery_app inspect scheduled  # waiting for an ETA, e.g. a fetch retry's countdown
```

In Redis (kombu's Redis transport):
```sh
docker compose exec redis redis-cli LLEN celery           # waiting in the queue (= queue_length)
docker compose exec redis redis-cli HLEN unacked          # delivered to a worker, not yet acknowledged
docker compose exec redis redis-cli ZCARD unacked_index
```

Tasks are acknowledged only when they finish (`task_acks_late`). A task the worker holds,
including a fetch retry waiting for its countdown, is an **unacked** message. If the worker
is killed, Redis delivers the message again after the visibility timeout of 3600 s, and the
task runs again (every task is idempotent). So:
- **never** run `FLUSHALL` or `FLUSHDB`, and never delete `unacked` entries blindly. You
  would drop a waiting retry or a task that is meant to run again.
- To stop work, stop the worker, and look at what it held (`HLEN unacked`) before you change
  anything.

### Postgres (read-only SQL)
Open a session in which every transaction is read-only:
```sh
docker compose exec -e PGOPTIONS=-cdefault_transaction_read_only=on postgres psql -U copilot -d copilot
```
`SHOW default_transaction_read_only;` should say `on`.

Latest run of each source (what the health page lists):
```sql
SELECT DISTINCT ON (source)
       source, id, started_at, finished_at,
       left(errors->>'error', 120) AS error,
       counts->>'skipped'          AS skipped
FROM ingest_runs
ORDER BY source, id DESC;
```

The daily arXiv fetch's recent attempts (one row per attempt, retry notes included), and the
catch-up's decisions:
```sql
SELECT id, started_at, finished_at, left(errors->>'error', 120) AS error
FROM ingest_runs
WHERE source = 'arxiv-daily'
ORDER BY id DESC
LIMIT 10;

SELECT id, started_at, counts->>'action' AS action, counts->>'reason' AS reason
FROM ingest_runs
WHERE source = 'daily-catchup'
ORDER BY id DESC
LIMIT 10;
```

Spend this month per feature, using the same month start as the budget guard (one
`llm_usage` row per OpenAI request; features include `chat-plan`, `chat-generate`,
`embed-query`, `embed-interest`, `compare-fill`, `paper-analysis-batch`):
```sql
SELECT feature, model, count(*) AS calls,
       sum(tokens_in) AS tokens_in, sum(tokens_out) AS tokens_out,
       sum(cost_usd) AS usd
FROM llm_usage
WHERE created_at >= date_trunc('month', now() AT TIME ZONE 'Europe/Istanbul')
                    AT TIME ZONE 'Europe/Istanbul'
GROUP BY feature, model
ORDER BY usd DESC;
```

Spend per month:
```sql
SELECT date_trunc('month', created_at AT TIME ZONE 'Europe/Istanbul')::date AS month,
       count(*) AS calls, sum(cost_usd) AS usd
FROM llm_usage
GROUP BY 1
ORDER BY 1;
```

Papers per full-text state:
```sql
SELECT fulltext_status, count(*) AS papers
FROM papers
GROUP BY fulltext_status
ORDER BY papers DESC;
```

Chunks per type, with the total in the last row. Compare the total with the Qdrant point
count:
```sql
SELECT chunk_type, count(*) AS chunks
FROM chunks
GROUP BY ROLLUP (chunk_type)
ORDER BY chunk_type NULLS LAST;
```
```sh
curl -s http://127.0.0.1:6333/collections/chunks   # result.points_count
```
Every chunk (`abstract`, `section`, `analysis`) is one point with the same id in the app's
collection, so the two numbers match when the index is in sync. The evaluation's chunking and
embedding variants live in their own collections and are not counted here. Postgres is the
source of truth: Qdrant can be rebuilt from it.

### The budget guard
- **Where the cap lives:** `/settings` → monthly budget cap, stored in the `app_settings` row
  `budget_cap_usd` (`GET` / `PUT /settings`). No cap stored means no guard. A cap of 0 pauses
  every guarded job.
- **What it compares:** this month's spend from `llm_usage` (since the 1st, 00:00
  Europe/Istanbul) against the cap. The check runs when a task starts.
- **What pauses** (background jobs that spend money only): `ingest.index_abstracts` (no
  abstract embeddings), `ingest.embed_paper` (the paper stays `fetched`),
  `ingest.select_fulltext` (starts no full-text chain), the interest embedding in
  `digest.score_relevance` (scoring with stored vectors goes on), `analysis.submit` and
  `analysis.collect`. Fetching papers, cleaning, chunking and the digest keep running. So do
  chat, search, compare, timeline and review, which a person starts.
- **How you see it:** a banner on every page, `budget.paused` in `GET /status`, and
  `skipped: budget cap reached` on the paused runs.
- **How to resume:** raise or remove the cap on `/settings`, or wait for the new month. No
  restart is needed. Nothing is lost: the next `index_abstracts` finds the unindexed
  abstracts, `select_fulltext` restarts the clean, chunk and embed steps for `fetched`
  papers, and open batches are collected once the guard allows it.
- **Limits:** a run can overshoot the cap by its own cost, because the check happens before
  the run. A Batch job's cost is counted when its results are collected (`docs/adr/0013`).

### Langfuse traces
Tracing is on only when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Traces
go to `LANGFUSE_HOST`. Without the keys, or when the SDK cannot start, the tracer is a no-op
and the app behaves the same. `LANGFUSE_TRACING_ENABLED=false` in the environment also
turns it off (the SDK reads that switch itself).

What is traced at this version:
- **Every chat turn** (`POST /chat`): one trace named `chat-turn` with exactly six spans,
  `condense`, `plan`, `search`, `rerank`, `generate` and `verify`. A stage that did not run
  still gets its span, marked skipped with the reason. The trace's **session is the
  conversation id**, and the trace id is stored on the assistant message
  (`messages.trace_id`). The turn's summary is on the `verify` span's metadata: the outcome,
  the **prompt versions** used (for example `condense_v1`; the prompt files are
  `app/rag/prompts/<name>_v<N>.txt`), the skipped stages and the errors.
- **Compare** (`compare`: spans `compare-retrieve`, `compare-fill`), **timeline**
  (`timeline`: `timeline-search`, `timeline-lines`) and the **literature review** (`review`:
  `review-map`, `review-write`). These have no session.

**Not traced:** the Celery tasks (the daily ingest chain, full-text chains, digest build,
Batch analyses), `GET /search`, the PDF upload and the evaluation runs (they use a no-op
tracer on purpose). For ingest, use `ingest_runs`, the worker log and `/health` instead.

Values are sanitized before they are sent (no keys, tokens or headers, no data URIs, no
huge strings).

To find the trace of a chat answer:
```sql
SELECT id, conversation_id, trace_id, created_at
FROM messages
WHERE role = 'assistant' AND trace_id IS NOT NULL
ORDER BY id DESC
LIMIT 5;
```

### The `/evals` dashboard and `eval_runs`
Every evaluation run is a row in `eval_runs`: `name`, `config` (the settings the run used),
`git_sha` (the commit), `metrics` (per split, `all`, `heldout` and `generated`, for answer
runs, or flat for retrieval runs) and `created_at`. Each question's result is a row in
`eval_results`: the retrieved chunk ids, the answer and that question's scores.

`/evals` reads them through the read-only `GET /evals/runs`, `/evals/runs/{run_id}`,
`/evals/runs/{run_id}/results` and `/evals/runs/{run_id}/results/{item_id}`. These use
Postgres only, with no model call and no cost. It shows metric trends across runs. **To
compare two runs**, pick run A and run B: the page shows their headline metrics with the
difference B − A, and their questions joined by question, for all, held-out (golden) or
generated questions. Opening a question shows the retrieved chunks, the answer and the
judge's verdicts. In SQL:

```sql
SELECT id, name, left(git_sha, 12) AS git_sha, created_at
FROM eval_runs
ORDER BY id DESC
LIMIT 20;

-- the two newest runs' configuration and headline metrics
SELECT id, name, config, coalesce(metrics->'all', metrics) AS headline
FROM eval_runs
WHERE id IN (SELECT id FROM eval_runs ORDER BY id DESC LIMIT 2)
ORDER BY id;
```

How to read the numbers (`docs/adr/0012`, README "Evaluation"):
- Faithfulness, response relevancy, context precision and context recall **follow the RAGAS
  definitions, but they are computed by this project's own judge calls** (`eval/judge.py`,
  `EVAL_MODEL` at temperature 0), not by the `ragas` library.
- The generated questions in `backend/eval/generated.jsonl` are **AI-generated and
  AI-reviewed** (a Claude Code subagent reviewed them, not a person). The golden questions
  were AI-drafted and checked against the stored chunk text. Neither set was reviewed by a
  domain expert.
- The project makes **no claim that the reranker helps**. On the generated questions it
  raised context recall. On the golden questions the `bge` reranker's Recall@8 was lower than
  without a reranker. It stays off by default.
- Differences of about 0.04 (generated) or 0.06 (golden) are within the noise of one
  question.

### The egress guard (`EGRESS_GUARD_POLICY`)
`app/egress_guard.py` is a tool for **checks and proof runs**, not a production feature. It
is off unless `EGRESS_GUARD_POLICY` holds a JSON policy. The policy sets the approved hosts
with a request cap each, a total cap, the internal hosts that are not counted, and a spend
cap on a worst-case bound for OpenAI requests. A request that would break any of these is
refused before it is sent, and every later external request is refused too (a hard stop).
Every decision is logged to stderr as an `egress-guard:` line (host, path, counts, spend
bound). Bodies and headers are never logged.

Only the API process reads `EGRESS_GUARD_POLICY` (`app/main.py` installs the guard when it
starts). The Celery worker and Beat do not. The paid evaluation commands install the same
guard in their own process from `MAX_USD` (`eval/budget.py`). The policy format is in the
module's docstring.

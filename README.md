# AI Research Copilot

A personal, local-first research assistant for AI papers. It ingests new arXiv papers
every morning, indexes their abstracts (and the full text of the papers you care about)
for hybrid search, and will answer questions with citations.

Status: ingestion (M2) and hybrid retrieval with its evaluation (M3) are in place. Chat
with citations (M4) is being built: the pipeline and `POST /chat` exist; tracing to
Langfuse is not connected yet.

## Stack
- Backend: Python 3.12, FastAPI, SQLAlchemy 2 + Alembic, Celery + Redis (Beat schedules)
- Search: Qdrant with two named vectors per chunk: `text-dense` (OpenAI
  `text-embedding-3-small`, 1536 dims, cosine, int8 quantization) and `text-sparse`
  (BM25 via FastEmbed, IDF applied by Qdrant)
- Frontend: Next.js, TypeScript, Tailwind
- Everything runs locally with Docker Compose (designed for an 8 GB Apple Silicon Mac)

## Setup
1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (set a monthly spending limit
   in your OpenAI account). Other settings have defaults, listed in `.env.example`.
2. `make up` starts Postgres, Qdrant, Redis, the API, the Celery worker and Beat.
3. `make migrate` creates the database schema.
4. The frontend runs natively: `cd frontend && pnpm dev` (see "Frontend").

## Commands
| Command | What it does |
| --- | --- |
| `make up` / `make down` / `make logs` | start, stop, tail the stack |
| `make migrate` | apply database migrations |
| `make test` / `make lint` | backend tests (offline) and frontend checks and unit tests / linters |
| `make e2e` | the browser tests: Playwright tests of the chat, digest, paper, compare, timeline and library pages (see "Browser test"); not part of `make test` |
| `make ingest-one ID=<arxiv_id> [SOURCE=hf\|html\|pdf] [METADATA_ONLY=1]` | ingest one paper end to end: metadata, abstract, full text, chunks, embeddings |
| `make enrich-one ID=<arxiv_id>` | refresh a held paper's GitHub repo, Hugging Face models, datasets and Spaces, and Semantic Scholar citations (5 free calls, no OpenAI call) |
| `make analyze-submit [DRY_RUN=1] [IDS="<id> ..."] [MAX=N]` | submit one OpenAI Batch job of paper analyses for full-text papers; `DRY_RUN=1` only prints the papers and the cost bound |
| `make analyze-collect` | store and index the analyses of finished Batch jobs |
| `make digest [ARGS="--dry-run"]` | build today's digest by hand: embed new or edited interests (one small embeddings request), score the last week's papers, write the digest and list which papers the nightly selection would send to full text; `--dry-run` writes nothing and makes no request, `--date`, `--rebuild`, `--no-embed` also work |
| `make compare IDS="<id> <id> ..." [DIMENSIONS=...] [ESTIMATE=1]` | compare 2 to 5 papers (paid: one embeddings and one chat request); `ESTIMATE=1` prints an upper bound of the tokens and cost from the stored chunk sizes, no network, no key |
| `make timeline TOPIC="..." [FROM=...] [TO=...] [ESTIMATE=1]` | a topic timeline (paid: one embeddings and one chat request); `ESTIMATE=1` as above |
| `make review COLLECTION="..." [FOCUS="..."] [MAX=30] [ESTIMATE=1\|BIB=1]` | a related-work draft for a library collection (paid: up to one chat call per eight papers, plus one); `ESTIMATE=1` prints an upper bound of the tokens and cost from the stored text, no network, no key; `BIB=1` prints the collection's `.bib` file |
| `make ingest-pdf FILE=paper.pdf TITLE="..." [YEAR=...] [COLLECTION=...] [ESTIMATE=1]` | add a PDF that is not on arXiv through the same ingestion steps (paid: embeddings, about $0.0002 per 10 pages); `ESTIMATE=1` parses and chunks in memory and prints the cost bound, no network, no key |
| `make backfill DAYS=7 [CONFIRM=1]` | fetch the last N days of the configured categories and index abstracts only; more than 7 days needs `CONFIRM=1` |
| `make fetch-reranker` | download the reranker model once (about 1 GB) into `~/.cache/fastembed` |
| `make eval-retrieval [EMBED=1]` | retrieval metrics on the golden set for four configurations, stored in `eval_runs` |

`make ingest-one`, `make enrich-one` and `make backfill` run inside the worker container
and make real network calls (`ingest-one` and `backfill` also small embedding calls). `make fetch-reranker` downloads from Hugging Face.
`make eval-retrieval` makes no paid calls: it stops if a golden question has no cached
embedding and prints what embedding it would cost; `EMBED=1` embeds those questions.

## Search
`GET /search?q=...` returns the best chunks for a question, each with its arXiv abstract
link. Optional parameters: `limit` (1-50, default 10), `category` and `arxiv_id`
(repeatable), `published_from` and `published_to` (dates, inclusive), and `rerank`
(`true` to rerank; off by default). Three stages run in order, each with its own setting
(see `.env.example` and `docs/adr/0005`):

1. **search**: one Qdrant query in `dense`, `bm25` or `hybrid` mode. Hybrid takes 50
   candidates from each vector and fuses them with Reciprocal Rank Fusion; filters run
   inside Qdrant.
2. **rerank** (only with `rerank=true`, or `RERANK_ENABLED=true`): a cross-encoder
   (`BAAI/bge-reranker-base`) rescores the candidates. On a laptop CPU this takes about
   20 s for 50 candidates, which is why it is off by default. The model is only read from
   the local cache (`make fetch-reranker`), never downloaded by the app.
3. **cap**: at most 3 chunks per paper.

The dense and hybrid modes embed the question with OpenAI (one small paid request per
search, logged in `llm_usage`).

## Chat
`POST /chat` answers a question from the indexed papers. The answer streams as
server-sent events, and every factual sentence cites its source as `[n]`. Run
`make migrate` first (the chat stores extra data on each message).

```sh
curl -N -X POST localhost:8000/chat -H 'content-type: application/json' \
     -d '{"message": "How does LoRA reduce the number of trained parameters?"}'
```

A follow-up sends `"conversation_id"` from the first event. The events, in order:
`meta` (conversation id, intent, the standalone question and the search plan),
`sources` (the numbered chunks with their arXiv links), `token` (pieces of the answer)
or `refusal`, `verification` (the citation checks), and finally `done` (message id,
cited sources, tokens and cost) or `error`.

Six stages, each a function in `backend/app/rag/` with its own setting (see
`.env.example` and `docs/adr/0006`):

1. **condense**: a follow-up ("and its limitations?") becomes a standalone question,
   using the last 6 messages.
2. **plan**: one structured call returns the intent, two rephrasings and filters
   (arXiv ids, categories, dates). The code keeps only filters it can verify, for example
   an arXiv id that is actually written in the question. There is no author filter.
3. **search**: the question and the rephrasings, merged with Reciprocal Rank Fusion.
4. **rerank**: optional and off by default, as for `GET /search`.
5. **generate**: the answer, from at most 8 numbered sources. Each source shows its
   evidence level (`section`, `abstract` or `analysis`). An answer that rests only on
   abstracts says so, and the `done` event carries `abstract_only`.
6. **verify**: every `[n]` must exist, and a second call checks each cited sentence
   against its sources. Unsupported sentences are flagged, not removed.

If nothing in the index matches, or the sources do not answer the question, the reply is
a fixed refusal instead of a guess. A turn makes up to four chat requests (`gpt-4o-mini`)
and up to three small embedding requests; each one is logged in `llm_usage`.

## Frontend
The web app is Next.js (App Router) and runs natively: `cd frontend && pnpm dev`, then
open http://localhost:3000. It talks to the API through a same-origin proxy (`/api/*` is
rewritten to `API_ORIGIN`, default `http://127.0.0.1:8000`), so the API needs no CORS
setup. Pages:

- `/`: the daily digest. The day's papers ranked against your interests, with a trending
  badge, the best-matching interest, a one-line summary, links to the paper page and to
  arXiv, and Save and Dismiss (with Undo). Opening it asks the API for stored rows only; it
  never builds a digest or calls a model. `?date=YYYY-MM-DD` shows an earlier digest.
- `/settings`: your interests (name, description, keywords, categories, active), the digest
  size, how many top papers get full text each night, the arXiv categories to fetch (adding
  one does not backfill older papers) and a monthly budget cap, shown next to this month's
  spending (the cap pauses nothing yet). Saving costs nothing; a new or edited interest is
  embedded by the next nightly run.
- `/chat`: ask a question; the answer streams in, and every `[n]` is a chip. A chip opens
  the cited passage in a side panel (it stays open beside the chat; Escape or the close
  button closes it) with a link to the paper's arXiv abstract page. A dashed chip marked
  "abstract" means the source is only the paper's abstract.
- `/search`: find passages by topic, category and date, grouped by paper. Each search
  costs one embedding request.
- `/paper/[id]`: one paper's metadata, abstract and indexed passages (no model call).
- `/compare`: pick 2 to 5 papers (a title or arXiv id lookup and a list of the papers with
  full text; free) and the dimensions (task, method, data, metrics, compute, limitations).
  **Compare** makes one embeddings request and one model call (a fraction of a cent) and
  fills a table. A cell shows a claim only together with citations that open the passage it
  rests on; a cell the sources do not support says "Not stated in the sources." A paper with
  only its abstract indexed is marked "abstract only" and will often have such cells. The
  table exports as markdown (download or copy) with `[n]` markers and a Sources list that
  links the arXiv abstract pages.
- `/timeline`: papers on a topic by date (plain SVG), key papers marked, one line each on
  what the paper changed, written from its abstract. **Run** makes one embeddings request and
  one model call. Citation counts are thin (Semantic Scholar covers few of the stored
  papers) and the stored papers cover a short time for most topics; the page says so.

- `/library`: your collections of papers (create, rename, delete), each paper with a reading
  status (to read, reading, read, archived) and a note, a `.bib` download built from stored
  metadata (no model), an add-by-id box (only papers already in the database), a PDF upload
  for papers that are not on arXiv, and the literature review builder: **Generate review**
  writes a related-work draft from the collection's generated analyses and abstracts
  (one model call per eight papers above eight, plus one writing call). Every sentence about a
  paper carries a citation chip, labelled "analysis" or "abstract", that opens the cited text;
  a sentence without a valid citation is removed and counted. The draft rests on generated
  analyses and abstracts, not on full text, and a citation that opens does not prove it
  supports the sentence. The References list shows the BibTeX keys of the `.bib` file. The
  draft exports as markdown. Upload and Generate review are the only buttons that cost money.
  Decisions: `docs/adr/0011`.

The TypeScript client is generated from the API's OpenAPI schema: `make openapi`
rewrites `frontend/openapi.json` and `frontend/lib/api/schema.d.ts`, and `make lint`
fails if they are stale. The chat stream (server-sent events) cannot be described in
OpenAPI, so its event types are written by hand in `frontend/lib/chat/events.ts`; the
backend test `tests/unit/test_sse_fixtures.py` writes real streams from the real route
into `frontend/lib/chat/testdata/`, and the frontend tests replay them.
Answers are shown as pre-wrapped text, not rendered markdown, so a `**bold**` or a numbered
list in an answer appears as the characters the model wrote. Plan filters (category, dates)
are shown above each answer. Decisions: `docs/adr/0007`.

To look at the real data in a browser, start the stack (`make up`), then `pnpm dev` in
`frontend/` (or `pnpm build && pnpm start`). The chat page asks the real model: each turn
makes about six small OpenAI requests (about $0.001), and each search one embedding request.

### Browser test
`make e2e` runs the Playwright tests. `frontend/e2e/chat-citation.spec.ts`: ask a question,
watch the answer stream in, open a citation, read the cited passage.
`frontend/e2e/digest-flow.spec.ts`: open the digest, save one card, dismiss one, undo, then
add, edit and delete an interest on `/settings` and save the settings.
`frontend/e2e/paper-page.spec.ts`: a paper page shows its analysis, code, Hugging Face
models and datasets, and the three related-paper tabs (keyboard included).
`frontend/e2e/explore-flow.spec.ts`: pick three papers, Compare, open a citation, export the
markdown; run a timeline (and that typing on either page sends nothing paid).
`frontend/e2e/library-flow.spec.ts`: create a collection, add papers, change a status, edit a
note, download the `.bib`, generate a review, open a chip, export the markdown; upload a PDF
(one request, a repeat changes nothing, a 12 MB body reaches the API whole through the proxy).
It starts the Next
dev server (with `--webpack`) and a small **mock API** with invented data
(`frontend/e2e/mock-api.ts`), and drives the Google Chrome that is already installed
(Playwright `channel: "chrome"`, so no browser is downloaded). The mock API is reached
through the real `/api` rewrite, so the test proves the UI and that the rewrite passes the
stream through unbuffered. It proves nothing about the real backend. It makes no OpenAI
call and costs nothing.

Nothing can leave the machine, by four layers: (1) on macOS everything runs inside
`sandbox-exec` with the profile `frontend/e2e/e2e.sb`, which allows TCP only to
`localhost:3999` (Next) and `localhost:18000` (the mock API); (2) Chrome starts with a
resolver rule under which no name except loopback resolves, and a dead proxy; (3) the test
aborts every request that is not local; (4) the test makes one external request on purpose
(to a `.invalid` name) and expects it to be blocked. The test never opens or clicks an
external link: the arXiv link is checked by its `href` only. Chrome runs with
`--no-sandbox` because Chrome's own sandbox cannot start inside `sandbox-exec`; the profile
is what contains it, and the page loads only the local app.

Without `sandbox-exec` (Linux, for instance) `make e2e` refuses to run. `E2E_UNSANDBOXED=1
make e2e` runs it without the sandbox profile; layers 2 to 4 still apply, but layer 1 does
not, so use it only on a machine where you are comfortable with that. It needs Google
Chrome installed. Playwright is not run in CI yet.

## The daily pipeline
Celery Beat starts it every day at **07:00 Europe/Istanbul**:

1. **fetch_arxiv**: new papers and new versions in the configured categories (the list on
   `/settings` replaces the configured one), from the last run's cursor minus 4 days (arXiv
   announces papers with a delay).
2. **fetch_hf_daily**: Hugging Face daily papers of yesterday and today (upvotes, GitHub
   repo, project page); trending papers we do not hold are added.
3. **index_abstracts**: every paper's title and abstract become its first chunk.
4. **enrich_new**: GitHub repo, Hugging Face models and datasets, and Semantic Scholar
   citations for the new papers.
5. **score_relevance**: embeds interests that are new or edited (the only paid call, one
   small embeddings request) and stores each recent paper's relevance to your interests.
6. **select_fulltext**: papers in the library, requested by hand, or among the top 20 by
   relevance get their full text (Hugging Face markdown, else arXiv HTML, else the PDF),
   which is cleaned, chunked (about 512 tokens, by section) and indexed.

At **08:00** a separate Beat entry builds the day's digest (`digest.build`) from the papers
of the last 7 days that were not dismissed and are not in an earlier digest. Beat does not
catch up a missed run; `make digest` builds one by hand. Decisions: `docs/adr/0009`.

Every step is a separate, idempotent Celery task, retried with backoff; each run is
recorded in the `ingest_runs` table. A missed run (computer asleep) is not caught up; the
next run's 4-day overlap covers it.

Postgres is the source of truth; the Qdrant collection is derived from it and can be
rebuilt. Design decisions are recorded in `docs/adr/`.

## Licence
The project will be released under AGPL-3.0 (see `docs/adr/0004-licence-agpl.md`),
because the PDF parser it uses (PyMuPDF4LLM) is AGPL-licensed. arXiv PDFs are only parsed
in memory and never stored or re-served; the app links to arXiv abstract pages.

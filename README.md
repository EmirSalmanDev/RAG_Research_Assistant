# AI Research Copilot

A personal, local-first research assistant for AI papers. It ingests new arXiv papers
every morning, indexes their abstracts (and the full text of the papers you care about)
for hybrid search, and will answer questions with citations.

Status: ingestion (milestone M2) is in place; hybrid retrieval and its evaluation (M3)
are being built; chat with citations follows.

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
4. The frontend runs natively: `cd frontend && pnpm dev`.

## Commands
| Command | What it does |
| --- | --- |
| `make up` / `make down` / `make logs` | start, stop, tail the stack |
| `make migrate` | apply database migrations |
| `make test` / `make lint` | backend tests (offline) and frontend checks / linters |
| `make ingest-one ID=<arxiv_id> [SOURCE=hf\|html\|pdf] [METADATA_ONLY=1]` | ingest one paper end to end: metadata, abstract, full text, chunks, embeddings |
| `make backfill DAYS=7 [CONFIRM=1]` | fetch the last N days of the configured categories and index abstracts only; more than 7 days needs `CONFIRM=1` |
| `make fetch-reranker` | download the reranker model once (about 1 GB) into `~/.cache/fastembed` |
| `make eval-retrieval [EMBED=1]` | retrieval metrics on the golden set for four configurations, stored in `eval_runs` |

`make ingest-one` and `make backfill` run inside the worker container and make real
network and (small) embedding calls. `make fetch-reranker` downloads from Hugging Face.
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

## The daily pipeline
Celery Beat starts it every day at **07:00 Europe/Istanbul**:

1. **fetch_arxiv**: new papers and new versions in the configured categories, from the
   last run's cursor minus 4 days (arXiv announces papers with a delay).
2. **fetch_hf_daily**: Hugging Face daily papers of yesterday and today (upvotes, GitHub
   repo, project page); trending papers we do not hold are added.
3. **index_abstracts**: every paper's title and abstract become its first chunk.
4. **select_fulltext**: papers in the library or requested by hand get their full text
   (Hugging Face markdown, else arXiv HTML, else the PDF), which is cleaned, chunked
   (about 512 tokens, by section) and indexed.

Every step is a separate, idempotent Celery task, retried with backoff; each run is
recorded in the `ingest_runs` table. A missed run (computer asleep) is not caught up; the
next run's 4-day overlap covers it.

Postgres is the source of truth; the Qdrant collection is derived from it and can be
rebuilt. Design decisions are recorded in `docs/adr/`.

## Licence
The project will be released under AGPL-3.0 (see `docs/adr/0004-licence-agpl.md`),
because the PDF parser it uses (PyMuPDF4LLM) is AGPL-licensed. arXiv PDFs are only parsed
in memory and never stored or re-served; the app links to arXiv abstract pages.

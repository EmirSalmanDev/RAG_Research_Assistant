.PHONY: up down logs test lint openapi migrate ingest-one backfill fetch-reranker eval-retrieval

# Start the stack and wait until every service's healthcheck passes.
up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

# Backend tests are offline (stores mocked); frontend runs typecheck + eslint.
# PYTHONFAULTHANDLER: if a native library crashes the process (even at exit, after the
# tests passed), Python prints the stacks, so the culprit can be identified.
test:
	cd backend && PYTHONFAULTHANDLER=1 uv run pytest
	cd frontend && pnpm typecheck && pnpm lint

lint:
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app eval tests
	cd backend && uv run python -m app.export_openapi ../frontend/openapi.json --check
	cd frontend && pnpm gen:api:check && pnpm lint && pnpm format:check

# Regenerate the API schema (frontend/openapi.json) and the TypeScript client types built
# from it (frontend/lib/api/schema.d.ts). Built from the route definitions only: no api
# container, no network, no keys. `make lint` fails when either committed file is stale.
openapi:
	cd backend && uv run python -m app.export_openapi ../frontend/openapi.json
	cd frontend && pnpm gen:api

# Apply Alembic migrations to the Postgres from `make up` (published on localhost:5432).
migrate:
	cd backend && uv run alembic upgrade head

# Ingest one paper end to end (metadata, abstract, full text, chunks, embeddings):
#   make ingest-one ID=1706.03762 [SOURCE=hf|html|pdf] [METADATA_ONLY=1]
# SOURCE forces one full-text source instead of the fallback chain (HF .md, arXiv HTML,
# PDF); METADATA_ONLY=1 only fetches metadata and enrichment (no OpenAI calls).
# Runs in the worker container (same image, deps and model files as the scheduled jobs).
# Needs `make up` and `make migrate` first. Makes real network calls and costs a little.
ingest-one:
	@test -n "$(ID)" || (echo "usage: make ingest-one ID=<arxiv_id> [SOURCE=hf|html|pdf] [METADATA_ONLY=1]"; exit 2)
	docker compose exec -T worker python -m app.ingest.cli one "$(ID)" \
		$(if $(SOURCE),--force-source "$(SOURCE)") $(if $(METADATA_ONLY),--metadata-only)

# Seed the corpus: fetch the last DAYS days of the configured arXiv categories (new
# papers and new versions) and index their abstracts only (no full text):
#   make backfill DAYS=7
# More than 7 days needs CONFIRM=1 (costs more; ask first). Runs in the worker container
# in the foreground; makes real network calls and embedding requests.
backfill:
	@test -n "$(DAYS)" || (echo "usage: make backfill DAYS=7 [CONFIRM=1]"; exit 2)
	docker compose exec -T worker python -m app.ingest.cli backfill --days "$(DAYS)" \
		$(if $(CONFIRM),--confirm)

# Download the reranker model (RERANKER_MODEL, default BAAI/bge-reranker-base, about 1 GB)
# once into FASTEMBED_CACHE_DIR (default ~/.cache/fastembed). Makes real network calls to
# Hugging Face: ask first. Runs natively; the API container mounts that folder read-only.
# Nothing else ever downloads it: the app loads the model from the cache only.
fetch-reranker:
	cd backend && uv run python -m app.rag.rerank fetch

# Retrieval metrics (Recall@8, MRR, nDCG@10) on backend/eval/golden.jsonl for dense, BM25,
# hybrid and hybrid + rerank; each run is stored in eval_runs. Runs natively against the
# stores (needs the reranker from `make fetch-reranker`). No LLM calls, no cost: question
# embeddings come from a local cache (backend/eval/.cache/), and a run with uncached
# questions stops and prints what embedding them would cost. EMBED=1 embeds just those
# (paid OpenAI requests: ask first).
eval-retrieval:
	cd backend && uv run python -m eval.run $(if $(EMBED),--embed)

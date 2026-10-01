.PHONY: up down logs test lint migrate ingest-one

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
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app tests
	cd frontend && pnpm lint && pnpm format:check

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

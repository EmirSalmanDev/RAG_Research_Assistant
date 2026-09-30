.PHONY: up down logs test lint migrate ingest-one

# Start the stack and wait until every service's healthcheck passes.
up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

# Backend tests are offline (stores mocked); frontend runs typecheck + eslint.
test:
	cd backend && uv run pytest
	cd frontend && pnpm typecheck && pnpm lint

lint:
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy app tests
	cd frontend && pnpm lint && pnpm format:check

# Apply Alembic migrations to the Postgres from `make up` (published on localhost:5432).
migrate:
	cd backend && uv run alembic upgrade head

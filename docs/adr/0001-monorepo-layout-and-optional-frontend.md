# 0001: Monorepo layout; frontend optional in Compose

## Context
The project is a local-first RAG assistant run on an 8 GB Apple Silicon Mac. The stack needs
Postgres, Qdrant, Redis, an API, a Celery worker, Celery Beat and a Next.js frontend.
Running all of them in containers plus a dev server is tight on memory.

## Decision
- One repo with `backend/` (uv, Python 3.12) and `frontend/` (Next.js, pnpm 9, pinned through
  `packageManager`). The repo root holds compose, Makefile, CI and docs.
- `make up` starts the stores, api, worker and beat, and waits for every healthcheck.
  The frontend is in the Compose profile `frontend`; day to day it runs natively with `pnpm dev`.
  Superseded in part by ADR 0017: the frontend has no profile and `make up` starts it.
- The worker runs with concurrency 1. Celery healthchecks poll every 30 s.
- Qdrant's image has no curl, so its healthcheck uses bash `/dev/tcp` against `/readyz`.

## Consequences
- Lower memory use; the frontend needs a separate command (`cd frontend && pnpm dev`).
- Stores are reachable from the host on 5432, 6333 and 6379, so native backend runs work.
- pnpm 12 was broken with the local corepack, so pnpm 9.15.9 is pinned. Revisit when fixed.

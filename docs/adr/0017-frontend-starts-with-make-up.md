# 0017: The frontend starts with `make up`

## Context
ADR 0001 put the frontend in the compose profile `frontend`, so `make up` did not start it,
and day to day it ran natively with `pnpm dev`. The only reason was memory: the project was
developed on an 8 GB Mac, where all containers plus a dev server were tight. Since then the
development Mac has 24 GB of RAM, and Docker's VM gets 7.65 GiB.

Measured on 2026-10-31 with `docker stats --no-stream`, idle, about a minute after `make up`
returned, in a disposable compose project with empty stores and images built from the same
Dockerfiles:

| Service | Memory |
| --- | --- |
| frontend (Next.js standalone server) | 40.8 MiB idle; 43.5 MiB after five page loads |
| api | 160.9 MiB (reranker not loaded) |
| worker | 165.8 MiB |
| postgres / qdrant / redis, empty | 24.8 / 276.8 / 10.4 MiB |

The real stores, which hold the corpus, read 137.6 MiB (postgres), 1.02 GiB (qdrant) and
23.9 MiB (redis) at the same time. The whole idle stack, the frontend included, is therefore
about 1.5 GiB of the 7.65 GiB VM. The reranker adds well over 1 GB to the api process once it
is loaded (ADR 0015).

## Decision
- The frontend service has no profile. `make up` (`docker compose up -d --build --wait`)
  builds and starts it with postgres, qdrant, redis, the api and the worker, and waits for
  its healthcheck (a `node -e` fetch of port 3000).
- It depends on the api with `condition: service_healthy`: every page reads from the API,
  so it starts once the API answers `/health`.
- Its port stays bound to 127.0.0.1 (3000), like every other published port.
- Native `pnpm dev` stays the way to develop the frontend. Both use port 3000, so stop the
  compose frontend first (`docker compose stop frontend`), or start only the backend
  services.
- Unchanged: Beat stays alone in the `scheduler` profile (ADR 0013), the worker keeps
  concurrency 1, and the image still fixes the `/api` proxy target when it is built
  (`FRONTEND_API_ORIGIN`, default `http://api:8000`; ADR 0007).

This supersedes the part of ADR 0001 that puts the frontend in a profile and runs it
natively day to day.

## Consequences
- One command starts the whole app; http://localhost:3000 works after `make up` without Node
  on the host. Node and pnpm are still needed for `make test`, `make lint` and frontend
  development.
- The first `make up` also builds the frontend image: the Node base image and the frontend
  packages are downloaded once, and later builds come from the build cache.
- About 41-44 MiB more memory while the stack runs, by the measurement above.
- Running `pnpm dev` now needs the compose frontend stopped first, and the next `make up`
  starts it again.
- A different `FRONTEND_API_ORIGIN` still needs a rebuild (`docker compose build frontend`,
  or `make up`, which builds); setting it when the container starts has no effect.
- On a smaller machine, `docker compose stop frontend` frees its memory while the web app is
  not needed.
- `make up` builds on every run but recreates a container only when its image or its
  configuration changed. At first every `make up` recreated the api, the worker and the
  frontend even with nothing changed: BuildKit's default provenance attestation records the
  build's time and session, so every build, a fully cached one too, gave a new image ID.
  Recreating the worker kills a running task, which Redis delivers again only after the
  3600 s visibility timeout. `make up` and `make up-scheduler` therefore build with
  `BUILDX_NO_DEFAULT_ATTESTATIONS=1`. `provenance: false` in the compose file was tried first
  and had no effect: Compose 2.40 builds through Bake and does not pass it on. Checked on
  2026-10-31 in a disposable compose project: before, two builds with nothing changed had the
  same image manifest and config and differed only in the attestation; after, a second
  `make up` kept all six containers, and a changed frontend or backend file still rebuilt and
  recreated only the services built from it. A plain `docker compose up --build` still adds
  the attestation. The release workflow has its own build settings, so the published images
  do not change.

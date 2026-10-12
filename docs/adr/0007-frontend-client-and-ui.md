# 0007: Frontend client and UI

## Context
The frontend is a Next.js app (App Router) that talks to the FastAPI server. It needs a
typed API client that cannot drift from the server, a chat page that shows a streamed
answer with citations, and UI components that are reviewed in the repository rather than
pulled in as a black box. The app is local-first: it runs on localhost next to the API.

## Decision

**Generated client.** `backend/app/export_openapi.py` writes the OpenAPI schema from the
route definitions (no server, stores, network or API keys) to `frontend/openapi.json`.
`pnpm gen:api` turns it into `frontend/lib/api/schema.d.ts` with `openapi-typescript`, and
`lib/api/client.ts` wraps it with `openapi-fetch`. `make openapi` runs both steps and
`make lint` fails if either committed file is stale. The `/chat` stream is the exception:
OpenAPI cannot describe server-sent events, so its event types are written by hand.

**Same-origin proxy.** The browser calls `/api/*`; `next.config.ts` rewrites that to the
API (`API_ORIGIN`, default `http://127.0.0.1:8000`). There is no CORS configuration and
the chat stream is not cross-origin. Two settings keep the stream alive through the
proxy: `compress: false` (gzip would hold chunks back) and `experimental.proxyTimeout`
of five minutes (a turn is silent while search and rerank run). The destination is read
when Next builds or starts, so a container build must set `API_ORIGIN`.

**Three read-only endpoints added for the frontend** (`app/api/papers.py`):
`GET /papers/{arxiv_id}`, `GET /papers/{arxiv_id}/chunks` and `GET /chunks/{chunk_id}`.
They were not part of the original M5 scope. They are needed because the chat stream's
`sources` events carry no chunk text (the side panel fetches it by chunk id) and there was
no paper lookup for `/paper/[id]`. They read Postgres only, so they cost nothing; listing a
paper's chunks through `/search` would embed a query (a paid request) for no reason.

**shadcn/ui components are copied into the repository.** `components/ui/{button, input,
badge, card, sheet}.tsx` come from the shadcn registry (style `new-york`; the registry's
JSON files were fetched once, six requests, and the sources written from them). They
declare only `@radix-ui/react-slot` and `@radix-ui/react-dialog`. The shadcn CLI is not a
dependency.
- Local change: the Sheet's close icon was `X` from `lucide-react`. That package is about
  36 MB unpacked for one icon, so it is not installed and the icon is an inline SVG
  (the same two strokes). Each file starts with a comment saying where it came from.
- Formatting: the sources were run through the repository's Prettier (semicolons).
- The components use semantic colour tokens (`bg-primary`, `border-input`, `ring-ring`);
  they are defined in `app/globals.css` for light and dark, and the open and close
  animations come from `tw-animate-css`.

**No remote fonts.** The default `next/font/google` setup downloads font files from
Google at dev and build time. The app uses system font stacks instead.

**Server state with TanStack Query.** Queries are never retried or refetched on window
focus, and stay fresh for five minutes, because `/search` embeds the question (a paid
OpenAI request) and must not repeat behind the user's back.

**Pinned dependencies.** Exact versions with install scripts disabled; the lockfile is
committed. Added: `@tanstack/react-query`, `openapi-fetch`, `openapi-typescript` (dev),
`class-variance-authority`, `clsx`, `tailwind-merge`, `tw-animate-css`,
`@radix-ui/react-slot`, `@radix-ui/react-dialog`.

## Consequences
- A route change in the API shows up as a type error in the frontend after `make openapi`.
- Component sources can be edited like any other code; updating one from shadcn is a
  manual diff, and the Sheet icon change must be kept.
- The three new endpoints widen the API surface; they are read-only and unauthenticated,
  like the rest of the local API.
- Because the proxy target is fixed at build time, running the frontend in a container
  needs `API_ORIGIN=http://api:8000` at build.

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

**Chat streams over fetch, not EventSource.** `/chat` takes a POST body and EventSource
only does GET, so `lib/chat/sse.ts` is a small parser for fetch bodies. It follows the
SSE format (all three line endings, comments, multi-line data) and keeps unfinished lines
and split multi-byte characters between chunks. `lib/chat/events.ts` types every event and
checks each payload instead of casting it; an unknown event name is ignored, so a newer
backend still works. A stream that ends before `done` or `error` is an error, not a
silent half answer.

**Citations.** The model writes `[3]` or `[1, 2]`; `lib/chat/citations.ts` splits the
answer into text and citation segments with the same pattern the backend uses. A number
that is one of the answer's sources becomes a chip; a number with no source stays plain
text, so no chip ever opens nothing. A chip for a source whose `chunk_type` is `abstract`
has a dashed border and the word "abstract". While the answer streams, an unfinished
marker such as `[1,` at the end is held back so the line does not jump when it completes.
The chunk text is fetched from `GET /chunks/{id}` when a chip is opened. The arXiv link
is only rendered if it starts with `https://arxiv.org/abs/`.

**Tests without a new dependency.** The modules in `lib/chat/` are plain TypeScript with
no framework imports, and Node 24 strips types itself, so their tests run with
`node --test` (`pnpm test:unit`, part of `make test`). To make that safe, `tsconfig.json`
sets `erasableSyntaxOnly` (no enums or parameter properties), `verbatimModuleSyntax`
(type-only imports must say `import type`) and `allowImportingTsExtensions` (relative
imports in `lib/` name the `.ts` file, as Node needs). React components are covered by the
browser test, not here. `package.json` is `"type": "module"`.

**Streaming through the proxy was measured, not assumed.** A fake backend that sends
tokens 600 ms apart, with the headers and chunked encoding of the real route, was read
through the Next rewrite by the real client code: in `next dev --webpack`, in
`next build --webpack` plus `next start`, and in `node .next/standalone/server.js` the
tokens arrived 600 ms apart, as they do when read from the backend directly. An optional
timing log (`localStorage["chat-debug"] = "1"`, then `window.__chatTimeline`) records
when each event reaches the page, for the same check against the real API.

**The sources panel is non-modal.** A citation opens the cited passage in a side panel
that has no backdrop and no focus trap, so the answer stays readable, scrollable and
clickable beside it. Escape and the close button close it; clicking elsewhere does not, so
a click on another chip switches the panel instead of closing it (Radix would otherwise
dismiss on any outside interaction; `onInteractOutside` is prevented, with the rule kept in
`lib/chat/panel.ts` where it is tested). On wide screens the chat column moves over to make
room. `components/ui/sheet.tsx` gets a second local change for this: `overlay={false}`.

**Search and paper pages.** `/search` keeps its query in the URL (`?q=&category=&from=&to=`)
so a search can be linked and the back button works; `lib/search/params.ts` validates it
against the API's limits. It calls `GET /search` with `rerank=false` (the cross-encoder
needs about 20 s on a CPU) and a fixed limit of 20, groups the passages by paper in rank
order, and shows an `abstract` badge on abstract passages. Categories offered are the
backend's default fetch list. `/paper/[id]` uses only `GET /papers/{id}` and
`GET /papers/{id}/chunks`, which read Postgres, so opening a paper costs nothing; there is
no "ask about this paper" yet. Both pages share one expandable `Passage` component. All
links from API data go through `lib/links.ts`: the arXiv link must start with
`https://arxiv.org/abs/` (never a PDF), code and project links must be http(s).
The endpoint functions in `lib/api/endpoints.ts` take the client as an argument, so their
tests use a mocked `fetch` and check the exact request URL without any network.

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
- The paper route is `/paper/[id]`, one path segment. Old-style arXiv ids contain a slash
  (`hep-th/9901001`); they work through the API and in links (the id is URL-encoded), but
  are not tested, and the index holds recent papers only.
- Answers are shown as pre-wrapped text, not rendered markdown: no markdown library was
  added. A list or emphasis in an answer appears as the characters the model wrote.
- Because the proxy target is fixed at build time, running the frontend in a container
  needs `API_ORIGIN=http://api:8000` at build.

import createClient from "openapi-fetch";

import type { paths } from "./schema";

// Typed client for the FastAPI routes in openapi.json (regenerate with `pnpm gen:api`).
// The browser calls /api/*; next.config.ts proxies that to the API, so there is no CORS.
// /chat is not used through this client: it is a server-sent-events stream, which the
// OpenAPI schema cannot describe (see lib/chat).
export const api = createClient<paths>({ baseUrl: "/api" });

import type { NextConfig } from "next";

// Where the FastAPI server runs. Read when Next builds or starts, not in the browser.
const API_ORIGIN = process.env.API_ORIGIN ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  // Self-contained server bundle so the Docker image stays small.
  output: "standalone",
  // Local app on localhost: nothing to gain from gzip, and gzip would hold back the
  // chunks of the chat stream (server-sent events) that /api/chat proxies.
  compress: false,
  // The browser calls /api/*, same origin as the pages, so the API needs no CORS setup.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/:path*` }];
  },
  experimental: {
    // A chat turn can be silent for a while (search and rerank run before the sources
    // are sent); the proxy must not cut the stream after its default 30 s.
    proxyTimeout: 5 * 60 * 1000,
  },
};

export default nextConfig;

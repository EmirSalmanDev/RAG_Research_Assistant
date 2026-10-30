# Pinned by digest, not only by tag (like the backend's base images): the digest is the
# multi-platform index, so the amd64 and arm64 builds of the release use the same Node.
# Updating Node is then a deliberate change of both FROM lines.
FROM node:24-slim@sha256:c86ecead9e55a5210ac1727de38a4ca840bf03dcaf34dc9749addbf383b1920c AS build
RUN corepack enable
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY . .
# next.config.ts reads API_ORIGIN when Next builds: the /api proxy target is written into
# the build output, so setting it when the container starts changes nothing. The default is
# the API's compose service name; docker-compose.yml passes FRONTEND_API_ORIGIN here.
ARG API_ORIGIN=http://api:8000
ENV API_ORIGIN=$API_ORIGIN
# `next build` reports anonymous usage data to Vercel unless this is set; the image build
# (here and in the release workflow) contacts no host but the package registry.
ENV NEXT_TELEMETRY_DISABLED=1
RUN pnpm build

FROM node:24-slim@sha256:c86ecead9e55a5210ac1727de38a4ca840bf03dcaf34dc9749addbf383b1920c
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
WORKDIR /app
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
EXPOSE 3000
CMD ["node", "server.js"]

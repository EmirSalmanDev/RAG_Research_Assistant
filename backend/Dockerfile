# Pinned by digest, not only by tag: when the tag moves, every layer below is rebuilt and
# all dependencies are downloaded again. Updating the base image (e.g. for security fixes)
# is then a deliberate change of this line.
FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016

# uv is copied from its official image; it installs exactly what uv.lock pins.
# Pinned by version and digest so every build uses the same uv (keep the local uv on the
# same version: `uv self update <version>`).
COPY --from=ghcr.io/astral-sh/uv:0.12.21@sha256:a7aed3216253ee804de3e2d8afa5073baa1a177335345d43845cd4165e43b711 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first so this layer is cached until the lockfile changes. The cache mount
# keeps uv's download cache between builds (it is not part of the image), so a lockfile
# change downloads only the packages that changed. UV_LINK_MODE=copy (above) is needed
# because the cache lives on a different filesystem than the virtualenv.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Tokenizer file for chunking (tiktoken cl100k_base, ~1.7 MB), fetched once at build time
# so the containers never download it. tiktoken checks it against its pinned SHA-256.
# BM25 needs no download: its stopword list is committed in app/ingest/data/.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .
ENV PATH="/app/.venv/bin:$PATH"

# Default command = the API; worker and beat override it in docker-compose.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

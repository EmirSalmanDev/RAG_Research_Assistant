FROM python:3.12-slim

# uv is copied from its official image; it installs exactly what uv.lock pins.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

# Dependencies first so this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Tokenizer file for chunking (tiktoken cl100k_base, ~1.7 MB), fetched once at build time
# so the containers never download it. tiktoken checks it against its pinned SHA-256.
# BM25 needs no download: its stopword list is committed in app/ingest/data/.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY . .
ENV PATH="/app/.venv/bin:$PATH"

# Default command = the API; worker and beat override it in docker-compose.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

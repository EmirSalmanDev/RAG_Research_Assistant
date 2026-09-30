# 0002: Sync data layer and source clients; citations without a foreign key

## Context
M1 adds the Postgres schema and four external API clients (arXiv, Hugging Face, GitHub,
Semantic Scholar). Ingestion runs in Celery tasks and a CLI, which are synchronous. arXiv
allows 1 request per 3 s and Semantic Scholar 1 request/s for this project, across all
processes. S2 will run without an API key and often answers 429.

## Decision
- SQLAlchemy sessions and the source clients are synchronous. `/health` keeps its own
  async engine; nothing else needs async yet.
- One Redis limiter (`app/sources/ratelimit.py`) serves all processes: a lock (one request at
  a time, with a TTL so a crashed worker cannot block others) plus a "gate" key (minimum
  spacing between request starts). Retries happen outside the lock.
- Retries (tenacity): exponential backoff with jitter, `Retry-After` honoured, only 429, 5xx
  and network errors. The `arxiv` package runs with `num_retries=0` and `delay_seconds=0`;
  we page manually so every page is one limiter slot.
- Semantic Scholar failures never raise: the client returns a typed `S2Unavailable` and
  ingest-one records it in `ingest_runs.errors`.
- `citations.cited_id` has no foreign key: references often point at papers we do not hold.
  Only outgoing edges (references) are stored; inbound edges can be derived.
- Enum-like columns use CHECK constraints, not Postgres ENUM types.
- Tests replay cassettes with `pytest-recording`; pytest runs with `--block-network`.
  The recorder replaces Hugging Face full text with invented text and strips credentials.

## Consequences
- Tests for the schema and the limiter need the local Postgres and Redis (`make up`, service
  containers in CI); everything else runs without services.
- Citation edges to papers not yet ingested are kept, so they can be joined later.
- Hugging Face model/dataset/space lists are capped at 100 per kind (`limit=100`).

# Cassettes

Recorded HTTP responses, replayed by `pytest-recording` so tests run with the network off.
`pyproject.toml` runs pytest with `--block-network --record-mode=none`: a test that makes
a call without a cassette fails instead of reaching the internet.

Record a module (real network; ask the owner first, CLAUDE.md rule 5):

    cd backend
    uv run pytest -o addopts="" --record-mode=once tests/unit/test_<name>_cassettes.py

`-o addopts=""` switches off the network block for this run only. Delete the cassette file
first to re-record it. arXiv and Semantic Scholar calls go through the Redis limiter while
recording (needs `make up`), so the 3 s / 1 s limits hold.

What the recorder strips or replaces (`tests/conftest.py`, `before_record_response`):
- request headers `authorization`, `x-api-key`, `cookie`; query params `token`, `api_key`,
  `access_token`, `key`; response `set-cookie`;
- Hugging Face paper full text (`.md`): replaced by an invented sentence (CLAUDE.md rule 14);
- GitHub README: cut to 1000 bytes; HF `daily_papers`: 3 entries, no submitter profiles;
  HF model/dataset/space lists: 10 entries;
- any body over 300 KB makes recording fail.

Always grep a new cassette for secrets and read it before committing.

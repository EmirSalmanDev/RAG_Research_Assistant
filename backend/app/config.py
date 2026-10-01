"""Application settings, read from environment variables / `.env`.

Model names live here and nowhere else (never hard-code them in other modules).
"""

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Inside Docker Compose these are overridden to use service hostnames
    # (postgres, qdrant, redis); the defaults suit running natively.
    # `.env` lives at the repo root (docker compose reads it there); natively we run from
    # backend/, so look in both places. Real environment variables win over the files.
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    database_url: str = "postgresql+psycopg://copilot:copilot@localhost:5432/copilot"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    github_token: SecretStr | None = None
    s2_api_key: SecretStr | None = None

    # --- Ingestion -------------------------------------------------------------------
    # arXiv categories fetched daily and by the backfill. In .env, a JSON list:
    # ARXIV_CATEGORIES='["cs.CL","cs.LG"]'
    arxiv_categories: list[str] = ["cs.CL", "cs.LG", "cs.AI", "cs.IR"]

    # Embeddings. Changing the model or the dimension means re-embedding everything,
    # and the Qdrant collection must be recreated with the new size.
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_price_per_1m: float = 0.02  # USD per 1M input tokens, for `llm_usage`
    embedding_batch_size: int = 100  # inputs per embeddings request
    sparse_model: str = "Qdrant/bm25"  # FastEmbed model for the `text-sparse` vector
    qdrant_collection: str = "chunks"

    # Chunking, measured in tokens of the embedding model's tokenizer.
    chunk_tokens: int = 512  # target size of a chunk inside a long section
    chunk_overlap: int = 64  # tokens repeated between neighbouring windows
    chunk_min_tokens: int = 64  # smaller sections are merged into the next one
    # Tables, code and display math are kept whole up to this size, then split (tables
    # by rows with the header repeated). Must stay well below the model's 8191-token
    # input limit, which also has to fit the "Title > Section" prefix.
    chunk_hard_max_tokens: int = 2048

    @field_validator(
        "openai_api_key", "langfuse_public_key", "langfuse_secret_key", "github_token",
        "s2_api_key", mode="before",
    )  # fmt: skip
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        """`S2_API_KEY=` in .env arrives as "": treat it as "no key", not an empty key."""
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()

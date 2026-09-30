"""Application settings, read from environment variables / `.env`.

Model names will be added here in later milestones (never hard-code them elsewhere).
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

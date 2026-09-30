"""Application settings, read from environment variables / `.env`.

Model names will be added here in later milestones (never hard-code them elsewhere).
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Inside Docker Compose these are overridden to use service hostnames
    # (postgres, qdrant, redis); the defaults suit running natively.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://copilot:copilot@localhost:5432/copilot"
    qdrant_url: str = "http://localhost:6333"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    github_token: SecretStr | None = None
    s2_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()

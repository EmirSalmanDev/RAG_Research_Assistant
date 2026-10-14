"""Application settings, read from environment variables / `.env`.

Model names live here and nowhere else (never hard-code them in other modules).
"""

from functools import lru_cache
from typing import Literal

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

    # Enrichment (GitHub, Hugging Face Hub, Semantic Scholar) in the daily chain.
    enrich_max_papers: int = 500  # papers looked at per run (bounds a first or late run)
    # Papers added within this many days are looked at again while S2 has not answered for
    # them yet, or while they have a GitHub link but no repo row (both cheap retries).
    enrich_recheck_days: int = 7

    # Chunking, measured in tokens of the embedding model's tokenizer.
    chunk_tokens: int = 512  # target size of a chunk inside a long section
    chunk_overlap: int = 64  # tokens repeated between neighbouring windows
    chunk_min_tokens: int = 64  # smaller sections are merged into the next one
    # Tables, code and display math are kept whole up to this size, then split (tables
    # by rows with the header repeated). Must stay well below the model's 8191-token
    # input limit, which also has to fit the "Title > Section" prefix.
    chunk_hard_max_tokens: int = 2048

    # --- Retrieval (docs/adr/0005) ---------------------------------------------------
    # Every stage has its own switch, so the evaluation can measure what each one adds.
    # search_mode: "dense" (the question's OpenAI embedding), "bm25" (local, free) or
    # "hybrid" (both lists, fused with Reciprocal Rank Fusion inside Qdrant).
    search_mode: Literal["dense", "bm25", "hybrid"] = "hybrid"
    search_prefetch_limit: int = 50  # candidates taken from each list in hybrid mode
    search_rrf_k: int = 60  # RRF constant k (the RRF paper's value; Qdrant's default is 2)
    search_candidates: int = 50  # results handed from search to the reranker
    # Default for GET /search (a request can override it with ?rerank=true). Off because a
    # rerank of 50 candidates took about 22 s on an Apple M1 CPU (docs/adr/0005). The
    # retrieval evaluation sets rerank per configuration and does not read this.
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"  # a FastEmbed TextCrossEncoder model
    rerank_batch_size: int = 16  # (question, chunk) pairs per model run; bounds memory
    max_chunks_per_paper: int = 3  # diversity cap after reranking; 0 switches it off
    # Where the reranker's files are. They are never downloaded at runtime: only
    # `make fetch-reranker` downloads them (a deliberate step, see app/rag/rerank.py).
    fastembed_cache_dir: str = "~/.cache/fastembed"

    # --- Chat (docs/adr/0006) ---------------------------------------------------------
    # One model for every chat call (condense, plan, generate, verify). The prices are
    # only used to write `llm_usage.cost_usd`; no cached-input discount is assumed.
    chat_model: str = "gpt-4o-mini"
    chat_price_in_per_1m: float = 0.15  # USD per 1M input tokens
    chat_price_out_per_1m: float = 0.60  # USD per 1M output tokens
    chat_temperature: float = 0.0  # 0 = as repeatable as the API allows
    chat_max_output_tokens: int = 1000  # cap for a qa answer (and for any other call)
    chat_max_output_tokens_explain: int = 1600  # explain answers may be longer
    chat_history_messages: int = 6  # earlier messages condense reads (N; no running summary)
    chat_sources: int = 8  # numbered sources [1]..[n] given to generate
    # Stage switches, so the evaluation can measure what each stage adds. Search always
    # runs; rerank uses `rerank_enabled` above (off by default).
    condense_enabled: bool = True  # rewrite follow-ups into a standalone question
    plan_enabled: bool = True  # filters, intent and rephrasings from one structured call
    plan_rephrasings_enabled: bool = True  # also search the planner's two rephrasings
    verify_enabled: bool = True  # second call: is each cited sentence supported?

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

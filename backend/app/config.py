"""Application settings, read from environment variables / `.env`.

Model names live here and nowhere else (never hard-code them in other modules).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# How long Redis waits for an unacknowledged Celery message before it delivers the message
# again (kombu's own default, set explicitly in app/worker/celery_app.py). A task retry
# waiting for its countdown is such a message, so every retry delay must stay below it.
BROKER_VISIBILITY_TIMEOUT_S = 3600


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
    # The daily arXiv fetch reads from the last run's cursor minus 4 days, but never more
    # than this many days back: after a long pause (weeks without a run) one daily run
    # would otherwise page through tens of thousands of papers. The skipped days are
    # recorded on the run (`window_capped`, `skipped_from`, `skipped_to`) and shown on the
    # health page; `make backfill` fetches them on purpose. Not applied to the backfill.
    arxiv_daily_max_window_days: int = 7
    # Task-level retries of the chain's two fetch steps, fetch_arxiv and fetch_hf_daily
    # (docs/adr/0003). They wait minutes, not seconds, so a source that is down for a few
    # minutes does not cost the day: retry n waits first_delay * factor**(n-1) minutes,
    # times a random factor between 1 - jitter and 1 + jitter (2, 6 and 18 min +/-20 %).
    # The longest delay must stay below BROKER_VISIBILITY_TIMEOUT_S (checked below).
    daily_fetch_retries: int = Field(default=3, ge=0)
    daily_fetch_retry_first_delay_min: float = Field(default=2.0, gt=0)
    daily_fetch_retry_factor: float = Field(default=3.0, ge=1)
    daily_fetch_retry_jitter: float = Field(default=0.2, ge=0, lt=1)

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
    # ONNX Runtime settings of the reranker (docs/adr/0012). They change speed and memory,
    # not what is ranked first; the reranker comparison of the evaluation measures them.
    # None = ONNX Runtime's own thread count (one per core); False switches its CPU memory
    # arena off. The defaults keep the behaviour measured in docs/adr/0005.
    rerank_threads: int | None = None
    rerank_cpu_arena: bool = True
    # diversity cap after reranking, on a paper's own text; 0 switches this cap off (the
    # analysis limit below still applies)
    max_chunks_per_paper: int = 3
    # Generated analysis chunks (a model's summary, not the paper's words) are capped on
    # their own and never take one of the paper's slots above (docs/adr/0012): at most
    # this many per paper. 0 leaves them out; -1 counts them like any chunk (as before).
    max_analysis_chunks_per_paper: int = 1
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

    # --- Paper analysis (OpenAI Batch API; docs/adr/0008) ----------------------------
    # Full-text papers are analysed in the background: results arrive within 24 hours at
    # half the chat price. The Batch prices are only used to write `llm_usage.cost_usd`.
    analysis_model: str = "gpt-4o-mini"
    analysis_prompt: str = "analyze_v1"  # the prompt file (app/rag/prompts/<name>.txt)
    batch_price_in_per_1m: float = 0.075  # USD per 1M input tokens on the Batch API
    batch_price_out_per_1m: float = 0.30  # USD per 1M output tokens on the Batch API
    analysis_max_papers: int = 20  # papers per submission (one submission a day)
    analysis_max_input_tokens: int = 12_000  # the paper text is cut to this many tokens
    analysis_max_output_tokens: int = 1200  # per analysis (a cut-off answer is a failure)
    analysis_max_attempts: int = 2  # failed batches per paper and prompt before giving up
    # The scheduled daily submission spends money, so it is off until the owner turns it on.
    analysis_auto_submit: bool = False

    # --- Daily digest (docs/adr/0009) ------------------------------------------------
    # The settings page can override digest_size, the categories, fulltext_top_n and the
    # budget cap (table app_settings); these are the defaults when nothing is stored.
    digest_size: int = 20  # papers in one day's digest
    digest_window_days: int = 7  # only papers published within this many days are candidates
    digest_trending_upvotes: int = 10  # hf_upvotes at or above this show a "trending" badge
    digest_timezone: str = "Europe/Istanbul"  # a digest's date is the date in this zone
    digest_summary_max_chars: int = 200  # fallback card summary: first sentence, cut here
    fulltext_top_n: int = 20  # top papers by relevance that get full text each night
    # Relevance score of a paper against one interest i (app/features/relevance.py):
    #   w_i * (cosine_weight * cos + keyword_weight * keyword_share) + category_bonus * match
    # and the paper's score is the best interest's value plus the upvote boost.
    relevance_cosine_weight: float = 0.8
    relevance_keyword_weight: float = 0.2
    relevance_category_bonus: float = 0.05
    relevance_upvote_boost: float = 0.05  # largest boost; reached at relevance_upvote_cap
    relevance_upvote_cap: int = 100  # upvotes at which the boost stops growing (log scale)
    # Feedback: Save and Dismiss nudge the weight of the paper's best interest.
    interest_save_delta: float = 0.10
    interest_dismiss_delta: float = -0.05
    interest_weight_min: float = 0.5
    interest_weight_max: float = 2.0
    interest_weight_decimals: int = 6  # weights are rounded so that Undo returns exactly

    # --- Paper comparison and topic timeline (docs/adr/0010) -------------------------
    # Both features spend money only when the user presses the button (nothing runs on
    # page load) and persist nothing.
    # Comparison: the fixed question each dimension is retrieved with. The same text is
    # used for every paper, so all of them go to the embeddings API in one request.
    compare_dimension_queries: dict[str, str] = {
        "task": "What problem or task does the paper address, and what is its goal?",
        "method": "What method, model or approach does the paper propose, and how does it work?",
        "data": "Which datasets, corpora or benchmarks does the paper use for training and "
        "evaluation?",
        "metrics": "Which evaluation metrics does the paper report, and what are its main results?",
        "compute": "What compute does the paper use: hardware, training time, model size, "
        "number of parameters or cost?",
        "limitations": "What limitations, failure cases or open problems does the paper state?",
    }  # fmt: skip
    compare_top_k: int = 3  # chunks kept per paper and dimension (reduced 3 -> 2 -> 1 if too big)
    # A paper-filtered hybrid query returns chunks of every type; generated analysis chunks
    # are dropped in code (they are model output, not evidence), so ask for a few more.
    compare_fetch_limit: int = 12
    compare_evidence_types: list[str] = ["abstract", "section"]
    compare_max_input_tokens: int = 30_000  # cl100k_base count of the fill call's input
    compare_source_max_tokens: int = 800  # a longer source (a big table) is cut to this
    compare_max_output_tokens: int = 4000  # 5 papers x 6 dimensions of <= 40 words, as JSON
    compare_cell_max_words: int = 40
    compare_prompt: str = "compare_v1"
    # Timeline: candidates come from one hybrid query over abstracts; the result list is
    # the union of the two prefetch lists at most (2 x search_prefetch_limit).
    timeline_search_limit: int = 100
    # key_score = relevance_weight * topic_relevance + citation_weight * citation_norm,
    # where topic_relevance is 1 for the best-matching paper down to 0 for the last, and
    # citation_norm = ln(1 + citations) / ln(1 + the set's largest count).
    timeline_relevance_weight: float = 0.7
    timeline_citation_weight: float = 0.3
    timeline_line_max_words: int = 30  # a longer "what changed" line is replaced
    timeline_line_fallback_chars: int = 200  # fallback: the abstract's first sentence, cut here
    timeline_abstract_max_tokens: int = 400  # an abstract is cut to this in the prompt
    timeline_max_output_tokens: int = 1500
    timeline_narrow_days: int = 30  # a result spanning fewer days is flagged as a narrow range
    timeline_prompt: str = "timeline_v1"

    # --- Library, literature review and PDF upload (docs/adr/0011) --------------------
    # The review reads Postgres only (stored analyses and abstracts): no retrieval, no
    # embeddings. It spends money only when its button is pressed and stores nothing.
    review_map_prompt: str = "review_map_v1"
    review_write_prompt: str = "review_write_v1"
    review_default_papers: int = 30  # papers a review uses unless the request says otherwise
    review_max_papers: int = 60  # hard cap on `max_papers`
    review_direct_max_papers: int = 8  # up to this many papers the writer reads them directly
    review_map_group_size: int = 8  # papers per map call above that
    review_field_max_tokens: int = 250  # an analysis field or an abstract is cut to this
    review_map_max_output_tokens: int = 1500  # theme notes of one group, as JSON
    review_write_max_output_tokens: int = 4000  # the draft, as JSON
    review_note_max_words: int = 60  # a longer theme note statement is cut
    review_max_citations_per_sentence: int = 4  # further source numbers of a sentence are dropped
    # Transitions ("Several lines of work follow.") may be uncited but are limited to this
    # share of the kept sentences; a transition that names an author or holds a number is
    # an uncited claim and is removed.
    review_max_transition_share: float = 0.25
    # PDF upload: the raw request body is read up to this size before any parsing.
    upload_max_bytes: int = 25 * 1024 * 1024
    upload_abstract_chars: int = 1500  # no abstract given: the abstract chunk is the body's start
    upload_min_text_chars: int = 500  # less parsed text than this: no text layer (scanned PDF)

    # --- Evaluation (docs/adr/0012) --------------------------------------------------
    # One model writes the test-set questions and judges the answers (faithfulness,
    # response relevancy, context precision and recall, citation accuracy). It is fixed
    # here and recorded on every run, always at temperature 0. The prices only feed
    # `llm_usage.cost_usd`. Nothing here runs unless an evaluation command is started.
    eval_model: str = "gpt-4o-mini"
    eval_temperature: float = 0.0
    eval_price_in_per_1m: float = 0.15
    eval_price_out_per_1m: float = 0.60
    eval_judge_max_output_tokens: int = 2000  # one structured verdict list per call
    eval_relevancy_questions: int = 3  # questions written back from an answer (RAGAS: strictness)
    # Response relevancy compares embeddings. Its model is fixed here, apart from the
    # retrieval's embedding_model, so an ablation of the retrieval embeddings does not change
    # the yardstick it is measured with.
    eval_embedding_model: str = "text-embedding-3-small"
    eval_embedding_price_per_1m: float = 0.02
    eval_testset_max_output_tokens: int = 400  # one question and its reference answer
    # Ablations (backend/eval/variants.py). The models the variants try are named here, like
    # every model. The second reranker (ablation 5) is also the small model of the reranker
    # comparison: the one candidate docs/adr/0005 left that FastEmbed loads and that has a
    # licence allowing any use. The large embedding model is ablation 8 (at its own size and
    # shortened to embedding_dim); its price feeds llm_usage when a variant embeds with it.
    eval_second_reranker_model: str = "Xenova/ms-marco-MiniLM-L-12-v2"
    eval_large_embedding_model: str = "text-embedding-3-large"
    eval_large_embedding_dim: int = 3072
    eval_large_embedding_price_per_1m: float = 0.13

    @field_validator(
        "openai_api_key", "langfuse_public_key", "langfuse_secret_key", "github_token",
        "s2_api_key", mode="before",
    )  # fmt: skip
    @classmethod
    def _empty_means_unset(cls, value: object) -> object:
        """`S2_API_KEY=` in .env arrives as "": treat it as "no key", not an empty key."""
        return None if value == "" else value

    @model_validator(mode="after")
    def _fetch_retries_fit_the_broker(self) -> "Settings":
        """Refuse a retry schedule whose longest delay could reach the visibility timeout:
        Redis would deliver the waiting retry a second time (two runs of the same step)."""
        if self.daily_fetch_retries > 0:
            longest_s = (
                self.daily_fetch_retry_first_delay_min
                * self.daily_fetch_retry_factor ** (self.daily_fetch_retries - 1)
                * (1 + self.daily_fetch_retry_jitter)
                * 60
            )
            if longest_s >= BROKER_VISIBILITY_TIMEOUT_S:
                raise ValueError(
                    f"the longest daily fetch retry delay ({longest_s:.0f} s) must stay below "
                    f"the broker visibility timeout ({BROKER_VISIBILITY_TIMEOUT_S} s)"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

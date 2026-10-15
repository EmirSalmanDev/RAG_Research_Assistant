"""All tables. Postgres is the source of truth; Qdrant only holds derived vectors.

Conventions:
- every timestamp is timezone-aware (`DateTime(timezone=True)`), stored in UTC;
- every foreign key column is indexed;
- enum-like text columns get a CHECK constraint instead of a Postgres ENUM type
  (easier to extend in a later migration).
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# papers.fulltext_status (CHECK constraint below and in migration 0002):
#   none        - nobody asked for the full text (default; abstract only)
#   requested   - asked for by hand; the next selection step picks it up
#   fetched     - full text stored in paper_texts, not yet chunked and indexed
#   indexed     - full-text chunks are in Postgres and Qdrant
#   unavailable - no source (HF .md, arXiv HTML, PDF) had usable text
#   failed      - a step failed after its last retry (shown on the health page)
FULLTEXT_STATUSES = ("none", "requested", "fetched", "indexed", "unavailable", "failed")


def _paper_fk() -> ForeignKey:
    # A ForeignKey object belongs to one column, so build a fresh one each time.
    return ForeignKey("papers.arxiv_id", ondelete="CASCADE")


def _now() -> Any:
    return func.now()


class Paper(Base):
    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(Text, primary_key=True)  # e.g. "1706.03762"
    version: Mapped[int] = mapped_column(Integer, default=1)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text)
    authors: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    # arXiv's order is kept: the primary category comes first.
    categories: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    primary_category: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    hf_upvotes: Mapped[int | None] = mapped_column(Integer)
    citation_count: Mapped[int | None] = mapped_column(Integer)
    github_url: Mapped[str | None] = mapped_column(Text)
    project_url: Mapped[str | None] = mapped_column(Text)  # project page, from Hugging Face
    # See FULLTEXT_STATUSES for what each value means.
    fulltext_status: Mapped[str] = mapped_column(Text, default="none", server_default="none")
    fulltext_source: Mapped[str | None] = mapped_column(Text)
    relevance_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_now(), onupdate=_now()
    )

    __table_args__ = (
        Index("ix_papers_published_at", "published_at"),
        CheckConstraint(
            "fulltext_status IN (" + ", ".join(f"'{s}'" for s in FULLTEXT_STATUSES) + ")",
            name="fulltext_status",
        ),
    )


class PaperText(Base):
    """A paper's full text, kept so it can be re-cleaned and re-chunked without a download.

    Two versions of the text are stored:
    - `raw_markdown`: the text exactly as fetched or converted (Hugging Face .md,
      arXiv HTML -> markdown, or PDF -> markdown), before any cleaning;
    - `markdown`: the CLEANED text (output of `app/ingest/clean.py`), the input to chunking.
    The reference list cut out by cleaning goes to `references`.
    """

    __tablename__ = "paper_texts"

    arxiv_id: Mapped[str] = mapped_column(_paper_fk(), primary_key=True)
    version: Mapped[int] = mapped_column(Integer)  # arXiv version the text was fetched for
    raw_markdown: Mapped[str] = mapped_column(Text)  # as fetched/converted, NOT cleaned
    markdown: Mapped[str] = mapped_column(Text)  # cleaned; this is what gets chunked
    references: Mapped[str | None] = mapped_column(Text)  # reference list cut by cleaning
    token_count: Mapped[int] = mapped_column(Integer)  # tokens in the cleaned `markdown`
    parsed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class PaperAnalysis(Base):
    """One analysis per paper (unique `arxiv_id`), replaced when a newer prompt version
    analyses the paper again. Written by the Batch API collector (app/llm/batch.py)."""

    __tablename__ = "paper_analyses"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    arxiv_id: Mapped[str] = mapped_column(_paper_fk(), index=True)
    tldr: Mapped[str | None] = mapped_column(Text)
    contributions: Mapped[str | None] = mapped_column(Text)
    method: Mapped[str | None] = mapped_column(Text)
    results: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    prompt_version: Mapped[str | None] = mapped_column(Text)  # e.g. "analyze_v1" (rule 7)
    batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("analysis_batches.id", ondelete="SET NULL"), index=True
    )

    __table_args__ = (UniqueConstraint("arxiv_id", name="uq_paper_analyses_arxiv_id"),)


class AnalysisBatch(Base):
    """One OpenAI Batch API job of paper analyses.

    status: 'created' (our row exists, OpenAI does not know the batch yet), then OpenAI's
    own status ('validating', 'in_progress', 'finalizing', 'completed', 'failed',
    'expired', 'cancelling', 'cancelled'), then 'collected' once its results are stored.
    No CHECK constraint: the values after 'created' are OpenAI's, not ours.
    """

    __tablename__ = "analysis_batches"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    openai_batch_id: Mapped[str | None] = mapped_column(Text, unique=True)
    input_file_id: Mapped[str | None] = mapped_column(Text)
    output_file_id: Mapped[str | None] = mapped_column(Text)
    error_file_id: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text)
    request_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    collected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    errors: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )


class AnalysisBatchItem(Base):
    """One paper in a batch. status: 'pending' (queued; at most one per paper, by the
    partial unique index), 'done' (analysis stored) or 'failed' (`error` says why)."""

    __tablename__ = "analysis_batch_items"

    batch_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_batches.id", ondelete="CASCADE"), primary_key=True
    )
    arxiv_id: Mapped[str] = mapped_column(_paper_fk(), primary_key=True)
    status: Mapped[str] = mapped_column(Text, default="pending", server_default="pending")
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index(
            "uq_batch_items_one_pending",
            "arxiv_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    # Same uuid as the Qdrant point id (uuid5 of arxiv_id, version, ord; see CLAUDE.md rule 5).
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    arxiv_id: Mapped[str] = mapped_column(_paper_fk())  # indexed via the unique (arxiv_id, ord)
    chunk_type: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(Text)
    ord: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        CheckConstraint("chunk_type IN ('abstract', 'section', 'analysis')", name="chunk_type"),
        UniqueConstraint("arxiv_id", "ord", name="uq_chunks_arxiv_id_ord"),
    )


class Repo(Base):
    __tablename__ = "repos"

    arxiv_id: Mapped[str] = mapped_column(_paper_fk(), primary_key=True)
    full_name: Mapped[str] = mapped_column(Text, primary_key=True)  # "owner/name"
    stars: Mapped[int | None] = mapped_column(Integer)
    last_push: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    license: Mapped[str | None] = mapped_column(Text)
    readme_summary: Mapped[str | None] = mapped_column(Text)


class HFArtifact(Base):
    __tablename__ = "hf_artifacts"

    arxiv_id: Mapped[str] = mapped_column(_paper_fk(), primary_key=True)
    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    repo_id: Mapped[str] = mapped_column(Text, primary_key=True)
    downloads: Mapped[int | None] = mapped_column(BigInteger)
    likes: Mapped[int | None] = mapped_column(Integer)
    pipeline_tag: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (CheckConstraint("kind IN ('model', 'dataset', 'space')", name="kind"),)


class Citation(Base):
    __tablename__ = "citations"

    citing_id: Mapped[str] = mapped_column(_paper_fk(), primary_key=True)
    # Deliberately NO foreign key: references point at papers we may not hold yet.
    # Join to `papers` later when they are ingested.
    cited_id: Mapped[str] = mapped_column(Text, primary_key=True, index=True)


class Interest(Base):
    __tablename__ = "interests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    keywords: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    categories: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)


class LibraryItem(Base):
    __tablename__ = "library_items"

    arxiv_id: Mapped[str] = mapped_column(_paper_fk(), primary_key=True)
    collection: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, default="to_read")
    notes: Mapped[str | None] = mapped_column(Text)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    trace_id: Mapped[str | None] = mapped_column(Text)  # Langfuse trace id
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    # Assistant messages only: intent, plan, verification, skipped stages, prompt versions
    # (shape and version in app/rag/meta.py; migration 0003). NULL on user messages.
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    __table_args__ = (CheckConstraint("role IN ('user', 'assistant', 'system')", name="role"),)


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    counts: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    errors: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class EvalItem(Base):
    __tablename__ = "eval_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text)
    reference_answer: Mapped[str | None] = mapped_column(Text)
    # Not a foreign key: chunks are rebuilt when chunking changes.
    reference_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    git_sha: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class EvalResult(Base):
    """One row per (run, question): what was retrieved, what was answered, per-question scores."""

    __tablename__ = "eval_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("eval_items.id"), index=True)
    retrieved_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    answer: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())

    __table_args__ = (UniqueConstraint("run_id", "item_id", name="uq_eval_results_run_id_item_id"),)


class LLMUsage(Base):
    __tablename__ = "llm_usage"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    feature: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    tokens_in: Mapped[int] = mapped_column(Integer)
    tokens_out: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())

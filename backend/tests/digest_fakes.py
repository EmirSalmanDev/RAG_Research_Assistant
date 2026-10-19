"""Fakes and builders for the digest tests (no Qdrant, no OpenAI; all text is invented)."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.db.models import Interest, Paper
from app.features import interests as interest_repo

# 08:00 on 2026-10-19 in Istanbul (UTC+3)
NOW = datetime(2026, 10, 19, 5, 0, tzinfo=UTC)


class FakeVectors:
    """Abstract vectors by arXiv id; an id without an entry has no vector in 'Qdrant'."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[list[tuple[str, int]]] = []

    def get(self, papers: Sequence[tuple[str, int]]) -> dict[str, list[float]]:
        self.calls.append(list(papers))
        return {a: self.vectors[a] for a, _ in papers if a in self.vectors}


class FakeDense:
    """Stands in for DenseEmbedder: one deterministic 2-number vector per text."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.calls.append((list(texts), feature))
        return [self.vectors.get(t, [1.0, 0.0]) for t in texts]


def make_paper(
    session: Session,
    arxiv_id: str,
    *,
    age_days: float = 1.0,
    upvotes: int | None = None,
    title: str = "An invented title",
    abstract: str = "An invented abstract. It has two sentences.",
    categories: Sequence[str] = ("cs.CL",),
    version: int = 1,
    now: datetime = NOW,
) -> Paper:
    paper = Paper(
        arxiv_id=arxiv_id, version=version, title=title, abstract=abstract, authors=[],
        categories=list(categories), primary_category=categories[0],
        published_at=now - timedelta(days=age_days), hf_upvotes=upvotes,
    )  # fmt: skip
    session.add(paper)
    session.commit()
    return paper


def make_interest(
    session: Session,
    name: str,
    *,
    embedding: list[float] | None = None,
    keywords: Sequence[str] = (),
    categories: Sequence[str] = (),
    weight: float = 1.0,
    active: bool = True,
) -> Interest:
    """An interest; with `embedding` it counts as already embedded (its hash is current)."""
    row = interest_repo.create_interest(
        session, name=name, description=f"About {name}.", keywords=keywords,
        categories=categories, is_active=active,
    )  # fmt: skip
    row.weight = weight
    if embedding is not None:
        row.embedding = embedding
        row.embedding_model = "m"
        row.embedding_hash = interest_repo.current_hash(row)
    session.commit()
    return row

"""Fakes shared by the compare and timeline tests: papers, search hits, encoders, tracing.
All text is invented."""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from qdrant_client import models as m

from app.features.paper_facts import PaperFacts
from app.rag.search import SearchFilters, SearchHit

DAY = datetime(2026, 9, 1, tzinfo=UTC)


def chunk_uuid(arxiv_id: str, name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{arxiv_id}/{name}")


def make_hit(
    arxiv_id: str,
    name: str,
    *,
    chunk_type: str = "section",
    text: str | None = None,
    title: str | None = None,
    published_at: datetime = DAY,
    section_path: str | None = "3 Method",
    score: float = 0.0,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_uuid(arxiv_id, name),
        arxiv_id=arxiv_id,
        title=title or f"Invented Paper {arxiv_id}",
        section_path=None if chunk_type == "abstract" else section_path,
        chunk_type=chunk_type,
        published_at=published_at.isoformat(),
        text=text or f"Invented passage {name} of {arxiv_id}.",
        score=score,
    )


def make_facts(
    arxiv_id: str,
    *,
    has_fulltext: bool = True,
    citation_count: int | None = None,
    published_at: datetime = DAY,
    abstract: str | None = None,
) -> PaperFacts:
    return PaperFacts(
        arxiv_id=arxiv_id,
        title=f"Invented Paper {arxiv_id}",
        abstract=abstract or f"Invented abstract of {arxiv_id}. It has a second sentence.",
        published_at=published_at,
        citation_count=citation_count,
        has_fulltext=has_fulltext,
    )


class FakeFacts:
    def __init__(self, *facts: PaperFacts) -> None:
        self.by_id = {f.arxiv_id: f for f in facts}
        self.calls: list[list[str]] = []

    def get_facts(self, arxiv_ids: Sequence[str]) -> dict[str, PaperFacts]:
        self.calls.append(list(arxiv_ids))
        return {a: self.by_id[a] for a in arxiv_ids if a in self.by_id}


class FakeDense:
    """Records every embedding request; vectors are tiny and distinct per text."""

    def __init__(self) -> None:
        self.requests: list[tuple[list[str], str]] = []

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.requests.append((list(texts), feature))
        return [[float(i), 1.0, 0.0, 0.0] for i, _ in enumerate(texts)]


class FakeSparse:
    def embed(self, texts: Sequence[str]) -> list[m.SparseVector]:
        return [m.SparseVector(indices=[1], values=[1.0]) for _ in texts]


@dataclass
class FakeSearch:
    """A `search` stand-in: returns the hits registered for the filtered paper."""

    by_paper: dict[str, list[SearchHit]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self, query: str, filters: SearchFilters | None = None, **kwargs: Any
    ) -> list[SearchHit]:
        self.calls.append({"query": query, "filters": filters, **kwargs})
        if filters is not None and filters.arxiv_ids:
            return list(self.by_paper.get(filters.arxiv_ids[0], []))
        return [h for hits in self.by_paper.values() for h in hits]


def dense_factory(dense: FakeDense) -> Callable[[Any], FakeDense]:
    """Deps.dense is a factory that receives the usage recorder. The fake embedder calls
    it once per request, like the real one (one llm_usage row per request)."""

    class Metered(FakeDense):
        def __init__(self, record: Any) -> None:
            self.record = record

        def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
            vectors = dense.embed(texts, feature)
            self.record(feature, "embedding-model", 10 * len(texts))
            return vectors

    return Metered

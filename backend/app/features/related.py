"""Related papers for the paper page: "similar", "cites this" and "cited by".

- similar: the nearest neighbours of the paper's abstract vector in Qdrant. The abstract
  chunk (ord 0) doubles as the paper-level vector, so the query passes that point's id
  and Qdrant uses its stored dense vector: no embedding request, no cost. Only other
  papers' abstract points are searched, so each similar paper appears once.
- cites: the paper's references from `citations` (filled by enrichment from Semantic
  Scholar). A referenced paper we do not hold has no title here, only its arXiv id.
- cited by: the papers we hold whose references include this paper (`citations` can
  only hold citing papers we hold; docs/adr/0008).

Everything returned is shown as data; links go to the arXiv abstract page.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Citation, Paper
from app.ingest.chunk import ABSTRACT_ORD, chunk_id
from app.ingest.index import DENSE_VECTOR

SimilarStatus = Literal["ok", "not_indexed", "unavailable"]


@dataclass(frozen=True)
class RelatedPaper:
    arxiv_id: str
    title: str | None  # None: a referenced paper we do not hold
    published_at: datetime | None
    held: bool
    score: float | None = None  # cosine similarity, for "similar" only


def similar_papers(
    client: QdrantClient, collection: str, arxiv_id: str, version: int, limit: int
) -> tuple[SimilarStatus, list[RelatedPaper]]:
    """Papers whose abstract vector is nearest to this paper's. 'not_indexed' when the
    paper's own abstract point is missing (nothing to compare with)."""
    point = str(chunk_id(arxiv_id, version, ABSTRACT_ORD))
    if not client.retrieve(collection, ids=[point], with_payload=False):
        return "not_indexed", []
    response = client.query_points(
        collection,
        query=point,  # a point id: Qdrant searches with that point's stored vector
        using=DENSE_VECTOR,
        query_filter=m.Filter(
            must=[m.FieldCondition(key="chunk_type", match=m.MatchValue(value="abstract"))],
            must_not=[m.FieldCondition(key="arxiv_id", match=m.MatchValue(value=arxiv_id))],
        ),
        limit=limit,
        with_payload=["arxiv_id", "title", "published_at"],
    )
    papers = []
    for hit in response.points:
        payload = hit.payload or {}
        published = payload.get("published_at")
        papers.append(
            RelatedPaper(
                arxiv_id=str(payload.get("arxiv_id")),
                title=payload.get("title"),
                published_at=datetime.fromisoformat(published) if published else None,
                held=True,
                score=hit.score,
            )
        )
    return "ok", papers


def cites(session: Session, arxiv_id: str, limit: int) -> list[RelatedPaper]:
    """The papers this one cites: held ones first (newest first), then the others by id."""
    rows = session.execute(
        select(Citation.cited_id, Paper.title, Paper.published_at)
        .outerjoin(Paper, Paper.arxiv_id == Citation.cited_id)
        .where(Citation.citing_id == arxiv_id)
        .order_by(Paper.published_at.desc().nulls_last(), Citation.cited_id)
        .limit(limit)
    ).all()
    return [
        RelatedPaper(arxiv_id=cited, title=title, published_at=published, held=title is not None)
        for cited, title, published in rows
    ]


def cited_by(session: Session, arxiv_id: str, limit: int) -> list[RelatedPaper]:
    """The papers we hold that cite this one, newest first."""
    rows = session.execute(
        select(Paper.arxiv_id, Paper.title, Paper.published_at)
        .join(Citation, Citation.citing_id == Paper.arxiv_id)
        .where(Citation.cited_id == arxiv_id)
        .order_by(Paper.published_at.desc(), Paper.arxiv_id)
        .limit(limit)
    ).all()
    return [
        RelatedPaper(arxiv_id=citing, title=title, published_at=published, held=True)
        for citing, title, published in rows
    ]

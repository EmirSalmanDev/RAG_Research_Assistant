"""The paper facts compare and timeline need from Postgres (read-only, no cost).

`has_fulltext` is true when the paper has at least one full-text `section` chunk, i.e.
when a citation can point into the paper's body and not only to its abstract. It is
derived from the chunks themselves, not from `papers.fulltext_status`, so it states what
the evidence actually is.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import exists, select

from app.db.models import Chunk, Paper
from app.db.session import session_scope


@dataclass(frozen=True)
class PaperFacts:
    arxiv_id: str
    title: str
    abstract: str
    published_at: datetime
    citation_count: int | None
    has_fulltext: bool


class FactsStore(Protocol):
    def get_facts(self, arxiv_ids: Sequence[str]) -> dict[str, PaperFacts]:
        """The facts of the papers that exist (an unknown id is simply missing)."""
        ...


def _has_section_chunks() -> object:
    return exists().where(Chunk.arxiv_id == Paper.arxiv_id, Chunk.chunk_type == "section")


class SqlFactsStore:
    def get_facts(self, arxiv_ids: Sequence[str]) -> dict[str, PaperFacts]:
        if not arxiv_ids:
            return {}
        with session_scope() as session:
            rows = session.execute(
                select(Paper, _has_section_chunks().label("fulltext")).where(  # type: ignore[attr-defined]
                    Paper.arxiv_id.in_(list(arxiv_ids))
                )
            ).all()
            return {
                paper.arxiv_id: PaperFacts(
                    arxiv_id=paper.arxiv_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    published_at=paper.published_at,
                    citation_count=paper.citation_count,
                    has_fulltext=bool(fulltext),
                )
                for paper, fulltext in rows
            }

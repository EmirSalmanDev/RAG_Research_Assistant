"""Ingest ONE paper's metadata into Postgres (the `make ingest-one` command).

Order: arXiv (required) -> Hugging Face page -> GitHub repo -> HF artifacts -> Semantic
Scholar. Only arXiv is required. The other four are the enrichment steps
(`app/ingest/enrich.py`): best-effort, a failure (or missing data) is logged and
recorded in `ingest_runs.errors`, and the run continues.

Idempotent (CLAUDE.md rule 5): every write is an upsert keyed on the natural primary
key, so running the command twice creates no duplicate rows.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import IngestRun, Paper
from app.ingest.enrich import EnrichClients, GitHubSource, HFSource, S2Source, describe, enrich_one
from app.sources.arxiv import ArxivPaper, validate_arxiv_id

logger = logging.getLogger(__name__)


class ArxivSource(Protocol):
    def get_paper(self, arxiv_id: str) -> ArxivPaper | None: ...


@dataclass
class Clients:
    """The four source clients (typed by Protocols, so tests can pass fakes)."""

    arxiv: ArxivSource
    hf: HFSource
    github: GitHubSource
    s2: S2Source


def default_clients() -> Clients:
    from app.sources.arxiv import ArxivClient
    from app.sources.github import GitHubClient
    from app.sources.hf import HFClient
    from app.sources.s2 import S2Client

    return Clients(arxiv=ArxivClient(), hf=HFClient(), github=GitHubClient(), s2=S2Client())


class IngestError(Exception):
    """The required step (arXiv metadata) failed, so nothing was stored for this paper."""


class IngestResult(BaseModel):
    arxiv_id: str
    counts: dict[str, int]
    errors: dict[str, str]


def ingest_one(session: Session, clients: Clients, raw_arxiv_id: str) -> IngestResult:
    arxiv_id = validate_arxiv_id(raw_arxiv_id)
    run = IngestRun(source="ingest-one", counts={}, errors={})
    session.add(run)
    session.flush()
    errors: dict[str, str] = {}
    counts: dict[str, int] = {"papers": 0, "repos": 0, "hf_artifacts": 0, "citations": 0}

    def finish() -> None:
        run.counts, run.errors, run.finished_at = counts, errors, datetime.now(UTC)
        session.commit()

    # --- 1. arXiv: the only required source -----------------------------------------
    try:
        paper = clients.arxiv.get_paper(arxiv_id)
        if paper is None:
            raise IngestError(f"arXiv has no paper {arxiv_id}")
        upsert_paper(session, paper)
        counts["papers"] = 1
    except Exception as err:
        session.rollback()  # drop the partial work, then record the failure on the run
        failed = IngestRun(source="ingest-one", counts=counts, errors={"arxiv": describe(err)})
        failed.finished_at = datetime.now(UTC)
        session.add(failed)
        session.commit()
        if isinstance(err, IngestError):
            raise
        raise IngestError(f"arXiv step failed: {err}") from err
    session.commit()

    # --- 2-5. Hugging Face page, GitHub repo, HF artifacts, Semantic Scholar ------------
    # (app/ingest/enrich.py; every one of them is best-effort)
    enriched = enrich_one(
        session, EnrichClients(hf=clients.hf, github=clients.github, s2=clients.s2), arxiv_id
    )
    counts.update(enriched.counts)
    errors.update(enriched.errors)

    finish()
    return IngestResult(arxiv_id=arxiv_id, counts=counts, errors=errors)


def primary_category_of(p: ArxivPaper) -> str | None:
    """arXiv's own primary category; if the feed left it empty, the first listed category
    (arXiv lists the primary first). None only if the paper has no categories at all."""
    return p.primary_category or (p.categories[0] if p.categories else None)


def upsert_paper(session: Session, p: ArxivPaper) -> None:
    values = {
        "arxiv_id": p.arxiv_id,
        "version": p.version,
        "title": p.title,
        "abstract": p.abstract,
        "authors": [{"name": name} for name in p.authors],
        "categories": p.categories,  # kept in arXiv's order
        "primary_category": primary_category_of(p),
        "published_at": p.published_at,
    }
    stmt = pg_insert(Paper).values(**values)
    # Only arXiv-owned columns are refreshed on conflict, so re-running never wipes
    # data that came from other sources (upvotes, github_url, citation_count, ...).
    stmt = stmt.on_conflict_do_update(
        index_elements=[Paper.arxiv_id],
        set_={**{k: stmt.excluded[k] for k in values if k != "arxiv_id"}, "updated_at": func.now()},
    )
    session.execute(stmt)

"""Ingest ONE paper's metadata into Postgres (the `make ingest-one` command).

Order: arXiv (required) -> Hugging Face page -> GitHub repo -> HF artifacts -> Semantic
Scholar. Only arXiv is required. Every other source is best-effort: a failure (or
missing data) is logged and recorded in `ingest_runs.errors`, and the run continues.

Idempotent (CLAUDE.md rule 5): every write is an upsert keyed on the natural primary
key, so running the command twice creates no duplicate rows.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import Citation, HFArtifact, IngestRun, Paper, Repo
from app.sources.arxiv import ArxivPaper, validate_arxiv_id
from app.sources.github import GitHubRepo, parse_github_url
from app.sources.hf import ArtifactKind, HFPaper
from app.sources.hf import HFArtifact as HFArtifactData
from app.sources.s2 import S2Paper, S2Result, S2Unavailable

logger = logging.getLogger(__name__)

README_MAX_CHARS = 2000  # stored as `repos.readme_summary` (raw, truncated; untrusted text)
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("model", "dataset", "space")


class ArxivSource(Protocol):
    def get_paper(self, arxiv_id: str) -> ArxivPaper | None: ...


class HFSource(Protocol):
    def paper(self, arxiv_id: str) -> HFPaper | None: ...
    def artifacts(self, arxiv_id: str, kind: ArtifactKind) -> list[HFArtifactData]: ...


class GitHubSource(Protocol):
    def repo(self, full_name: str) -> GitHubRepo | None: ...
    def readme(self, full_name: str) -> str | None: ...


class S2Source(Protocol):
    def batch(self, arxiv_ids: list[str]) -> dict[str, S2Result]: ...


@dataclass
class Clients:
    """The four source clients (typed by the Protocols above, so tests can pass fakes)."""

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
        _upsert_paper(session, paper)
        counts["papers"] = 1
    except Exception as err:
        session.rollback()  # drop the partial work, then record the failure on the run
        failed = IngestRun(source="ingest-one", counts=counts, errors={"arxiv": _describe(err)})
        failed.finished_at = datetime.now(UTC)
        session.add(failed)
        session.commit()
        if isinstance(err, IngestError):
            raise
        raise IngestError(f"arXiv step failed: {err}") from err
    session.commit()

    def optional(name: str, step: Callable[[], None]) -> None:
        """Run a best-effort step in its own savepoint; never let it fail the command."""
        try:
            with session.begin_nested():
                step()
        except Exception as err:
            logger.warning("ingest-one %s: step %s failed: %s", arxiv_id, name, err)
            errors[name] = _describe(err)

    # --- 2. Hugging Face paper page: upvotes + linked GitHub repo --------------------
    github_url: str | None = None

    def hf_page() -> None:
        nonlocal github_url
        hf_paper = clients.hf.paper(arxiv_id)
        if hf_paper is None:
            errors["hf"] = "paper not found on Hugging Face"
            return
        github_url = hf_paper.github_repo
        session.execute(
            update(Paper)
            .where(Paper.arxiv_id == arxiv_id)
            .values(hf_upvotes=hf_paper.upvotes, github_url=github_url or Paper.github_url)
        )

    optional("hf", hf_page)

    # --- 3. GitHub repo (only if HF told us about one) -------------------------------
    def github() -> None:
        full_name = parse_github_url(github_url)
        if full_name is None:
            return  # nothing to look up; not an error
        repo = clients.github.repo(full_name)
        if repo is None:
            errors["github"] = f"repo {full_name} not found"
            return
        readme = clients.github.readme(full_name)
        _upsert_repo(session, arxiv_id, repo, readme)
        counts["repos"] = 1

    optional("github", github)

    # --- 4. Hugging Face models / datasets / spaces that cite the paper ---------------
    for kind in ARTIFACT_KINDS:

        def artifacts(kind: ArtifactKind = kind) -> None:
            items = clients.hf.artifacts(arxiv_id, kind)
            _upsert_artifacts(session, arxiv_id, items)
            counts["hf_artifacts"] += len(items)

        optional(f"hf_{kind}s", artifacts)

    # --- 5. Semantic Scholar: citation count + references (via the batch endpoint) ----
    def s2() -> None:
        result = clients.s2.batch([arxiv_id])[arxiv_id]
        if isinstance(result, S2Unavailable):
            errors["s2"] = result.reason  # typed "unavailable": citation data stays missing
            return
        assert isinstance(result, S2Paper)
        session.execute(
            update(Paper)
            .where(Paper.arxiv_id == arxiv_id)
            .values(citation_count=result.citation_count)
        )
        stored = _upsert_citations(session, arxiv_id, result.references)
        counts["citations"] = stored
        # Only set when S2 answered: absent keys mean "unknown", 0 would mean "no references".
        counts["references_total"] = result.references_total
        counts["references_stored"] = stored
        # References S2 lists without an arXiv id can never join `papers`, so they are not stored.
        counts["references_dropped_no_arxiv_id"] = result.references_total - len(result.references)

    optional("s2", s2)

    finish()
    return IngestResult(arxiv_id=arxiv_id, counts=counts, errors=errors)


def _describe(err: Exception) -> str:
    return f"{type(err).__name__}: {err}"[:500]


def _upsert_paper(session: Session, p: ArxivPaper) -> None:
    values = {
        "arxiv_id": p.arxiv_id,
        "version": p.version,
        "title": p.title,
        "abstract": p.abstract,
        "authors": [{"name": name} for name in p.authors],
        "categories": p.categories,
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


def _upsert_repo(session: Session, arxiv_id: str, repo: GitHubRepo, readme: str | None) -> None:
    values = {
        "arxiv_id": arxiv_id,
        "full_name": repo.full_name,
        "stars": repo.stars,
        "last_push": repo.last_push,
        "license": repo.license,
        "readme_summary": readme[:README_MAX_CHARS] if readme else None,
    }
    stmt = pg_insert(Repo).values(**values)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Repo.arxiv_id, Repo.full_name],
            set_={k: stmt.excluded[k] for k in values if k not in ("arxiv_id", "full_name")},
        )
    )


def _upsert_artifacts(session: Session, arxiv_id: str, items: list[HFArtifactData]) -> None:
    for item in items:
        values = {
            "arxiv_id": arxiv_id,
            "kind": item.kind,
            "repo_id": item.repo_id,
            "downloads": item.downloads,
            "likes": item.likes,
            "pipeline_tag": item.pipeline_tag,
        }
        stmt = pg_insert(HFArtifact).values(**values)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[HFArtifact.arxiv_id, HFArtifact.kind, HFArtifact.repo_id],
                set_={k: stmt.excluded[k] for k in ("downloads", "likes", "pipeline_tag")},
            )
        )


def _upsert_citations(session: Session, arxiv_id: str, cited_ids: list[str]) -> int:
    rows = [
        {"citing_id": arxiv_id, "cited_id": c} for c in dict.fromkeys(cited_ids) if c != arxiv_id
    ]
    if rows:
        session.execute(pg_insert(Citation).values(rows).on_conflict_do_nothing())
    return len(rows)

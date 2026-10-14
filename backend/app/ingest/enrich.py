"""Enrichment: GitHub repo stats, Hugging Face models/datasets/Spaces and Semantic Scholar
citations for papers we already hold.

Two entry points share the same writes:
- `enrich_one()`: one paper, every source (used by `make ingest-one` and `make enrich-one`).
- `enrich_daily()`: the daily chain's step, for the papers added since its last run.

Sources and calls:
- GitHub: only the repo already linked to the paper (`papers.github_url`, from Hugging
  Face). One call per repo (`GET /repos/{owner}/{repo}`); READMEs are not fetched, so
  `repos.readme_summary` is never written here.
- Hugging Face Hub: models, datasets and Spaces tagged `arxiv:<id>` (one call per kind,
  at most 100 each).
- Semantic Scholar: citation count and references through the batch endpoint (one
  request per 500 papers). "Cited by" is not fetched: a citing paper we do not hold
  cannot be stored (`citations.citing_id` references `papers`), so "cited by" is read
  from the references of the papers we hold.

Every source is best-effort: a failure is recorded and the run carries on. Writes are
idempotent: a source that answered replaces that paper's rows for it (stale rows go),
a source that failed leaves the stored rows as they are.

Everything fetched here is untrusted third-party data (CLAUDE.md rule 8): it is stored
and shown as data only.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import httpx
from sqlalchemy import delete, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Citation, HFArtifact, IngestRun, Paper, Repo
from app.sources.github import GitHubRepo, parse_github_url
from app.sources.hf import ArtifactKind, HFPaper
from app.sources.hf import HFArtifact as HFArtifactData
from app.sources.http import RetryableError
from app.sources.s2 import S2Paper, S2Result, S2Unavailable

logger = logging.getLogger(__name__)

ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("model", "dataset", "space")
HF_ARTIFACT_CAP = 100  # the HF client asks for at most 100 per kind (`limit=100`)
DAILY_SOURCE = "enrich-daily"  # ingest_runs.source of the daily step (holds its cursor)
MAX_RECORDED_ERRORS = 20  # per run; the counts say how many there were in total


class HFSource(Protocol):
    def paper(self, arxiv_id: str) -> HFPaper | None: ...
    def artifacts(self, arxiv_id: str, kind: ArtifactKind) -> list[HFArtifactData]: ...


class GitHubSource(Protocol):
    def repo(self, full_name: str) -> GitHubRepo | None: ...


class S2Source(Protocol):
    def batch(self, arxiv_ids: list[str]) -> dict[str, S2Result]: ...


@dataclass
class EnrichClients:
    """The three enrichment clients (typed by the Protocols above, so tests pass fakes)."""

    hf: HFSource
    github: GitHubSource
    s2: S2Source


def default_enrich_clients() -> EnrichClients:
    from app.sources.github import GitHubClient
    from app.sources.hf import HFClient
    from app.sources.s2 import S2Client

    return EnrichClients(hf=HFClient(), github=GitHubClient(), s2=S2Client())


def describe(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"[:500]


def is_rate_limited(err: BaseException) -> bool:
    """429 after the client's own retries, or GitHub's 403 for an exhausted quota."""
    if isinstance(err, RetryableError):
        return err.status == 429
    if isinstance(err, httpx.HTTPStatusError):
        return err.response.status_code in (403, 429)
    return False


# --- writes (each one replaces what a source said about one paper) -----------------------------


def store_repo(session: Session, arxiv_id: str, repo: GitHubRepo) -> None:
    """The paper's repo: upsert it and drop any other repo row of this paper (the link
    changed). `readme_summary` is not touched (an older value is kept, never shown)."""
    session.execute(delete(Repo).where(Repo.arxiv_id == arxiv_id, Repo.full_name != repo.full_name))
    values = {
        "arxiv_id": arxiv_id,
        "full_name": repo.full_name,  # GitHub's spelling (it follows renames)
        "stars": repo.stars,
        "last_push": repo.last_push,
        "license": repo.license,
    }
    stmt = pg_insert(Repo).values(**values)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[Repo.arxiv_id, Repo.full_name],
            set_={k: stmt.excluded[k] for k in ("stars", "last_push", "license")},
        )
    )


def store_artifacts(
    session: Session, arxiv_id: str, kind: ArtifactKind, items: list[HFArtifactData]
) -> None:
    """Replace the paper's artifacts of one kind with `items`."""
    keep = [item.repo_id for item in items]
    session.execute(
        delete(HFArtifact).where(
            HFArtifact.arxiv_id == arxiv_id,
            HFArtifact.kind == kind,
            HFArtifact.repo_id.not_in(keep),
        )
    )
    for item in items:
        values = {
            "arxiv_id": arxiv_id,
            "kind": kind,
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


def store_s2(session: Session, result: S2Paper) -> int:
    """Citation count, and the paper's references replaced by S2's arXiv-id list.
    Returns how many reference rows the paper has now."""
    arxiv_id = result.arxiv_id
    session.execute(
        update(Paper).where(Paper.arxiv_id == arxiv_id).values(citation_count=result.citation_count)
    )
    cited = [c for c in dict.fromkeys(result.references) if c != arxiv_id]
    session.execute(
        delete(Citation).where(Citation.citing_id == arxiv_id, Citation.cited_id.not_in(cited))
    )
    if cited:
        rows = [{"citing_id": arxiv_id, "cited_id": c} for c in cited]
        session.execute(pg_insert(Citation).values(rows).on_conflict_do_nothing())
    return len(cited)


def s2_counts(result: S2Paper, stored: int) -> dict[str, int]:
    # Only set when S2 answered: absent keys mean "unknown", 0 would mean "no references".
    return {
        "citations": stored,
        "references_total": result.references_total,
        "references_stored": stored,
        # References S2 lists without an arXiv id can never join `papers`: not stored.
        "references_dropped_no_arxiv_id": result.references_total - len(result.references),
    }


# --- one paper, every source -------------------------------------------------------------------


@dataclass
class EnrichResult:
    counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def enrich_one(session: Session, clients: EnrichClients, arxiv_id: str) -> EnrichResult:
    """Every source for one paper we hold. Each step runs in its own savepoint, so a
    failing source never undoes another one. The caller commits."""
    out = EnrichResult(counts={"repos": 0, "hf_artifacts": 0, "citations": 0})
    errors = out.errors

    def optional(name: str, step: Callable[[], None]) -> None:
        try:
            with session.begin_nested():
                step()
        except Exception as err:
            logger.warning("enrich %s: step %s failed: %s", arxiv_id, name, err)
            errors[name] = describe(err)

    # 1. Hugging Face paper page: upvotes, linked GitHub repo, project page.
    def hf_page() -> None:
        hf_paper = clients.hf.paper(arxiv_id)
        if hf_paper is None:
            errors["hf"] = "paper not found on Hugging Face"
            return
        session.execute(
            update(Paper)
            .where(Paper.arxiv_id == arxiv_id)
            .values(
                hf_upvotes=hf_paper.upvotes,
                # untrusted third-party URLs (rule 8): stored and shown as data only
                github_url=hf_paper.github_repo or Paper.github_url,
                project_url=hf_paper.project_page or Paper.project_url,
            )
        )

    optional("hf", hf_page)

    # 2. GitHub: the linked repo (from this HF answer or stored earlier), one call.
    def github() -> None:
        full_name = parse_github_url(
            session.scalar(select(Paper.github_url).where(Paper.arxiv_id == arxiv_id))
        )
        if full_name is None:
            return  # nothing to look up; not an error
        repo = clients.github.repo(full_name)
        if repo is None:
            errors["github"] = f"repo {full_name} not found"
            return
        store_repo(session, arxiv_id, repo)
        out.counts["repos"] = 1

    optional("github", github)

    # 3. Hugging Face models / datasets / Spaces that cite the paper.
    for kind in ARTIFACT_KINDS:

        def artifacts(kind: ArtifactKind = kind) -> None:
            items = clients.hf.artifacts(arxiv_id, kind)
            store_artifacts(session, arxiv_id, kind, items)
            out.counts["hf_artifacts"] += len(items)

        optional(f"hf_{kind}s", artifacts)

    # 4. Semantic Scholar: citation count + references (batch endpoint, one id).
    def s2() -> None:
        result = clients.s2.batch([arxiv_id])[arxiv_id]
        if isinstance(result, S2Unavailable):
            errors["s2"] = result.reason  # citation data stays as it was
            return
        assert isinstance(result, S2Paper)
        out.counts.update(s2_counts(result, store_s2(session, result)))

    optional("s2", s2)
    return out


class NotHeldError(LookupError):
    """Enrichment was asked for a paper that is not in `papers`."""


def enrich_paper(session: Session, clients: EnrichClients, arxiv_id: str) -> EnrichResult:
    """`make enrich-one`: every source for one paper we hold, recorded as an 'enrich-one'
    run (its counts, and per-source errors in `errors`). Commits."""
    if session.get(Paper, arxiv_id) is None:
        raise NotHeldError(f"paper {arxiv_id} is not in the database (ingest it first)")
    run = IngestRun(source="enrich-one", counts={}, errors={})
    session.add(run)
    session.flush()
    result = enrich_one(session, clients, arxiv_id)
    run.counts, run.errors = {"arxiv_id": arxiv_id, **result.counts}, result.errors
    run.finished_at = datetime.now(UTC)
    session.commit()
    return result


# --- the daily step: many papers ---------------------------------------------------------------


@dataclass
class EnrichPlan:
    """Which papers each source looks at in one run (arXiv ids, oldest first)."""

    s2: list[str]
    github: list[str]
    artifacts: list[str]


def plan_daily(
    session: Session, since: datetime, recheck_since: datetime, max_papers: int
) -> EnrichPlan:
    """- S2: papers added since the last run, plus papers added within the recheck
      window that S2 has not answered for yet (`citation_count` is NULL; S2 often lists
      a paper only days after arXiv).
    - GitHub: papers in either set that have a GitHub link but no repo row yet (Hugging
      Face can link a repo days after the paper appears).
    - Hugging Face artifacts: papers added since the last run that Hugging Face lists
      (`hf_upvotes` set); 3 calls each, so they are not rechecked.
    Each list holds at most `max_papers` ids."""
    new = Paper.created_at > since
    recent = Paper.created_at >= recheck_since

    def ids(*where: Any) -> list[str]:
        query = select(Paper.arxiv_id).where(*where).order_by(Paper.created_at, Paper.arxiv_id)
        return list(session.scalars(query.limit(max_papers)))

    no_repo = ~exists().where(Repo.arxiv_id == Paper.arxiv_id)
    return EnrichPlan(
        s2=ids(or_(new, recent & Paper.citation_count.is_(None))),
        github=ids(or_(new, recent), Paper.github_url.is_not(None), no_repo),
        artifacts=ids(new, Paper.hf_upvotes.is_not(None)),
    )


class _RunLog:
    """Counts and the first few errors of one run; a source that rate-limits us is
    stopped for the rest of the run (its remaining papers wait for the next run)."""

    def __init__(self, counts: dict[str, Any]) -> None:
        self.counts = counts
        self.errors: list[str] = []
        self.stopped: set[str] = set()

    def add(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def failed(self, source: str, arxiv_id: str, err: BaseException) -> None:
        self.add(f"{source}_errors")
        if len(self.errors) < MAX_RECORDED_ERRORS:
            self.errors.append(f"{source} {arxiv_id}: {describe(err)}")
        if is_rate_limited(err):
            logger.warning("enrich: %s is rate limiting us; stopping it for this run", source)
            self.stopped.add(source)


def run_plan(session: Session, clients: EnrichClients, plan: EnrichPlan, log: _RunLog) -> None:
    # Semantic Scholar: one batch request per 500 papers (the client splits the list).
    # The client returns "unavailable" instead of raising; anything else it raises is
    # recorded like a per-paper error, so the other sources still run.
    results: dict[str, S2Result] = {}
    if plan.s2:
        try:
            results = clients.s2.batch(plan.s2)
        except Exception as err:
            log.failed("s2", "batch", err)
        for arxiv_id in plan.s2:
            result = results.get(arxiv_id)
            if not isinstance(result, S2Paper):
                log.add("s2_unavailable")
                continue
            with session.begin_nested():
                log.add("citations", store_s2(session, result))
            log.add("s2_answered")

    # GitHub: one call per linked repo.
    for arxiv_id in plan.github:
        if "github" in log.stopped:
            log.add("github_skipped")
            continue
        url = session.scalar(select(Paper.github_url).where(Paper.arxiv_id == arxiv_id))
        full_name = parse_github_url(url)
        if full_name is None:
            log.add("github_unparsable")
            continue
        try:
            with session.begin_nested():
                repo = clients.github.repo(full_name)
                if repo is None:
                    log.add("repos_not_found")
                    continue
                store_repo(session, arxiv_id, repo)
            log.add("repos")
        except Exception as err:
            log.failed("github", arxiv_id, err)

    # Hugging Face Hub: three calls per paper.
    for arxiv_id in plan.artifacts:
        for kind in ARTIFACT_KINDS:
            if "hf" in log.stopped:
                log.add("hf_skipped")
                continue
            try:
                with session.begin_nested():
                    items = clients.hf.artifacts(arxiv_id, kind)
                    store_artifacts(session, arxiv_id, kind, items)
                log.add(f"hf_{kind}s", len(items))
                if len(items) >= HF_ARTIFACT_CAP:
                    log.add("hf_capped")  # there may be more than we stored
            except Exception as err:
                log.failed("hf", arxiv_id, err)


def enrich_daily(
    session: Session, clients: EnrichClients, *, max_papers: int | None = None
) -> dict[str, Any]:
    """The daily step, recorded as an 'enrich-daily' run. Its cursor is the database
    time when the run started, so the next run picks up papers added after it."""
    # Imported here: daily.py imports paper.py, which imports this module.
    from app.ingest.daily import FIRST_RUN_LOOKBACK, last_cursor, recorded_run

    settings = get_settings()
    max_papers = max_papers if max_papers is not None else settings.enrich_max_papers
    now: datetime = session.execute(select(func.now())).scalar_one()
    cursor = last_cursor(session, sources=(DAILY_SOURCE,))
    since = cursor if cursor is not None else now - FIRST_RUN_LOOKBACK
    recheck_since = now - timedelta(days=settings.enrich_recheck_days)
    with recorded_run(session, DAILY_SOURCE) as counts:
        plan = plan_daily(session, since, recheck_since, max_papers)
        counts.update(
            since=since.isoformat(), cursor=now.isoformat(),
            planned_s2=len(plan.s2), planned_github=len(plan.github),
            planned_artifacts=len(plan.artifacts),
        )  # fmt: skip
        log = _RunLog(counts)
        run_plan(session, clients, plan, log)
        if log.errors:
            counts["error_samples"] = log.errors  # per-paper errors: the run itself succeeded
        counts["stopped"] = sorted(log.stopped)
    return counts

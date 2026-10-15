"""app/ingest/enrich.py against a scratch Postgres, with fake GitHub / Hugging Face /
Semantic Scholar clients (no network). The real clients are covered by their cassettes."""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Citation, HFArtifact, IngestRun, Paper, Repo
from app.ingest import enrich
from app.ingest.daily import last_cursor
from app.ingest.enrich import EnrichClients, NotHeldError, enrich_daily, enrich_one, enrich_paper
from app.sources.github import GitHubRepo
from app.sources.hf import ArtifactKind, HFPaper
from app.sources.hf import HFArtifact as HFArtifactData
from app.sources.http import RetryableError
from app.sources.s2 import S2Paper, S2Result, S2Unavailable

pytestmark = pytest.mark.integration

ID = "2601.00001"


class FakeHF:
    def __init__(self, items: dict[str, list[str]] | None = None, fail: Exception | None = None):
        self.items = items if items is not None else {"model": ["org/m1", "org/m2"]}
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def paper(self, arxiv_id: str) -> HFPaper | None:
        return HFPaper(id=arxiv_id, upvotes=4, githubRepo="https://github.com/a/b")

    def artifacts(self, arxiv_id: str, kind: ArtifactKind) -> list[HFArtifactData]:
        self.calls.append((arxiv_id, kind))
        if self.fail is not None:
            raise self.fail
        return [HFArtifactData(kind=kind, repo_id=r, downloads=3) for r in self.items.get(kind, [])]


class FakeGitHub:
    def __init__(self, name: str = "a/b", fail: Exception | None = None) -> None:
        self.name, self.fail = name, fail
        self.calls: list[str] = []

    def repo(self, full_name: str) -> GitHubRepo | None:
        self.calls.append(full_name)
        if self.fail is not None:
            raise self.fail
        return GitHubRepo(full_name=self.name, stargazers_count=11, license="MIT")


class FakeS2:
    def __init__(self, references: list[str] | None = None, unavailable: bool = False) -> None:
        self.references = references if references is not None else ["2001.00001", "2001.00002"]
        self.unavailable = unavailable
        self.calls: list[list[str]] = []

    def batch(self, arxiv_ids: list[str]) -> dict[str, S2Result]:
        self.calls.append(list(arxiv_ids))
        if self.unavailable:
            return {i: S2Unavailable(arxiv_id=i, reason="HTTP 429") for i in arxiv_ids}
        return {
            i: S2Paper(arxiv_id=i, citation_count=7, references=self.references) for i in arxiv_ids
        }


def _clients(**kw: Any) -> EnrichClients:
    return EnrichClients(
        hf=kw.get("hf") or FakeHF(), github=kw.get("github") or FakeGitHub(),
        s2=kw.get("s2") or FakeS2(),
    )  # fmt: skip


def _paper(s: Session, arxiv_id: str, created: datetime | None = None, **kw: Any) -> None:
    s.add(
        Paper(
            arxiv_id=arxiv_id, version=1, title=f"Invented {arxiv_id}", abstract="Invented.",
            authors=[], categories=["cs.CL"], published_at=datetime(2026, 1, 2, tzinfo=UTC),
            **({"created_at": created} if created else {}), **kw,
        )
    )  # fmt: skip
    s.commit()


def _count(s: Session, model: Any, *where: Any) -> int:
    return s.scalar(select(func.count()).select_from(model).where(*where)) or 0


# --- enrich_one / enrich_paper ---------------------------------------------------------------


def test_enrich_one_fills_every_table_with_one_github_call(migrated_session: Session) -> None:
    _paper(migrated_session, ID)
    github = FakeGitHub()
    result = enrich_one(migrated_session, _clients(github=github), ID)
    migrated_session.commit()
    assert result.errors == {}
    assert github.calls == ["a/b"]  # the repo only; no README call exists any more
    repo = migrated_session.get(Repo, (ID, "a/b"))
    assert repo is not None and (repo.stars, repo.license, repo.readme_summary) == (11, "MIT", None)
    assert _count(migrated_session, HFArtifact) == 2
    assert _count(migrated_session, Citation) == 2
    p = migrated_session.get(Paper, ID)
    assert p is not None and (p.hf_upvotes, p.citation_count) == (4, 7)
    assert result.counts == {
        "repos": 1, "hf_artifacts": 2, "citations": 2, "references_total": 2,
        "references_stored": 2, "references_dropped_no_arxiv_id": 0,
    }  # fmt: skip


def test_a_source_that_answers_replaces_stale_rows(migrated_session: Session) -> None:
    _paper(migrated_session, ID)
    enrich_one(migrated_session, _clients(), ID)
    migrated_session.commit()
    # Second run: the repo was renamed, one model and one reference are gone.
    clients = _clients(
        github=FakeGitHub(name="a/renamed"), hf=FakeHF({"model": ["org/m2"]}),
        s2=FakeS2(references=["2001.00002"]),
    )  # fmt: skip
    enrich_one(migrated_session, clients, ID)
    migrated_session.commit()
    assert list(migrated_session.scalars(select(Repo.full_name))) == ["a/renamed"]
    assert list(migrated_session.scalars(select(HFArtifact.repo_id))) == ["org/m2"]
    assert list(migrated_session.scalars(select(Citation.cited_id))) == ["2001.00002"]


def test_a_failing_source_keeps_the_stored_rows(migrated_session: Session) -> None:
    _paper(migrated_session, ID)
    enrich_one(migrated_session, _clients(), ID)
    migrated_session.commit()
    down = _clients(
        github=FakeGitHub(fail=RuntimeError("GitHub is down")),
        hf=FakeHF(fail=RuntimeError("HF is down")), s2=FakeS2(unavailable=True),
    )  # fmt: skip
    result = enrich_one(migrated_session, down, ID)
    migrated_session.commit()
    assert {"github", "hf_models", "hf_datasets", "hf_spaces", "s2"} <= set(result.errors)
    assert (_count(migrated_session, Repo), _count(migrated_session, HFArtifact)) == (1, 2)
    assert _count(migrated_session, Citation) == 2
    p = migrated_session.get(Paper, ID)
    assert p is not None and p.citation_count == 7


def test_enrich_one_uses_a_stored_github_link_when_hf_has_none(migrated_session: Session) -> None:
    _paper(migrated_session, ID, github_url="https://github.com/stored/repo")

    class NoHFPage(FakeHF):
        def paper(self, arxiv_id: str) -> HFPaper | None:
            return None

    github = FakeGitHub(name="stored/repo")
    result = enrich_one(migrated_session, _clients(hf=NoHFPage(), github=github), ID)
    assert github.calls == ["stored/repo"] and result.counts["repos"] == 1
    assert result.errors["hf"] == "paper not found on Hugging Face"


def test_enrich_paper_records_a_run_and_refuses_unknown_papers(migrated_session: Session) -> None:
    with pytest.raises(NotHeldError):
        enrich_paper(migrated_session, _clients(), ID)
    _paper(migrated_session, ID)
    enrich_paper(migrated_session, _clients(s2=FakeS2(unavailable=True)), ID)
    run = migrated_session.scalars(select(IngestRun)).one()
    assert run.source == "enrich-one" and run.finished_at is not None
    assert run.counts["arxiv_id"] == ID and run.counts["repos"] == 1
    assert run.errors == {"s2": "HTTP 429"}


# --- the daily step ---------------------------------------------------------------------------


def _db_now(s: Session) -> datetime:
    return s.execute(select(func.now())).scalar_one()


def test_daily_plan_picks_new_papers_and_cheap_rechecks(migrated_session: Session) -> None:
    s = migrated_session
    now = _db_now(s)
    s.rollback()  # end the transaction, so the papers below are added in a later one
    old = now - timedelta(days=30)
    recent = now - timedelta(days=3)
    _paper(s, "2601.00001", created=old, github_url="https://github.com/x/old")  # out of window
    # Recent, S2 never answered, linked repo not fetched yet:
    _paper(s, "2601.00002", created=recent, github_url="https://github.com/x/two", hf_upvotes=3)
    # Recent, already done: S2 answered, repo row exists:
    _paper(s, "2601.00003", created=recent, citation_count=0, github_url="https://github.com/x/3")
    s.add(Repo(arxiv_id="2601.00003", full_name="x/3"))
    s.commit()
    _paper(s, "2601.00004", hf_upvotes=1)  # new (created now), listed on HF
    _paper(s, "2601.00005", citation_count=5)  # new, not on HF, no link

    plan = enrich.plan_daily(s, since=now, recheck_since=now - timedelta(days=7), max_papers=10)
    assert plan.s2 == ["2601.00002", "2601.00004", "2601.00005"]
    assert plan.github == ["2601.00002"]
    assert plan.artifacts == ["2601.00004"]  # new AND listed on HF (3 calls each: no recheck)
    capped = enrich.plan_daily(s, since=now, recheck_since=now - timedelta(days=7), max_papers=1)
    assert (capped.s2, capped.artifacts) == (["2601.00002"], ["2601.00004"])


def test_enrich_daily_runs_once_per_paper_and_moves_its_cursor(migrated_session: Session) -> None:
    s = migrated_session
    _paper(s, "2601.00001", github_url="https://github.com/a/b", hf_upvotes=2)
    _paper(s, "2601.00002")
    s2, github, hf = FakeS2(), FakeGitHub(), FakeHF()
    counts = enrich_daily(s, _clients(s2=s2, github=github, hf=hf))
    assert s2.calls == [["2601.00001", "2601.00002"]]  # one batch request for both
    assert github.calls == ["a/b"]
    assert hf.calls == [("2601.00001", k) for k in ("model", "dataset", "space")]
    assert (counts["s2_answered"], counts["citations"], counts["repos"]) == (2, 4, 1)
    assert counts["hf_models"] == 2 and counts["stopped"] == []
    run = s.scalars(select(IngestRun).where(IngestRun.source == "enrich-daily")).one()
    assert run.errors == {} and run.counts["cursor"] == counts["cursor"]
    assert last_cursor(s, sources=("enrich-daily",)) == datetime.fromisoformat(counts["cursor"])
    assert last_cursor(s) is None  # the arXiv cursor is separate

    # Second run: nothing new, and both papers are done (S2 answered, repo stored).
    s2b, githubb, hfb = FakeS2(), FakeGitHub(), FakeHF()
    again = enrich_daily(s, _clients(s2=s2b, github=githubb, hf=hfb))
    assert (s2b.calls, githubb.calls, hfb.calls) == ([], [], [])
    assert (again["planned_s2"], again["planned_github"], again["planned_artifacts"]) == (0, 0, 0)


def test_rate_limited_sources_stop_for_the_rest_of_the_run(migrated_session: Session) -> None:
    s = migrated_session
    for n in (1, 2, 3):
        _paper(s, f"2601.0000{n}", github_url=f"https://github.com/a/r{n}", hf_upvotes=1)
    quota = httpx.HTTPStatusError(
        "403 rate limit",
        request=httpx.Request("GET", "https://api.github.com/repos/a/r1"),
        response=httpx.Response(403),
    )
    github = FakeGitHub(fail=quota)
    hf = FakeHF(fail=RetryableError("GET /api/models -> HTTP 429", status=429))
    counts = enrich_daily(s, _clients(github=github, hf=hf))
    assert github.calls == ["a/r1"] and len(hf.calls) == 1  # stopped after the first refusal
    assert counts["stopped"] == ["github", "hf"]
    assert (counts["github_skipped"], counts["hf_skipped"]) == (2, 8)
    assert len(counts["error_samples"]) == 2
    run = s.scalars(select(IngestRun)).one()
    assert run.errors == {}  # per-paper errors do not fail the run (the cursor moves on)
    assert counts["s2_answered"] == 3  # S2 still ran


def test_other_per_paper_errors_do_not_stop_a_source(migrated_session: Session) -> None:
    s = migrated_session
    for n in (1, 2):
        _paper(s, f"2601.0000{n}", github_url=f"https://github.com/a/r{n}")
    github = FakeGitHub(fail=RuntimeError("bad JSON"))
    counts = enrich_daily(s, _clients(github=github, s2=FakeS2(unavailable=True)))
    assert github.calls == ["a/r1", "a/r2"] and counts["github_errors"] == 2
    assert counts["stopped"] == [] and counts["s2_unavailable"] == 2
    assert _count(s, Paper, Paper.citation_count.is_not(None)) == 0

"""ingest_one against a scratch Postgres, with the four source clients replaced by fakes."""

from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Citation, HFArtifact, IngestRun, Paper, Repo
from app.ingest.paper import Clients, IngestError, ingest_one
from app.sources.arxiv import ArxivPaper
from app.sources.github import GitHubRepo
from app.sources.hf import ArtifactKind, HFPaper
from app.sources.hf import HFArtifact as HFArtifactData
from app.sources.s2 import S2Paper, S2Result, S2Unavailable

pytestmark = pytest.mark.integration

ID = "2601.00001"


class FakeArxiv:
    def __init__(self, paper: ArxivPaper | None | Exception) -> None:
        self.paper = paper

    def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        if isinstance(self.paper, Exception):
            raise self.paper
        return self.paper


class FakeHF:
    def __init__(self, paper: HFPaper | None = None, fail: bool = False) -> None:
        self.hf_paper, self.fail = paper, fail

    def paper(self, arxiv_id: str) -> HFPaper | None:
        if self.fail:
            raise RuntimeError("HF is down")
        return self.hf_paper

    def artifacts(self, arxiv_id: str, kind: ArtifactKind) -> list[HFArtifactData]:
        if self.fail:
            raise RuntimeError("HF is down")
        return [HFArtifactData(kind=kind, repo_id=f"org/{kind}-1", downloads=5, likes=1)]


class FakeGitHub:
    def __init__(self, repo: GitHubRepo | None) -> None:
        self._repo = repo

    def repo(self, full_name: str) -> GitHubRepo | None:
        return self._repo

    def readme(self, full_name: str) -> str | None:
        return "R" * 5000


class FakeS2:
    def __init__(self, result: S2Result) -> None:
        self.result = result

    def batch(self, arxiv_ids: list[str]) -> dict[str, S2Result]:
        return {arxiv_ids[0]: self.result}


def _arxiv_paper() -> ArxivPaper:
    return ArxivPaper(
        arxiv_id=ID, version=2, title="Invented", abstract="Invented abstract.",
        authors=["Ada Example"], categories=["cs.CL"], primary_category="cs.CL",
        published_at=datetime(2026, 1, 2, tzinfo=UTC), updated_at=datetime(2026, 1, 5, tzinfo=UTC),
    )  # fmt: skip


def _clients(
    *, hf: FakeHF | None = None, github: GitHubRepo | None = None, s2: S2Result | None = None
) -> Clients:
    return Clients(
        arxiv=FakeArxiv(_arxiv_paper()),
        hf=hf
        or FakeHF(
            HFPaper(
                id=ID, upvotes=9, githubRepo="https://github.com/a/b",
                projectPage="https://example.org/invented-project",
            )
        ),
        github=FakeGitHub(github or GitHubRepo(full_name="a/b", stargazers_count=7, license="MIT")),
        s2=FakeS2(
            s2 or S2Paper(arxiv_id=ID, citation_count=50, references=["2001.00001", "2001.00002"])
        ),
    )  # fmt: skip


def _counts(s: Session) -> dict[str, int]:
    return {
        t.__tablename__: s.scalar(select(func.count()).select_from(t)) or 0
        for t in (Paper, Repo, HFArtifact, Citation, IngestRun)
    }


def test_happy_path_fills_every_table(migrated_session: Session) -> None:
    result = ingest_one(migrated_session, _clients(), ID)
    assert result.errors == {}
    p = migrated_session.get(Paper, ID)
    assert p is not None
    assert (p.title, p.version, p.hf_upvotes, p.citation_count) == ("Invented", 2, 9, 50)
    assert p.github_url == "https://github.com/a/b" and p.categories == ["cs.CL"]
    assert p.project_url == "https://example.org/invented-project"
    assert p.authors == [{"name": "Ada Example"}]
    repo = migrated_session.get(Repo, (ID, "a/b"))
    assert repo is not None and repo.stars == 7 and repo.license == "MIT"
    assert repo.readme_summary is not None and len(repo.readme_summary) == 2000  # truncated
    assert migrated_session.scalar(select(func.count()).select_from(HFArtifact)) == 3
    assert _counts(migrated_session) == {
        "papers": 1, "repos": 1, "hf_artifacts": 3, "citations": 2, "ingest_runs": 1,
    }  # fmt: skip
    run = migrated_session.scalars(select(IngestRun)).one()
    assert run.finished_at is not None and run.counts["citations"] == 2 and run.errors == {}


def test_new_paper_gets_primary_category(migrated_session: Session) -> None:
    clients = _clients()
    paper = _arxiv_paper()
    paper.categories, paper.primary_category = ["cs.LG", "cs.CL"], "cs.LG"
    clients.arxiv = FakeArxiv(paper)
    ingest_one(migrated_session, clients, ID)
    p = migrated_session.get(Paper, ID)
    assert p is not None
    assert p.primary_category == "cs.LG" == p.categories[0]
    assert p.categories == ["cs.LG", "cs.CL"]  # order kept


def test_running_twice_creates_no_duplicates(migrated_session: Session) -> None:
    ingest_one(migrated_session, _clients(), ID)
    before = _counts(migrated_session)
    ingest_one(migrated_session, _clients(), ID)
    after = _counts(migrated_session)
    assert after.pop("ingest_runs") == before.pop("ingest_runs") + 1  # each run is logged
    assert after == before


def test_rerun_does_not_wipe_data_from_other_sources(migrated_session: Session) -> None:
    ingest_one(migrated_session, _clients(), ID)
    # Second run: HF and S2 are down. The arXiv upsert must keep upvotes/citation_count.
    down = _clients(hf=FakeHF(fail=True), s2=S2Unavailable(arxiv_id=ID, reason="rate limited"))
    result = ingest_one(migrated_session, down, ID)
    p = migrated_session.get(Paper, ID)
    assert p is not None and (p.hf_upvotes, p.citation_count) == (9, 50)
    assert set(result.errors) >= {"hf", "s2"}


def test_optional_source_failures_do_not_fail_and_are_recorded(migrated_session: Session) -> None:
    clients = _clients(hf=FakeHF(fail=True), s2=S2Unavailable(arxiv_id=ID, reason="HTTP 429"))
    result = ingest_one(migrated_session, clients, ID)
    assert migrated_session.get(Paper, ID) is not None  # arXiv data is stored
    assert result.errors["s2"] == "HTTP 429"
    assert "RuntimeError: HF is down" in result.errors["hf"]
    assert {"hf_models", "hf_datasets", "hf_spaces"} <= set(result.errors)
    run = migrated_session.scalars(select(IngestRun)).one()
    assert run.errors == result.errors
    p = migrated_session.get(Paper, ID)
    assert p is not None and p.citation_count is None and p.github_url is None


def test_github_not_found_is_recorded_and_skipped(migrated_session: Session) -> None:
    clients = _clients()
    clients.github = FakeGitHub(None)
    result = ingest_one(migrated_session, clients, ID)
    assert "not found" in result.errors["github"]
    assert migrated_session.scalar(select(func.count()).select_from(Repo)) == 0


def test_arxiv_failure_fails_the_command_but_is_logged(migrated_session: Session) -> None:
    clients = _clients()
    clients.arxiv = FakeArxiv(None)  # arXiv has no such paper
    with pytest.raises(IngestError):
        ingest_one(migrated_session, clients, ID)
    assert _counts(migrated_session)["papers"] == 0
    run = migrated_session.scalars(select(IngestRun)).one()
    assert "arxiv" in run.errors


def test_invalid_id_is_rejected_before_any_work(migrated_session: Session) -> None:
    with pytest.raises(ValueError):
        ingest_one(migrated_session, _clients(), "not-an-id/../x")
    assert _counts(migrated_session)["ingest_runs"] == 0


def test_version_suffix_is_stripped(migrated_session: Session) -> None:
    result = ingest_one(migrated_session, _clients(), f"{ID}v3")
    assert result.arxiv_id == ID


def test_reference_counts_are_recorded_in_the_run(migrated_session: Session) -> None:
    # S2 lists 6 references; 3 have an arXiv id (one of them twice) and 3 do not.
    s2 = S2Paper(
        arxiv_id=ID,
        citation_count=50,
        references=["2001.00001", "2001.00002", "2001.00002"],
        references_total=6,
    )
    result = ingest_one(migrated_session, _clients(s2=s2), ID)
    expected = {"references_total": 6, "references_stored": 2, "references_dropped_no_arxiv_id": 3}
    assert {k: result.counts[k] for k in expected} == expected
    run = migrated_session.scalars(select(IngestRun)).one()
    assert {k: run.counts[k] for k in expected} == expected  # persisted, not just returned
    assert result.counts["citations"] == 2  # existing key unchanged


def test_reference_counts_are_absent_when_s2_is_unavailable(migrated_session: Session) -> None:
    clients = _clients(s2=S2Unavailable(arxiv_id=ID, reason="HTTP 429"))
    result = ingest_one(migrated_session, clients, ID)
    assert not any(k.startswith("references_") for k in result.counts)

"""`python -m app.features.cli` (make digest) on a scratch database with fake vectors and
a fake embedder: a dry run writes and requests nothing; a real run builds the digest."""

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DigestEntry, Interest, Paper
from app.features import cli, digest
from tests.digest_fakes import FakeDense, FakeVectors, make_interest, make_paper

pytestmark = pytest.mark.integration


@pytest.fixture
def setup(migrated_session: Session, monkeypatch: pytest.MonkeyPatch) -> tuple[Session, FakeDense]:
    s = migrated_session
    now = datetime.now(UTC)
    make_interest(s, "Alpha topic")  # no vector yet
    make_paper(s, "2610.00001", age_days=1, now=now)
    make_paper(s, "2610.00002", age_days=2, now=now)
    vectors = FakeVectors({"2610.00001": [1.0, 0.0], "2610.00002": [0.5, 1.0]})
    dense = FakeDense()
    monkeypatch.setattr(digest, "QdrantAbstractVectors", lambda *a, **k: vectors)
    monkeypatch.setattr("app.ingest.embed.get_dense_embedder", lambda: dense)
    monkeypatch.setattr(cli, "close_qdrant", lambda: None)
    return s, dense


def test_dry_run_makes_no_request_and_writes_nothing(
    setup: tuple[Session, FakeDense], capsys: pytest.CaptureFixture[str]
) -> None:
    s, dense = setup
    assert cli.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out[: out.index("\n}\n") + 2])
    assert report["digest"]["status"] == "built" and report["digest"]["entries"] == 2
    assert report["fulltext_selection_would_pick"]["top_n"] == 20
    assert dense.calls == []  # no embedding request
    assert s.scalar(select(DigestEntry.id)) is None
    assert all(p.relevance_score is None for p in s.scalars(select(Paper)))
    assert "2610.00001" in out  # the ranked list is printed


def test_a_real_run_embeds_scores_builds_and_lists_the_selection(
    setup: tuple[Session, FakeDense], capsys: pytest.CaptureFixture[str]
) -> None:
    s, dense = setup
    assert cli.main(["--date", str(digest.today_in_zone())]) == 0
    out = capsys.readouterr().out
    report = json.loads(out[: out.index("\n}\n") + 2])
    assert report["interests_embedded"] == 1 and len(dense.calls) == 1
    assert report["score"]["papers"] == 2
    assert report["fulltext_selection_would_pick"]["arxiv_ids"] == ["2610.00001", "2610.00002"]
    s.expire_all()
    assert [e.arxiv_id for e in s.scalars(select(DigestEntry).order_by(DigestEntry.rank))] == [
        "2610.00001",
        "2610.00002",
    ]
    interest = s.scalar(select(Interest))
    assert interest is not None and interest.embedding == [1.0, 0.0]

    assert cli.main(["--no-embed"]) == 0  # a second run: the digest exists, nothing changes
    assert len(dense.calls) == 1
    assert "exists" in capsys.readouterr().out

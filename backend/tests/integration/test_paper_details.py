"""SqlPaperStore.details on a scratch database: analysis (or queued), repos and Hugging
Face artifacts for the paper page. Postgres only; all data is invented."""

from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.api.papers import SqlPaperStore
from app.db.models import AnalysisBatch, AnalysisBatchItem, HFArtifact, Paper, PaperAnalysis, Repo

pytestmark = pytest.mark.integration

ID = "2601.00001"


def _seed(s: Session) -> None:
    s.add(Paper(arxiv_id=ID, version=1, title="Invented", abstract="Invented.", authors=[],
                categories=["cs.CL"], published_at=datetime(2026, 1, 2, tzinfo=UTC)))  # fmt: skip
    s.flush()
    s.add(Repo(arxiv_id=ID, full_name="a/b", stars=7, license="MIT",
               readme_summary="Invented README text that must never be shown."))  # fmt: skip
    s.add_all([
        HFArtifact(arxiv_id=ID, kind="model", repo_id="org/small", downloads=5),
        HFArtifact(arxiv_id=ID, kind="model", repo_id="org/big", downloads=500, likes=3,
                   pipeline_tag="text-generation"),
        HFArtifact(arxiv_id=ID, kind="model", repo_id="org/unknown", downloads=None),
        HFArtifact(arxiv_id=ID, kind="dataset", repo_id="org/data", downloads=1),
        HFArtifact(arxiv_id=ID, kind="space", repo_id="org/demo"),
    ])  # fmt: skip
    s.commit()


def test_details_without_an_analysis(migrated_session: Session) -> None:
    _seed(migrated_session)
    out = SqlPaperStore().details(ID)
    assert out is not None and out.analysis_status == "none" and out.analysis is None
    assert [r.url for r in out.repos] == ["https://github.com/a/b"]
    assert "readme" not in out.repos[0].model_dump_json()  # never exposed
    assert [m.repo_id for m in out.models] == ["org/big", "org/small", "org/unknown"]
    assert out.models[0].url == "https://huggingface.co/org/big"
    assert out.datasets[0].url == "https://huggingface.co/datasets/org/data"
    assert out.spaces[0].url == "https://huggingface.co/spaces/org/demo"
    assert SqlPaperStore().details("9999.99999") is None


def test_details_queued_then_available(migrated_session: Session) -> None:
    s = migrated_session
    _seed(s)
    batch = AnalysisBatch(status="in_progress", model="m", prompt_version="analyze_v1",
                          request_count=1, errors={})  # fmt: skip
    s.add(batch)
    s.flush()
    s.add(AnalysisBatchItem(batch_id=batch.id, arxiv_id=ID, status="pending"))
    s.commit()
    out = SqlPaperStore().details(ID)
    assert out is not None and out.analysis_status == "queued"

    s.add(PaperAnalysis(arxiv_id=ID, tldr="Invented TL;DR.", contributions="One.\nTwo.\n",
                        method="M.", results="R.", limitations="L.", model="m",
                        prompt_version="analyze_v1", batch_id=batch.id))  # fmt: skip
    s.commit()
    out = SqlPaperStore().details(ID)
    assert out is not None and out.analysis_status == "available"  # an analysis wins
    assert out.analysis is not None and out.analysis.contributions == ["One.", "Two."]
    assert (out.analysis.prompt_version, out.analysis.tldr) == ("analyze_v1", "Invented TL;DR.")

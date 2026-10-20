"""Code paths that assume an arXiv id must skip an uploaded document (a `local-` id) without
error: enrichment, full-text selection, the digest and the read-only paper routes. Compare
and timeline with an upload are tested next to their other tests. All data is invented."""

import uuid
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, IngestRun, LibraryItem, Paper, PaperText
from app.features.digest import build_digest, window_papers
from app.ingest.chunk import abstract_chunk
from app.ingest.enrich import EnrichClients, enrich_one, enrich_paper, plan_daily
from app.ingest.fulltext import request_fulltext, select_for_fulltext
from app.ingest.index import save_chunks
from app.main import app
from tests.digest_fakes import NOW, FakeVectors, make_interest, make_paper

pytestmark = pytest.mark.integration

LOCAL = "local-0123456789ab"


class Untouchable:
    """A source client that fails the test if anything calls it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"an uploaded document must not reach a source: {name}")


CLIENTS = EnrichClients(hf=Untouchable(), github=Untouchable(), s2=Untouchable())


def _upload(s: Session, **kw: Any) -> Paper:
    paper = make_paper(s, LOCAL, title="An uploaded paper", **kw)
    save_chunks(s, LOCAL, [abstract_chunk(LOCAL, 1, paper.title, paper.abstract)])
    s.commit()
    return paper


def test_enrichment_of_an_upload_makes_no_call_and_is_not_an_error(
    migrated_session: Session,
) -> None:
    s = migrated_session
    _upload(s)
    result = enrich_one(s, CLIENTS, LOCAL)
    assert result.errors == {} and result.counts == {"repos": 0, "hf_artifacts": 0, "citations": 0}
    run = enrich_paper(s, CLIENTS, LOCAL)  # `make enrich-one ID=local-...`
    assert run.errors == {}
    row = s.scalars(select(IngestRun).where(IngestRun.source == "enrich-one")).one()
    assert row.errors == {} and row.counts["arxiv_id"] == LOCAL


def test_the_daily_enrichment_plan_leaves_uploads_out(migrated_session: Session) -> None:
    s = migrated_session
    _upload(s, upvotes=3)
    paper = make_paper(s, "2610.00001", upvotes=3)
    paper.github_url = "https://github.com/a/b"
    s.get(Paper, LOCAL).github_url = "https://github.com/c/d"  # type: ignore[union-attr]
    s.commit()
    plan = plan_daily(s, NOW - timedelta(days=10), NOW - timedelta(days=10), 100)
    assert plan.s2 == ["2610.00001"]
    assert plan.github == ["2610.00001"] and plan.artifacts == ["2610.00001"]


def test_full_text_selection_skips_uploads_even_in_the_library(
    migrated_session: Session,
) -> None:
    s = migrated_session
    upload = _upload(s)
    upload.fulltext_status = "requested"  # as a failed half-way upload might be
    upload.relevance_score = 0.99
    make_paper(s, "2610.00001").fulltext_status = "requested"
    s.add_all(
        [
            LibraryItem(arxiv_id=LOCAL, collection="T"),
            LibraryItem(arxiv_id="2610.00001", collection="T"),
        ]
    )
    s.commit()
    assert select_for_fulltext(s, top_n=5, now=NOW) == ["2610.00001"]
    request_fulltext(s, LOCAL)  # a manual request is a no-op for an upload
    s.refresh(upload)
    assert upload.fulltext_status == "requested"  # unchanged by the call


def test_an_upload_is_not_a_digest_candidate(migrated_session: Session) -> None:
    s = migrated_session
    _upload(s, age_days=1)  # published yesterday, inside the window
    make_paper(s, "2610.00001", age_days=1)
    make_interest(s, "Alpha", embedding=[1.0, 0.0])
    vectors = FakeVectors({LOCAL: [1.0, 0.0], "2610.00001": [1.0, 0.0]})
    assert [p.arxiv_id for p in window_papers(s, NOW)] == ["2610.00001"]
    result = build_digest(s, vectors, now=NOW)
    assert [e.arxiv_id for e in result.entries] == ["2610.00001"]
    assert vectors.calls and all(LOCAL not in [a for a, _ in call] for call in vectors.calls)


@pytest.fixture
async def api(migrated_session: Session) -> AsyncIterator[AsyncClient]:
    s = migrated_session
    _upload(s)
    s.add(Chunk(id=uuid.UUID(int=5), arxiv_id=LOCAL, chunk_type="section",
                section_path="2 Methods", ord=1, text="Invented.", token_count=2))  # fmt: skip
    s.add(PaperText(arxiv_id=LOCAL, version=1, raw_markdown="x", markdown="x", token_count=1))
    s.commit()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


async def test_the_read_routes_give_an_upload_no_arxiv_link(api: AsyncClient) -> None:
    paper = (await api.get(f"/papers/{LOCAL}")).json()
    assert paper["arxiv_id"] == LOCAL and paper["abs_url"] is None
    listing = (await api.get("/papers", params={"q": "uploaded"})).json()
    assert [p["arxiv_id"] for p in listing["papers"]] == [LOCAL]
    assert listing["papers"][0]["abs_url"] is None and listing["papers"][0]["has_fulltext"]
    chunks = (await api.get(f"/papers/{LOCAL}/chunks")).json()["chunks"]
    assert chunks and all(c["abs_url"] is None for c in chunks)
    chunk = (await api.get(f"/chunks/{chunks[0]['chunk_id']}")).json()
    assert chunk["abs_url"] is None and chunk["title"] == "An uploaded paper"
    # the other routes of the paper page do not break either
    assert (await api.get(f"/papers/{LOCAL}/details")).status_code == 200


async def test_the_old_papers_keep_their_arxiv_link(
    api: AsyncClient, migrated_session: Session
) -> None:
    make_paper(migrated_session, "2610.00009", title="A normal paper")
    body = (await api.get("/papers/2610.00009")).json()
    assert body["abs_url"] == "https://arxiv.org/abs/2610.00009"

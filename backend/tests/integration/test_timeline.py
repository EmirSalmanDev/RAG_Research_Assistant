"""The timeline on a scratch database and a scratch Qdrant collection: the real hybrid
`search` with filters inside Qdrant, the real facts store, the real ranking. OpenAI is
replaced (hand-made vectors, a scripted chat reply). All text is invented."""

import uuid
from datetime import UTC, date, datetime

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, Paper
from app.features.deps import ExploreDeps
from app.features.paper_facts import SqlFactsStore
from app.features.timeline import LineFill, TimelineLines, TimelineRequest, run_timeline
from app.ingest.embed import get_sparse_embedder
from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection
from tests.chat_fakes import fake_chat, parsed_reply
from tests.explore_fakes import FakeDense, dense_factory
from tests.tracing_fakes import RecordingTracer

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

# (arxiv_id, month, categories, citation_count, title word)
PAPERS = [
    ("2601.20001", 1, ["cs.CL"], 500, "alpha"),
    ("2603.20002", 3, ["cs.CL"], None, "beta"),
    ("2605.20003", 5, ["cs.LG"], 5, "gamma"),
]
VECTOR = [1.0, 0.0, 0.0, 0.0]


def _id(arxiv_id: str, name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{arxiv_id}/{name}")


@pytest.fixture
def stores(
    migrated_session: Session,
    qdrant: QdrantClient,
    scratch_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[tuple[uuid.UUID, str, str, str | None, str, datetime, list[str]]] = []
    for arxiv_id, month, categories, citations, word in PAPERS:
        published = datetime(2026, month, 10, tzinfo=UTC)
        migrated_session.add(
            Paper(
                arxiv_id=arxiv_id, title=f"{word} paper", abstract=f"Abstract about {word} topic.",
                authors=[], categories=categories, primary_category=categories[0],
                published_at=published, citation_count=citations,
            )
        )  # fmt: skip
        rows.append((_id(arxiv_id, "abs"), arxiv_id, "abstract", None,
                     f"Abstract about {word} topic.", published, categories))  # fmt: skip
    rows.append((_id(PAPERS[0][0], "sec"), PAPERS[0][0], "section", "3 Method",
                 "A section about the topic, closer to the query than any abstract.",
                 datetime(2026, 1, 10, tzinfo=UTC), ["cs.CL"]))  # fmt: skip
    migrated_session.flush()
    for i, (cid, arxiv_id, kind, section, text, _, _) in enumerate(rows):
        migrated_session.add(
            Chunk(id=cid, arxiv_id=arxiv_id, chunk_type=kind, section_path=section, ord=i,
                  text=text, token_count=8)
        )  # fmt: skip
    migrated_session.commit()
    ensure_collection(qdrant, scratch_collection, 4)
    sparse = get_sparse_embedder().embed([r[4] for r in rows])
    qdrant.upsert(
        scratch_collection,
        points=[
            m.PointStruct(
                id=str(cid), vector={DENSE_VECTOR: VECTOR, SPARSE_VECTOR: sv},
                payload={"arxiv_id": arxiv_id, "chunk_type": kind, "section_path": section,
                         "title": f"{arxiv_id} paper", "published_at": published.isoformat(),
                         "categories": cats, "text": text},
            )
            for (cid, arxiv_id, kind, section, text, published, cats), sv in zip(
                rows, sparse, strict=True
            )
        ],
        wait=True,
    )  # fmt: skip
    monkeypatch.setenv("QDRANT_COLLECTION", scratch_collection)
    get_settings.cache_clear()


def _deps(lines: TimelineLines) -> ExploreDeps:
    chat, _, _ = fake_chat([parsed_reply(lines)])
    return ExploreDeps(
        chat=chat, facts=SqlFactsStore(), tracer=RecordingTracer(),
        dense=dense_factory(FakeDense()), record_embedding=lambda *_: None,
    )  # fmt: skip


def test_each_paper_appears_once_with_its_abstract(stores: None) -> None:
    lines = TimelineLines(lines=[LineFill(arxiv_id=p[0], line=f"{p[4]} changed.") for p in PAPERS])
    out = run_timeline(TimelineRequest(topic="topic", key_papers=1), deps=_deps(lines))
    assert [p.arxiv_id for p in out.points] == [p[0] for p in PAPERS]  # oldest first, no dupes
    assert {p.chunk_id for p in out.points} == {_id(p[0], "abs") for p in PAPERS}  # abstracts
    (key,) = [p for p in out.points if p.is_key]
    assert key.key_score == max(p.key_score for p in out.points)
    assert key.line == f"{key.title.split()[0]} changed." and key.line_source == "model"
    assert [p.citation_count for p in out.points] == [500, None, 5]
    assert out.notes.known_citation_counts == 2 and not out.notes.narrow_range


def test_filters_run_inside_qdrant(stores: None) -> None:
    lines = TimelineLines(lines=[])
    out = run_timeline(
        TimelineRequest(topic="topic", date_from=date(2026, 2, 1), date_to=date(2026, 4, 30)),
        deps=_deps(lines),
    )
    assert [p.arxiv_id for p in out.points] == ["2603.20002"]
    out = run_timeline(TimelineRequest(topic="topic", categories=("cs.LG",)), deps=_deps(lines))
    assert [p.arxiv_id for p in out.points] == ["2605.20003"]
    # the only key paper has no usable line (none was returned): the abstract's first sentence
    assert out.points[0].line_source == "abstract_fallback"
    assert out.points[0].line == "Abstract about gamma topic."

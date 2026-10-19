"""Comparison end to end on a scratch database and a scratch Qdrant collection.

The real `search` (hybrid, filtered to one paper), the real facts store and the real
validation run; only OpenAI is replaced (hand-made 4-dimensional vectors, a scripted
chat reply). The real `chunks` collection is checked to be untouched. All text is invented.
"""

import uuid
from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, Paper
from app.features.compare import CellFill, CompareDeps, CompareFill, PaperFill, run_compare
from app.features.paper_facts import SqlFactsStore
from app.ingest.embed import get_sparse_embedder
from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection
from tests.chat_fakes import fake_chat, parsed_reply
from tests.explore_fakes import FakeDense, dense_factory
from tests.tracing_fakes import RecordingTracer

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

FULL, ABSTRACT_ONLY = "2601.10001", "2601.10002"
PUBLISHED = datetime(2026, 1, 10, tzinfo=UTC)
# (arxiv_id, ord, chunk_type, section_path, dense vector, text)
CHUNKS = [
    (FULL, 0, "abstract", None, [1, 0, 0, 0], "We train a small transformer on invented text."),
    (FULL, 1, "section", "3 Method", [0, 1, 0, 0], "The method freezes weights, trains adapters."),
    (FULL, 2, "section", "4 Data", [0, 0, 1, 0], "We use the invented corpus Foo."),
    (FULL, 3, "analysis", None, [0, 1, 0, 0], "Generated summary: the method trains adapters."),
    (ABSTRACT_ONLY, 0, "abstract", None, [0, 1, 0, 0], "We study adapters on another task."),
]  # fmt: skip


def _id(arxiv_id: str, ord_: int) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{arxiv_id}/{ord_}")


@pytest.fixture
def stores(
    migrated_session: Session,
    qdrant: QdrantClient,
    scratch_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for arxiv_id in (FULL, ABSTRACT_ONLY):
        migrated_session.add(
            Paper(
                arxiv_id=arxiv_id,
                title=f"Invented {arxiv_id}",
                abstract="Invented abstract.",
                authors=[],
                categories=["cs.CL"],
                primary_category="cs.CL",
                published_at=PUBLISHED,
            )
        )
    migrated_session.flush()
    for arxiv_id, ord_, kind, section, _, text in CHUNKS:
        migrated_session.add(
            Chunk(
                id=_id(arxiv_id, ord_), arxiv_id=arxiv_id, chunk_type=kind,
                section_path=section, ord=ord_, text=text, token_count=10,
            )
        )  # fmt: skip
    migrated_session.commit()
    ensure_collection(qdrant, scratch_collection, 4)
    sparse = get_sparse_embedder().embed([c[-1] for c in CHUNKS])
    qdrant.upsert(
        scratch_collection,
        points=[
            m.PointStruct(
                id=str(_id(arxiv_id, ord_)),
                vector={DENSE_VECTOR: [float(x) for x in vec], SPARSE_VECTOR: sv},
                payload={
                    "arxiv_id": arxiv_id, "chunk_type": kind, "section_path": section,
                    "title": f"Invented {arxiv_id}", "published_at": PUBLISHED.isoformat(),
                    "categories": ["cs.CL"], "text": text,
                },
            )
            for (arxiv_id, ord_, kind, section, vec, text), sv in zip(CHUNKS, sparse, strict=True)
        ],
        wait=True,
    )  # fmt: skip
    monkeypatch.setenv("QDRANT_COLLECTION", scratch_collection)
    get_settings.cache_clear()


def test_every_citation_resolves_to_a_chunk_of_its_own_paper(
    stores: None, migrated_session: Session
) -> None:
    # Sources are numbered paper by paper: FULL gets 1..3 (abstract + 2 sections; the
    # analysis chunk is excluded), ABSTRACT_ONLY gets 4.
    fill = CompareFill(
        papers=[
            PaperFill(arxiv_id=FULL, cells=[
                CellFill(dimension="method", text="Trains adapters.", source_numbers=[1, 2, 3]),
                CellFill(dimension="data", text="Uses the corpus Foo.", source_numbers=[1, 2, 3]),
            ]),
            PaperFill(arxiv_id=ABSTRACT_ONLY, cells=[
                CellFill(dimension="method", text="Studies adapters.", source_numbers=[4]),
                # source 1 belongs to the other paper
                CellFill(dimension="data", text="Uses corpus Foo.", source_numbers=[1]),
            ]),
        ]
    )  # fmt: skip
    chat, _, usage_rows = fake_chat([parsed_reply(fill)])
    deps = CompareDeps(
        chat=chat, facts=SqlFactsStore(), tracer=RecordingTracer(),
        dense=dense_factory(FakeDense()), record_embedding=lambda *_: None,
    )  # fmt: skip
    result = run_compare([FULL, ABSTRACT_ONLY], ["method", "data"], deps=deps)

    assert [r.has_fulltext for r in result.rows] == [True, False]
    assert {s.chunk_type for s in result.sources} == {"abstract", "section"}  # no analysis
    by_n = {s.n: s for s in result.sources}
    cited = [(row.arxiv_id, c) for row in result.rows for c in row.cells if c.status == "cited"]
    assert len(cited) == 3
    for arxiv_id, cell in cited:
        for citation in cell.citations:
            chunk = migrated_session.get(Chunk, citation.chunk_id)
            assert chunk is not None and chunk.arxiv_id == arxiv_id
            assert chunk.chunk_type == citation.chunk_type != "analysis"
            assert by_n[citation.n].chunk_id == citation.chunk_id
    # The abstract-only paper's "data" cell cited another paper's source: not shown.
    assert result.rows[1].cells[1].status == "not_stated"
    assert result.notes.invalid_citations_dropped == 1
    assert len(usage_rows) == 1

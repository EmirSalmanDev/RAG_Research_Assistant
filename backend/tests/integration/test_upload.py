"""PDF upload on a scratch database and a scratch Qdrant collection, with fake embedders (no
OpenAI) and PDFs generated from invented text. The real collection stays untouched."""

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, LibraryItem, Paper, PaperText
from app.ids import abs_url
from app.ingest.chunk import chunk_id
from app.ingest.index import ensure_collection
from app.ingest.upload import (
    EMBED_FEATURE,
    EmptyUpload,
    InvalidMetadata,
    NotAnUpload,
    NotAPdf,
    NoTextLayer,
    TooLarge,
    UploadBusy,
    UploadDeps,
    UploadNotFound,
    abstract_from_body,
    delete_upload,
    ingest_pdf,
    local_id_for,
    upload_slot,
)
from tests.integration.test_indexing import DIM, FakeDense, FakeSparse
from tests.pdf_fixtures import make_blank_pdf, make_pdf

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]


@pytest.fixture
def collection(qdrant: QdrantClient, scratch_collection: str) -> str:
    ensure_collection(qdrant, scratch_collection, DIM)
    return scratch_collection


class World:
    """The fakes of one test: embedders, a usage log and a parse counter."""

    def __init__(self, qdrant: QdrantClient, collection: str) -> None:
        self.dense = FakeDense()
        self.usage: list[tuple[str, str, int]] = []
        self.parsed = 0
        self.collection = collection
        from app.ingest.fulltext import pdf_to_markdown

        def parse(pdf: bytes) -> str:
            self.parsed += 1
            return pdf_to_markdown(pdf)  # the real parser: PyMuPDF4LLM on a generated PDF

        def metered(record: Callable[[str, str, int], None]) -> FakeDense:
            # one llm_usage row per request, like the real embedder
            outer = self

            class Metered(FakeDense):
                def embed(self, texts, feature):  # type: ignore[no-untyped-def]
                    vectors = outer.dense.embed(texts, feature)
                    record(feature, "embedding-model", 7 * len(texts))
                    return vectors

            return Metered()

        self.deps = UploadDeps(
            dense=metered,
            sparse=lambda: FakeSparse(),
            qdrant=lambda: qdrant,
            qdrant_collection=collection,
            parse=parse,
            record_embedding=lambda feature, model, tokens: self.usage.append(
                (feature, model, tokens)
            ),
        )


@pytest.fixture
def world(qdrant: QdrantClient, collection: str) -> World:
    return World(qdrant, collection)


def _points(qdrant: QdrantClient, collection: str, arxiv_id: str) -> int:
    f = m.Filter(must=[m.FieldCondition(key="arxiv_id", match=m.MatchValue(value=arxiv_id))])
    return qdrant.count(collection, count_filter=f, exact=True).count


def test_an_upload_goes_through_the_same_steps_and_is_searchable(
    migrated_session: Session, qdrant: QdrantClient, world: World
) -> None:
    pdf = make_pdf(3)
    result = ingest_pdf(
        migrated_session, pdf, title="  An Invented  Journal Paper ", year=2019, deps=world.deps
    )
    arxiv_id = local_id_for(pdf)
    assert result.arxiv_id == arxiv_id and arxiv_id.startswith("local-") and len(arxiv_id) == 18
    assert result.created and result.title == "An Invented Journal Paper" and result.year == 2019

    paper = migrated_session.get(Paper, arxiv_id)
    assert paper is not None
    assert (paper.version, paper.fulltext_source, paper.fulltext_status) == (1, "upload", "indexed")
    assert paper.categories == [] and paper.authors == [] and paper.github_url is None
    assert paper.published_at.year == 2019 and paper.primary_category is None
    text = migrated_session.get(PaperText, arxiv_id)
    assert text is not None and "Invented section" in text.markdown  # parsed text is kept
    assert abs_url(arxiv_id) is None

    rows = migrated_session.scalars(select(Chunk).where(Chunk.arxiv_id == arxiv_id)).all()
    assert result.chunks["abstract"] == 1 and result.chunks["section"] >= 1
    assert {r.ord: r.id for r in rows} == {r.ord: chunk_id(arxiv_id, 1, r.ord) for r in rows}
    assert _points(qdrant, world.collection, arxiv_id) == len(rows) == result.embedded
    # the abstract chunk is the start of the body (no abstract was given)
    abstract = next(r for r in rows if r.chunk_type == "abstract")
    assert abstract.text == paper.abstract and "invented sentence" in abstract.text
    assert len(paper.abstract) <= get_settings().upload_abstract_chars + 1
    # embeddings are filed under their own feature name, and logged
    assert {feature for feature, _ in [(c[0], c[1]) for c in world.dense.calls]} == {EMBED_FEATURE}
    assert {u[0] for u in world.usage} == {EMBED_FEATURE} and result.spend.embedding_requests >= 1
    # the payload makes it findable like any paper, with no category
    point = qdrant.retrieve(world.collection, ids=[str(abstract.id)], with_payload=True)[0]
    assert point.payload is not None
    assert point.payload["arxiv_id"] == arxiv_id and point.payload["categories"] == []
    assert point.payload["title"] == "An Invented Journal Paper"


def test_the_same_bytes_again_change_nothing_and_embed_nothing(
    migrated_session: Session, qdrant: QdrantClient, world: World
) -> None:
    pdf = make_pdf(2)
    first = ingest_pdf(migrated_session, pdf, title="First title", deps=world.deps)
    calls, usage, parsed = len(world.dense.calls), len(world.usage), world.parsed
    again = ingest_pdf(migrated_session, pdf, title="A different title", deps=world.deps)
    assert not again.created and again.arxiv_id == first.arxiv_id
    assert again.title == "First title"  # the stored record wins
    assert again.embedded == 0 and again.spend.embedding_requests == 0
    assert len(world.dense.calls) == calls and len(world.usage) == usage
    assert world.parsed == parsed  # not parsed again
    assert again.chunks == first.chunks
    assert migrated_session.scalar(select(func.count()).select_from(Paper)) == 1


def test_a_given_abstract_is_the_abstract_chunk(migrated_session: Session, world: World) -> None:
    pdf = make_pdf(2, seed="b")
    result = ingest_pdf(
        migrated_session, pdf, title="T", abstract="  An invented\nabstract. ", deps=world.deps
    )
    chunk = migrated_session.scalars(
        select(Chunk).where(Chunk.arxiv_id == result.arxiv_id, Chunk.chunk_type == "abstract")
    ).one()
    assert chunk.text == "An invented abstract."


def test_the_year_defaults_to_the_upload_year(migrated_session: Session, world: World) -> None:
    result = ingest_pdf(migrated_session, make_pdf(2, "c"), title="T", deps=world.deps)
    assert result.year == datetime.now(UTC).year


def test_an_upload_can_join_a_collection(migrated_session: Session, world: World) -> None:
    pdf = make_pdf(2, "d")
    result = ingest_pdf(migrated_session, pdf, title="T", collection="Thesis", deps=world.deps)
    items = migrated_session.scalars(select(LibraryItem)).all()
    assert [(i.arxiv_id, i.collection, i.status) for i in items] == [
        (result.arxiv_id, "Thesis", "to_read")
    ]
    ingest_pdf(migrated_session, pdf, title="T", collection="Thesis", deps=world.deps)
    assert migrated_session.scalar(select(func.count()).select_from(LibraryItem)) == 1


def test_bad_uploads_are_refused_before_parsing_or_spending(
    migrated_session: Session, world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(EmptyUpload):
        ingest_pdf(migrated_session, b"", title="T", deps=world.deps)
    with pytest.raises(NotAPdf):
        ingest_pdf(migrated_session, b"GIF89a" + b"x" * 100, title="T", deps=world.deps)
    with pytest.raises(NotAPdf):
        ingest_pdf(migrated_session, b" %PDF-1.7 starts with a space", title="T", deps=world.deps)
    monkeypatch.setenv("UPLOAD_MAX_BYTES", "1000")
    get_settings.cache_clear()
    with pytest.raises(TooLarge):
        ingest_pdf(migrated_session, make_pdf(2, "e"), title="T", deps=world.deps)
    monkeypatch.delenv("UPLOAD_MAX_BYTES")
    get_settings.cache_clear()
    for bad in ({"title": ""}, {"title": "   "}, {"title": "x" * 301}, {"title": "T", "year": 1800},
                {"title": "T", "year": 2200}, {"title": "T", "abstract": "a" * 3001}):  # fmt: skip
        with pytest.raises(InvalidMetadata):
            ingest_pdf(migrated_session, make_pdf(2, "f"), deps=world.deps, **bad)
    assert world.parsed == 0 and world.dense.calls == [] and world.usage == []
    assert migrated_session.scalar(select(func.count()).select_from(Paper)) == 0


def test_a_pdf_without_a_text_layer_is_refused_and_stores_nothing(
    migrated_session: Session, world: World
) -> None:
    with pytest.raises(NoTextLayer):
        ingest_pdf(migrated_session, make_blank_pdf(), title="A scan", deps=world.deps)
    assert world.dense.calls == [] and world.usage == []
    assert migrated_session.scalar(select(func.count()).select_from(Paper)) == 0


def test_an_upload_that_stopped_at_the_embeddings_is_finished_by_uploading_again(
    migrated_session: Session, qdrant: QdrantClient, world: World
) -> None:
    pdf = make_pdf(2, "g")
    healthy = world.deps.dense

    class Down:
        def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
            raise RuntimeError("the embeddings API is down")

    world.deps.dense = lambda record: Down()
    with pytest.raises(RuntimeError, match="down"):
        ingest_pdf(migrated_session, pdf, title="T", deps=world.deps)
    arxiv_id = local_id_for(pdf)
    paper = migrated_session.get(Paper, arxiv_id)
    assert paper is not None and paper.fulltext_status == "fetched"  # text stored, not indexed
    assert _points(qdrant, world.collection, arxiv_id) == 0
    parsed = world.parsed

    world.deps.dense = healthy
    result = ingest_pdf(migrated_session, pdf, title="T", deps=world.deps)
    assert result.embedded > 0 and world.parsed == parsed  # the stored text was reused
    migrated_session.refresh(paper)
    assert paper.fulltext_status == "indexed"
    assert _points(qdrant, world.collection, arxiv_id) == result.embedded


def test_uploads_run_one_at_a_time() -> None:
    with upload_slot():
        with pytest.raises(UploadBusy), upload_slot():
            pass
    with upload_slot():  # released again
        pass


def test_the_abstract_from_the_body_is_cut_at_a_word() -> None:
    body = "# Heading\n\n" + "word " * 1000
    cut = abstract_from_body(body, 100)
    assert len(cut) <= 101 and cut.endswith("…") and not cut.startswith("#")
    assert abstract_from_body("short  text", 100) == "short text"


# --- delete --------------------------------------------------------------------------------------


def test_deleting_an_upload_removes_everything_of_it_and_nothing_else(
    migrated_session: Session, qdrant: QdrantClient, world: World
) -> None:
    keep = ingest_pdf(migrated_session, make_pdf(2, "keep"), title="Keep", deps=world.deps)
    gone = ingest_pdf(
        migrated_session, make_pdf(3, "gone"), title="Gone", collection="Thesis", deps=world.deps
    )
    kept_points = _points(qdrant, world.collection, keep.arxiv_id)
    result = delete_upload(
        migrated_session, gone.arxiv_id, client=qdrant, collection=world.collection
    )
    assert (result.chunks, result.points, result.library_items) == (
        sum(gone.chunks.values()), gone.embedded, 1,
    )  # fmt: skip
    assert migrated_session.get(Paper, gone.arxiv_id) is None
    assert migrated_session.get(PaperText, gone.arxiv_id) is None
    assert migrated_session.scalar(select(func.count()).where(Chunk.arxiv_id == gone.arxiv_id)) == 0
    assert migrated_session.scalar(select(func.count()).select_from(LibraryItem)) == 0
    assert _points(qdrant, world.collection, gone.arxiv_id) == 0
    assert _points(qdrant, world.collection, keep.arxiv_id) == kept_points  # the other stays
    assert migrated_session.get(Paper, keep.arxiv_id) is not None


def test_only_uploads_can_be_deleted(
    migrated_session: Session, qdrant: QdrantClient, world: World
) -> None:
    migrated_session.add(
        Paper(
            arxiv_id="2601.00001", title="T", abstract="A.", authors=[], categories=[],
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )  # fmt: skip
    migrated_session.commit()
    with pytest.raises(NotAnUpload):
        delete_upload(migrated_session, "2601.00001", client=qdrant, collection=world.collection)
    with pytest.raises(UploadNotFound):
        delete_upload(
            migrated_session, "local-000000000000", client=qdrant, collection=world.collection
        )
    assert migrated_session.get(Paper, "2601.00001") is not None


def test_a_deleted_upload_can_be_uploaded_again(
    migrated_session: Session, qdrant: QdrantClient, world: World
) -> None:
    pdf = make_pdf(2, "again")
    first = ingest_pdf(migrated_session, pdf, title="T", deps=world.deps)
    delete_upload(migrated_session, first.arxiv_id, client=qdrant, collection=world.collection)
    second = ingest_pdf(migrated_session, pdf, title="T2", deps=world.deps)
    assert second.created and second.arxiv_id == first.arxiv_id and second.title == "T2"
    assert _points(qdrant, world.collection, second.arxiv_id) == second.embedded


def test_chunk_ids_are_the_uuid5_of_id_version_and_ord(
    migrated_session: Session, world: World
) -> None:
    result = ingest_pdf(migrated_session, make_pdf(2, "uuid"), title="T", deps=world.deps)
    ids = migrated_session.scalars(select(Chunk.id).where(Chunk.arxiv_id == result.arxiv_id)).all()
    assert all(isinstance(i, uuid.UUID) for i in ids)
    assert chunk_id(result.arxiv_id, 1, 0) in ids

"""Building index variants (eval/indexes.py) on a throwaway Postgres database and
`chunks_test_<random>` Qdrant collections. Every paper and vector is invented; OpenAI is a
fake. The real collection is checked to be untouched, and every variant collection a test
makes is deleted afterwards."""

import hashlib
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Paper, PaperText
from app.ingest.chunk import (
    ChunkSettings,
    abstract_chunk,
    analysis_chunks,
    chunk_paper,
)
from app.ingest.embed import get_sparse_embedder
from app.ingest.index import (
    DENSE_VECTOR,
    ensure_collection,
    save_chunks,
    upsert_analysis_chunks,
    upsert_paper_chunks,
)
from eval.indexes import FEATURE, IndexBuildError, build_index, plan_index
from eval.runner import index_embedder
from eval.scope import applied_settings
from eval.variants import find, resolve
from tests.eval_fakes import FakeOpenAIEmbeddings

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

FULL, ABSTRACT_ONLY = "2601.00011", "2601.00012"
SENTENCES = [
    f"Invented sentence {i} says that the adapter in layer {i % 7} keeps rank {i % 5 + 2} "
    "while the frozen weights carry the pretrained knowledge."
    for i in range(70)
]
MARKDOWN = "# Invented paper\n\n## 1 Method\n\n" + "\n\n".join(
    " ".join(SENTENCES[k : k + 5]) for k in range(0, 70, 5)
)


class HashDense:
    """Invented dense vectors of the size the current settings ask for; records texts."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        self.texts += texts
        dim = get_settings().embedding_dim
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:dim]] for t in texts]


@pytest.fixture
def base(
    migrated_session: Session, qdrant: QdrantClient, scratch_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Session, QdrantClient, str]]:  # fmt: skip
    """Two papers indexed in a scratch collection as the app would (chunks of 512 tokens,
    dimension 4), one of them with full text and an analysis."""
    monkeypatch.setenv("QDRANT_COLLECTION", scratch_collection)
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    monkeypatch.setenv("EVAL_LARGE_EMBEDDING_DIM", "6")
    get_settings.cache_clear()
    for arxiv_id in (FULL, ABSTRACT_ONLY):
        migrated_session.add(Paper(
            arxiv_id=arxiv_id, title=f"Invented {arxiv_id}",
            abstract=f"Invented abstract of {arxiv_id}.",
            authors=[], categories=["cs.CL"], primary_category="cs.CL",
            published_at=datetime(2026, 9, 1, tzinfo=UTC),
        ))  # fmt: skip
    migrated_session.flush()
    migrated_session.add(PaperText(arxiv_id=FULL, version=1, raw_markdown=MARKDOWN,
                                   markdown=MARKDOWN, token_count=900))  # fmt: skip
    for arxiv_id in (FULL, ABSTRACT_ONLY):
        chunks = [
            abstract_chunk(arxiv_id, 1, f"Invented {arxiv_id}", f"Invented abstract of {arxiv_id}.")
        ]
        if arxiv_id == FULL:
            chunks += chunk_paper(arxiv_id, 1, f"Invented {arxiv_id}", MARKDOWN)
            chunks += analysis_chunks(
                arxiv_id,
                1,
                f"Invented {arxiv_id}",
                {"tldr": "Invented summary.", "method": "Invented method."},
            )
        save_chunks(migrated_session, arxiv_id, chunks)
    migrated_session.commit()
    ensure_collection(qdrant, scratch_collection, 4)
    dense, sparse = HashDense(), get_sparse_embedder()
    for arxiv_id in (FULL, ABSTRACT_ONLY):
        upsert_paper_chunks(migrated_session, arxiv_id, dense=dense, sparse=sparse, client=qdrant,
                            collection=scratch_collection)  # fmt: skip
    upsert_analysis_chunks(migrated_session, FULL, dense=dense, sparse=sparse, client=qdrant,
                           collection=scratch_collection)  # fmt: skip
    migrated_session.commit()
    try:
        yield migrated_session, qdrant, scratch_collection
    finally:
        for c in qdrant.get_collections().collections:
            if c.name.startswith(f"{scratch_collection}__eval__"):
                qdrant.delete_collection(c.name)


def _plan(session: Session, client: QdrantClient, variant: str):  # type: ignore[no-untyped-def]
    s = get_settings()
    return plan_index(session, resolve(find(variant, s), s), s, client)


def test_another_chunk_size_rechunks_sections_and_copies_the_rest(
    base: tuple[Session, QdrantClient, str],
) -> None:
    session, client, collection = base
    plan = _plan(session, client, "a7-c256")
    assert plan.collection == f"{collection}__eval__c256o32__text-embedding-3-small-4"
    base_sections = len(chunk_paper(FULL, 1, f"Invented {FULL}", MARKDOWN))
    small = chunk_paper(FULL, 1, f"Invented {FULL}", MARKDOWN, ChunkSettings(256, 32, 64, 2048))
    assert len(small) > base_sections  # smaller chunks: more of them
    assert len(plan.to_embed) == len(small)  # only the new section chunks are embedded
    assert {i.chunk_type for i in plan.to_copy} == {"abstract", "analysis"}

    dense = HashDense()
    embedded = build_index(plan, dense=dense, sparse=get_sparse_embedder(), client=client,
                           source_collection=collection)  # fmt: skip
    assert embedded == len(small)
    assert client.count(plan.collection, exact=True).count == len(plan.items)
    # Abstract and analysis points keep the app's vectors (nothing re-embedded for them).
    abstract_id = str(abstract_chunk(FULL, 1, "", "x").id)
    (old,) = client.retrieve(collection, [abstract_id], with_vectors=[DENSE_VECTOR])
    (new,) = client.retrieve(plan.collection, [abstract_id], with_vectors=[DENSE_VECTOR],
                             with_payload=True)  # fmt: skip
    assert new.vector == old.vector and new.payload is not None
    assert new.payload["chunk_type"] == "abstract" and new.payload["arxiv_id"] == FULL
    assert all("abstract" not in t.split("\n\n")[0] for t in dense.texts)
    # Built again: everything is there, nothing is embedded or paid twice.
    again = _plan(session, client, "a7-c256")
    assert len(again.present) == len(again.items) and again.to_embed == []
    assert build_index(again, dense=HashDense(), sparse=get_sparse_embedder(), client=client,
                       source_collection=collection) == 0  # fmt: skip


def test_a_shortened_large_model_embeds_every_point_with_the_dimensions_parameter(
    base: tuple[Session, QdrantClient, str],
) -> None:
    session, client, collection = base
    s = get_settings()
    resolved = resolve(find("a8-large-short", s), s)
    plan = plan_index(session, resolved, s, client)
    assert plan.to_copy == [] and len(plan.to_embed) == len(plan.items)
    assert plan.cost_usd(s) > 0 and plan.requests() == 1
    fake = FakeOpenAIEmbeddings()
    logged: list[tuple[str, str, int]] = []
    with applied_settings(resolved.overrides(s)):
        encoder = index_embedder(
            resolved.index, lambda f, m, t: logged.append((f, m, t)), lambda: fake
        )
        build_index(plan, dense=encoder, sparse=get_sparse_embedder(), client=client,
                    source_collection=collection)  # fmt: skip
    assert fake.requests == [(s.eval_large_embedding_model, 4, 4)]  # shortened to 4
    assert logged == [(FEATURE, s.eval_large_embedding_model, 7)]
    info = client.get_collection(plan.collection)
    assert info.config.params.vectors[DENSE_VECTOR].size == 4  # type: ignore[index]


def test_a_paper_without_stored_text_stops_another_chunking(
    base: tuple[Session, QdrantClient, str],
) -> None:
    session, client, _ = base
    text = session.get(PaperText, FULL)
    session.delete(text)
    session.flush()
    with pytest.raises(IndexBuildError, match="no stored cleaned text"):
        _plan(session, client, "a7-c1024")


def test_the_builder_never_writes_to_the_apps_collection(
    base: tuple[Session, QdrantClient, str],
) -> None:
    session, client, collection = base
    plan = _plan(session, client, "a7-c1024")
    plan.collection = collection
    with pytest.raises(IndexBuildError, match="not a variant collection"):
        build_index(plan, dense=HashDense(), sparse=get_sparse_embedder(), client=client,
                    source_collection=collection)  # fmt: skip
    with pytest.raises(IndexBuildError, match="own collection"):
        _plan(session, client, "a7-c512")
    assert uuid.UUID(str(plan.items[0].id))


def test_the_command_line_estimates_for_free_and_needs_a_cap_to_embed(
    base: tuple[Session, QdrantClient, str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    from eval.indexes import main

    session, client, collection = base
    assert main(["estimate", "--variant", "a7-c256"]) == 0
    out = capsys.readouterr().out
    assert "to embed with text-embedding-3-small" in out and "nothing was sent or written" in out
    assert main(["build", "--variant", "a8-large", "--base", "a7-c512"]) == 2
    assert "needs --max-usd" in capsys.readouterr().err
    assert main(["build", "--variant", "a8-large", "--base", "a6-rewrite"]) == 2  # not a7-...
    assert main(["estimate", "--variant", "a8-large"]) == 2  # no stored chunking winner
    assert "--base" in capsys.readouterr().err
    assert not [c for c in client.get_collections().collections if "__eval__" in c.name]
    assert main(["drop", "--variant", "a7-c256"]) == 0  # nothing to drop: still fine

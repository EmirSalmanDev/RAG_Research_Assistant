"""ensure_collection() against the local Qdrant, on throwaway `chunks_test_<random>`
collections only. Every test checks that the real collection was not touched."""

import uuid
from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m

from app.config import get_settings
from app.ingest.index import (
    DENSE_VECTOR,
    PAYLOAD_INDEXES,
    SPARSE_VECTOR,
    CollectionConfigError,
    config_differences,
    ensure_collection,
)

pytestmark = pytest.mark.integration

DIM = 1536


@pytest.fixture
def client() -> Iterator[QdrantClient]:
    c = QdrantClient(url=get_settings().qdrant_url)
    yield c
    c.close()


def _real_state(client: QdrantClient) -> tuple[bool, int | None]:
    """(exists, point count) of the real collection, e.g. `chunks`."""
    name = get_settings().qdrant_collection
    if not client.collection_exists(name):
        return False, None
    return True, client.count(name, exact=True).count


@pytest.fixture(autouse=True)
def real_collection_untouched(client: QdrantClient) -> Iterator[None]:
    before = _real_state(client)
    yield
    assert _real_state(client) == before, "the real collection changed during the test"


@pytest.fixture
def scratch(client: QdrantClient) -> Iterator[str]:
    """A unique collection name; the collection is deleted afterwards, even on failure."""
    name = f"chunks_test_{uuid.uuid4().hex[:8]}"
    assert name != get_settings().qdrant_collection
    try:
        yield name
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)


def test_created_twice_is_a_noop(client: QdrantClient, scratch: str) -> None:
    assert ensure_collection(client, scratch, DIM) is True
    first = client.get_collection(scratch)
    assert ensure_collection(client, scratch, DIM) is False
    second = client.get_collection(scratch)
    assert second.config.params == first.config.params
    assert second.config.quantization_config == first.config.quantization_config
    assert second.payload_schema.keys() == first.payload_schema.keys()
    assert second.points_count == 0


def test_config_matches_the_expected_layout(client: QdrantClient, scratch: str) -> None:
    ensure_collection(client, scratch, DIM)
    info = client.get_collection(scratch)
    assert config_differences(info, DIM) == []

    vectors = info.config.params.vectors
    assert isinstance(vectors, dict)
    dense = vectors[DENSE_VECTOR]
    assert (dense.size, dense.distance, dense.on_disk) == (DIM, m.Distance.COSINE, True)
    sparse = (info.config.params.sparse_vectors or {})[SPARSE_VECTOR]
    assert sparse.modifier == m.Modifier.IDF
    quant = info.config.quantization_config
    assert isinstance(quant, m.ScalarQuantization)
    assert (quant.scalar.type, quant.scalar.always_ram) == (m.ScalarType.INT8, True)
    assert {f: i.data_type for f, i in info.payload_schema.items()} == PAYLOAD_INDEXES


def test_mismatch_raises_and_names_each_setting(client: QdrantClient, scratch: str) -> None:
    # Created by hand with a wrong size, no IDF, no quantization.
    client.create_collection(
        scratch,
        vectors_config={DENSE_VECTOR: m.VectorParams(size=768, distance=m.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: m.SparseVectorParams()},
    )
    with pytest.raises(CollectionConfigError) as err:
        ensure_collection(client, scratch, DIM)
    message = str(err.value)
    assert scratch in message
    assert "text-dense.size: expected 1536, found 768" in message
    assert "text-dense.on_disk: expected True" in message
    assert "text-sparse.modifier: expected idf" in message
    assert "quantization: expected scalar int8" in message
    # Nothing was changed: still the hand-made config, and no payload index was added.
    info = client.get_collection(scratch)
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict) and vectors[DENSE_VECTOR].size == 768
    assert info.payload_schema == {}


def test_wrong_payload_index_type_is_reported(client: QdrantClient, scratch: str) -> None:
    ensure_collection(client, scratch, DIM)
    client.delete_payload_index(scratch, "published_at", wait=True)
    client.create_payload_index(scratch, "published_at", m.PayloadSchemaType.KEYWORD, wait=True)
    expected = "payload index published_at: expected datetime, found keyword"
    with pytest.raises(CollectionConfigError, match=expected):
        ensure_collection(client, scratch, DIM)


def test_missing_payload_indexes_are_added(client: QdrantClient, scratch: str) -> None:
    ensure_collection(client, scratch, DIM)
    client.delete_payload_index(scratch, "categories", wait=True)
    assert ensure_collection(client, scratch, DIM) is False
    assert "categories" in client.get_collection(scratch).payload_schema


def test_name_and_size_come_from_config(
    client: QdrantClient, scratch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QDRANT_COLLECTION", scratch)  # points config at the scratch name
    get_settings.cache_clear()
    try:
        assert ensure_collection(client) is True
        vectors = client.get_collection(scratch).config.params.vectors
        assert isinstance(vectors, dict)
        assert vectors[DENSE_VECTOR].size == get_settings().embedding_dim
    finally:
        monkeypatch.delenv("QDRANT_COLLECTION")
        get_settings.cache_clear()

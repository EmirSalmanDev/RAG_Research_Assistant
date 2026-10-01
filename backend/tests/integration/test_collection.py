"""ensure_collection() against the local Qdrant, on throwaway `chunks_test_<random>`
collections only. Every test checks that the real collection was not touched
(fixtures in tests/conftest.py)."""

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

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]

DIM = 1536


def test_created_twice_is_a_noop(qdrant: QdrantClient, scratch_collection: str) -> None:
    assert ensure_collection(qdrant, scratch_collection, DIM) is True
    first = qdrant.get_collection(scratch_collection)
    assert ensure_collection(qdrant, scratch_collection, DIM) is False
    second = qdrant.get_collection(scratch_collection)
    assert second.config.params == first.config.params
    assert second.config.quantization_config == first.config.quantization_config
    assert second.payload_schema.keys() == first.payload_schema.keys()
    assert second.points_count == 0


def test_config_matches_the_expected_layout(qdrant: QdrantClient, scratch_collection: str) -> None:
    ensure_collection(qdrant, scratch_collection, DIM)
    info = qdrant.get_collection(scratch_collection)
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


def test_mismatch_raises_and_names_each_setting(
    qdrant: QdrantClient, scratch_collection: str
) -> None:
    # Created by hand with a wrong size, no IDF, no quantization.
    qdrant.create_collection(
        scratch_collection,
        vectors_config={DENSE_VECTOR: m.VectorParams(size=768, distance=m.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: m.SparseVectorParams()},
    )
    with pytest.raises(CollectionConfigError) as err:
        ensure_collection(qdrant, scratch_collection, DIM)
    message = str(err.value)
    assert scratch_collection in message
    assert "text-dense.size: expected 1536, found 768" in message
    assert "text-dense.on_disk: expected True" in message
    assert "text-sparse.modifier: expected idf" in message
    assert "quantization: expected scalar int8" in message
    # Nothing was changed: still the hand-made config, and no payload index was added.
    info = qdrant.get_collection(scratch_collection)
    vectors = info.config.params.vectors
    assert isinstance(vectors, dict) and vectors[DENSE_VECTOR].size == 768
    assert info.payload_schema == {}


def test_wrong_payload_index_type_is_reported(
    qdrant: QdrantClient, scratch_collection: str
) -> None:
    ensure_collection(qdrant, scratch_collection, DIM)
    qdrant.delete_payload_index(scratch_collection, "published_at", wait=True)
    qdrant.create_payload_index(
        scratch_collection, "published_at", m.PayloadSchemaType.KEYWORD, wait=True
    )
    expected = "payload index published_at: expected datetime, found keyword"
    with pytest.raises(CollectionConfigError, match=expected):
        ensure_collection(qdrant, scratch_collection, DIM)


def test_missing_payload_indexes_are_added(qdrant: QdrantClient, scratch_collection: str) -> None:
    ensure_collection(qdrant, scratch_collection, DIM)
    qdrant.delete_payload_index(scratch_collection, "categories", wait=True)
    assert ensure_collection(qdrant, scratch_collection, DIM) is False
    assert "categories" in qdrant.get_collection(scratch_collection).payload_schema


def test_name_and_size_come_from_config(
    qdrant: QdrantClient, scratch_collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "QDRANT_COLLECTION", scratch_collection
    )  # points config at the scratch_collection name
    get_settings.cache_clear()
    try:
        assert ensure_collection(qdrant) is True
        vectors = qdrant.get_collection(scratch_collection).config.params.vectors
        assert isinstance(vectors, dict)
        assert vectors[DENSE_VECTOR].size == get_settings().embedding_dim
    finally:
        monkeypatch.delenv("QDRANT_COLLECTION")
        get_settings.cache_clear()

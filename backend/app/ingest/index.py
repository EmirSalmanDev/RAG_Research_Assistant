"""The Qdrant collection that holds every searchable chunk.

`ensure_collection()` creates it if it is missing and otherwise checks that the existing
one has the settings below. It never changes or deletes an existing collection: a
mismatch raises `CollectionConfigError` naming each differing setting, because fixing it
means recreating the collection and re-indexing (Postgres is the source of truth, so that
is always possible, but it must be a deliberate step).

Layout (one point per chunk; the point id is the chunk id):
- `text-dense`: OpenAI embedding, cosine distance. Original float vectors live on disk
  (`on_disk=True`); an int8 copy (1 byte per value instead of 4) is kept in RAM
  (`always_ram=True`) for fast search, and the top hits are rescored with the originals.
- `text-sparse`: BM25 term weights from FastEmbed. Qdrant applies the IDF part at query
  time (`Modifier.IDF`), from statistics of the whole collection.
- payload indexes on the fields search filters by, so filters stay fast.

qdrant-client 1.19 marks `on_disk` and `always_ram` as deprecated in favour of `memory`
(`on_disk=True` ~ `memory=cold`, `always_ram=True` ~ `memory=pinned`). The old flags still
work and the server reports them back, so they are used here; the checks below also
accept the `memory` form in case a later server reports the setting that way.
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client import models as m
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings

DENSE_VECTOR = "text-dense"
SPARSE_VECTOR = "text-sparse"

PAYLOAD_INDEXES: dict[str, m.PayloadSchemaType] = {
    "arxiv_id": m.PayloadSchemaType.KEYWORD,
    "chunk_type": m.PayloadSchemaType.KEYWORD,
    "categories": m.PayloadSchemaType.KEYWORD,  # a list field: matches any element
    "published_at": m.PayloadSchemaType.DATETIME,
}


class CollectionConfigError(RuntimeError):
    """An existing collection's settings differ from what this code expects."""


@lru_cache
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collection(
    client: QdrantClient | None = None, name: str | None = None, dim: int | None = None
) -> bool:
    """Create the chunks collection if it is missing, then verify its config and make sure
    every payload index exists. Safe to call any number of times.

    Returns True if this call created the collection, False if it already existed.
    """
    client = client or get_qdrant()
    name = name or get_settings().qdrant_collection
    dim = dim or get_settings().embedding_dim

    created = False
    if not client.collection_exists(name):
        try:
            dense = m.VectorParams(size=dim, distance=m.Distance.COSINE, on_disk=True)
            sparse = m.SparseVectorParams(modifier=m.Modifier.IDF)
            int8 = m.ScalarQuantizationConfig(type=m.ScalarType.INT8, always_ram=True)
            client.create_collection(
                name,
                vectors_config={DENSE_VECTOR: dense},
                sparse_vectors_config={SPARSE_VECTOR: sparse},
                quantization_config=m.ScalarQuantization(scalar=int8),
            )
            created = True
        except UnexpectedResponse:
            # Another process (worker vs. CLI) may have created it a moment ago.
            if not client.collection_exists(name):
                raise

    info = client.get_collection(name)
    problems = config_differences(info, dim)
    if problems:
        raise CollectionConfigError(
            f"Qdrant collection {name!r} exists with a different config: "
            + "; ".join(problems)
            + ". It is not changed automatically: recreate it and re-index from Postgres."
        )

    for field, schema in PAYLOAD_INDEXES.items():
        if field not in info.payload_schema:  # missing index: add it (types checked above)
            client.create_payload_index(name, field, field_schema=schema, wait=True)
    return created


def config_differences(info: m.CollectionInfo, dim: int) -> list[str]:
    """Every setting that differs from the expected layout, as 'setting: expected X, found Y'."""
    problems: list[str] = []
    params = info.config.params

    vectors = params.vectors
    dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
    if dense is None:
        problems.append(f"{DENSE_VECTOR}: expected a named dense vector, found {vectors!r}")
    else:
        if dense.size != dim:
            problems.append(f"{DENSE_VECTOR}.size: expected {dim}, found {dense.size}")
        if dense.distance != m.Distance.COSINE:
            problems.append(f"{DENSE_VECTOR}.distance: expected Cosine, found {dense.distance}")
        if not (dense.on_disk is True or dense.memory == m.Memory.COLD):
            problems.append(
                f"{DENSE_VECTOR}.on_disk: expected True, found on_disk={dense.on_disk}, "
                f"memory={dense.memory}"
            )

    sparse = (params.sparse_vectors or {}).get(SPARSE_VECTOR)
    if sparse is None:
        problems.append(f"{SPARSE_VECTOR}: expected a named sparse vector, found none")
    elif sparse.modifier != m.Modifier.IDF:
        problems.append(f"{SPARSE_VECTOR}.modifier: expected idf, found {sparse.modifier}")

    quant = info.config.quantization_config
    if not isinstance(quant, m.ScalarQuantization):
        problems.append(f"quantization: expected scalar int8, found {quant!r}")
    else:
        if quant.scalar.type != m.ScalarType.INT8:
            problems.append(f"quantization.type: expected int8, found {quant.scalar.type}")
        if not (quant.scalar.always_ram is True or quant.scalar.memory == m.Memory.PINNED):
            problems.append(
                f"quantization.always_ram: expected True, found always_ram="
                f"{quant.scalar.always_ram}, memory={quant.scalar.memory}"
            )

    for field, schema in PAYLOAD_INDEXES.items():
        index = info.payload_schema.get(field)
        if index is not None and index.data_type != schema:
            problems.append(
                f"payload index {field}: expected {schema.value}, found {index.data_type.value}"
            )
    return problems

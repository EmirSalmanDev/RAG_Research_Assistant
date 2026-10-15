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

Writing points (below `ensure_collection`): Postgres `chunks` rows are the source of
truth, Qdrant points are derived from them. Re-indexing a paper upserts the current
points first and only then deletes stale ones, so search never sees the paper vanish.
Each kind of chunk is re-indexed on its own: `upsert_paper_chunks` (abstract and
section chunks, after chunking a paper) and `upsert_analysis_chunks` (after a paper
analysis is stored). Stale deletion only ever touches the kinds being re-indexed, so
re-chunking a paper never removes its analysis chunks and the other way round.
"""

import uuid
from collections.abc import Iterator, Sequence
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as m
from qdrant_client.http.exceptions import UnexpectedResponse
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk as ChunkRow
from app.db.models import Paper
from app.ingest.chunk import ABSTRACT_ORD, Chunk, abstract_chunk, chunk_id, embedding_text
from app.ingest.embed import DenseEncoder, SparseEncoder

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
    """One shared client per process. Scripts and the CLI call `close_qdrant()` before
    exiting (otherwise qdrant-client closes it during interpreter shutdown and prints a
    noisy 'Exception ignored in QdrantClient.__del__')."""
    return QdrantClient(url=get_settings().qdrant_url)


def close_qdrant() -> None:
    """Close the shared client if one was created (safe to call more than once)."""
    if get_qdrant.cache_info().currsize:
        get_qdrant().close()
        get_qdrant.cache_clear()


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


# --- writing chunks and points ----------------------------------------------------------------

# Only these chunk types are rebuilt (and their stale copies deleted) by this module.
REINDEXED_TYPES = ("abstract", "section")
ANALYSIS_TYPES = ("analysis",)
UPSERT_BATCH = 64  # points per Qdrant upsert request
LOOKUP_BATCH = 256  # ids per Qdrant retrieve / delete request
EMBED_GROUP = 1000  # abstracts embedded and upserted per round (bounds memory)
FEATURE = {  # llm_usage.feature of the embedding requests, per chunk type
    "abstract": "embed-abstract",
    "section": "embed-fulltext",
    "analysis": "embed-analysis",
}


def _batches[T](items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def save_chunks(session: Session, arxiv_id: str, chunks: Sequence[Chunk]) -> None:
    """Upsert chunk rows in Postgres, keyed on (arxiv_id, ord). A row from an older paper
    version at the same ord is overwritten (its id changes to the new version's id)."""
    for chunk in chunks:
        values = {
            "id": chunk.id,
            "arxiv_id": arxiv_id,
            "chunk_type": chunk.chunk_type,
            "section_path": chunk.section_path,
            "ord": chunk.ord,
            "text": chunk.text,
            "token_count": chunk.token_count,
        }
        stmt = pg_insert(ChunkRow).values(**values)
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[ChunkRow.arxiv_id, ChunkRow.ord],
                set_={k: stmt.excluded[k] for k in values if k not in ("arxiv_id", "ord")},
            )
        )


def payload(paper: Paper, row: ChunkRow) -> dict[str, Any]:
    """The Qdrant payload of a chunk point (fields as in the Data model)."""
    return {
        "arxiv_id": paper.arxiv_id,
        "chunk_type": row.chunk_type,
        "section_path": row.section_path,
        "title": paper.title,
        "published_at": paper.published_at.isoformat(),  # RFC 3339, for the datetime index
        "categories": list(paper.categories),
        "primary_category": paper.primary_category,
        "has_code": paper.github_url is not None,
        "text": row.text,
    }


def _points(
    rows: Sequence[tuple[Paper, ChunkRow]], dense: DenseEncoder, sparse: SparseEncoder
) -> list[m.PointStruct]:
    """Embed the rows (one dense request per batch and chunk type) and build points."""
    points: list[m.PointStruct] = []
    for chunk_type in FEATURE:
        group = [(p, r) for p, r in rows if r.chunk_type == chunk_type]
        if not group:
            continue
        texts = [embedding_text(p.title, r.section_path, r.text) for p, r in group]
        dense_vectors = dense.embed(texts, FEATURE[chunk_type])
        sparse_vectors = sparse.embed(texts)
        for (paper, row), dv, sv in zip(group, dense_vectors, sparse_vectors, strict=True):
            points.append(
                m.PointStruct(
                    id=str(row.id),
                    vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv},
                    payload=payload(paper, row),
                )
            )
    return points


def _upsert(client: QdrantClient, collection: str, points: Sequence[m.PointStruct]) -> None:
    for batch in _batches(points, UPSERT_BATCH):
        client.upsert(collection, points=list(batch), wait=True)


def _delete_stale_points(
    client: QdrantClient, collection: str, arxiv_ids: Sequence[str], types: Sequence[str],
    keep: Sequence[uuid.UUID],
) -> None:  # fmt: skip
    """Delete points of these papers and chunk types whose id is not in `keep`.
    One filtered delete per request, never touching other chunk types."""
    stale = m.Filter(
        must=[
            m.FieldCondition(key="arxiv_id", match=m.MatchAny(any=list(arxiv_ids))),
            m.FieldCondition(key="chunk_type", match=m.MatchAny(any=list(types))),
        ],
        must_not=[m.HasIdCondition(has_id=[str(i) for i in keep])] if keep else [],
    )
    client.delete(collection, points_selector=m.FilterSelector(filter=stale), wait=True)


def _embedded_fields(p: dict[str, Any]) -> tuple[Any, ...]:
    """The payload fields the vectors are computed from (see chunk.embedding_text)."""
    return (p.get("title"), p.get("section_path"), p.get("text"))


def _stored_payloads(
    client: QdrantClient, collection: str, ids: Sequence[uuid.UUID]
) -> dict[str, dict[str, Any]]:
    """Payloads of the points that already exist, keyed by point id."""
    found: dict[str, dict[str, Any]] = {}
    for batch in _batches(ids, LOOKUP_BATCH):
        for record in client.retrieve(collection, ids=[str(i) for i in batch], with_payload=True):
            found[str(record.id)] = dict(record.payload or {})
    return found


def upsert_paper_chunks(
    session: Session,
    arxiv_id: str,
    *,
    dense: DenseEncoder,
    sparse: SparseEncoder,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> int:
    """(Re)index one paper's abstract and section chunks from its Postgres rows.

    Current rows are those whose id matches the paper's current version. Only rows whose
    point is missing, or whose embedded text (title, section path, chunk text) changed,
    are embedded and upserted; a point whose other payload fields changed only gets its
    payload updated (no embedding call). So re-running on an unchanged paper costs
    nothing. Then stale points (older versions, removed ords) and stale rows are
    deleted, for chunk types abstract and section only.
    Returns the number of points embedded. The caller commits the session.
    """
    return _sync_chunks(
        session, arxiv_id, REINDEXED_TYPES, dense=dense, sparse=sparse, client=client,
        collection=collection,
    )  # fmt: skip


def upsert_analysis_chunks(
    session: Session,
    arxiv_id: str,
    *,
    dense: DenseEncoder,
    sparse: SparseEncoder,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> int:
    """(Re)index one paper's analysis chunks, exactly like `upsert_paper_chunks` does for
    its abstract and section chunks (unchanged points are not embedded again; stale
    analysis points and rows are deleted; other chunk types are never touched)."""
    return _sync_chunks(
        session, arxiv_id, ANALYSIS_TYPES, dense=dense, sparse=sparse, client=client,
        collection=collection,
    )  # fmt: skip


def _sync_chunks(
    session: Session,
    arxiv_id: str,
    types: Sequence[str],
    *,
    dense: DenseEncoder,
    sparse: SparseEncoder,
    client: QdrantClient | None,
    collection: str | None,
) -> int:
    client = client or get_qdrant()
    collection = collection or get_settings().qdrant_collection
    paper = session.get(Paper, arxiv_id)
    if paper is None:
        raise ValueError(f"no paper {arxiv_id} in Postgres")
    rows = session.scalars(
        select(ChunkRow)
        .where(ChunkRow.arxiv_id == arxiv_id, ChunkRow.chunk_type.in_(types))
        .order_by(ChunkRow.ord)
    ).all()
    current = [r for r in rows if r.id == chunk_id(arxiv_id, paper.version, r.ord)]
    stale_ids = [r.id for r in rows if r not in current]

    stored = _stored_payloads(client, collection, [r.id for r in current])
    to_embed: list[ChunkRow] = []
    for row in current:
        old, new = stored.get(str(row.id)), payload(paper, row)
        if old is None or _embedded_fields(old) != _embedded_fields(new):
            to_embed.append(row)
        elif old != new:  # same vectors, other metadata changed (e.g. categories, has_code)
            client.set_payload(collection, payload=new, points=[str(row.id)], wait=True)
    points = _points([(paper, r) for r in to_embed], dense, sparse)
    _upsert(client, collection, points)
    _delete_stale_points(client, collection, [arxiv_id], types, keep=[r.id for r in current])
    if stale_ids:
        session.execute(delete(ChunkRow).where(ChunkRow.id.in_(stale_ids)))
    return len(points)


def index_abstracts(
    session: Session,
    arxiv_ids: Sequence[str] | None = None,
    *,
    dense: DenseEncoder,
    sparse: SparseEncoder,
    client: QdrantClient | None = None,
    collection: str | None = None,
) -> int:
    """Make sure every paper (or the given ones) has its ord-0 abstract chunk indexed.

    A paper is skipped when Qdrant already holds the point for its current version, so a
    re-run makes no embedding calls. For the others: the ord-0 rows are written to
    Postgres and committed first, then embedded and upserted to Qdrant, then the
    abstract point of an older version (if any) is deleted. Returns how many were indexed.
    """
    client = client or get_qdrant()
    collection = collection or get_settings().qdrant_collection
    query = select(Paper).order_by(Paper.arxiv_id)
    if arxiv_ids is not None:
        query = query.where(Paper.arxiv_id.in_(list(arxiv_ids)))
    papers = session.scalars(query).all()

    # 1. Find every paper whose current abstract point is missing (Qdrant lookups in
    #    batches). Collecting them first lets the embedding requests be full batches,
    #    even when the missing papers are spread thinly across all papers.
    todo: list[Paper] = []
    for batch in _batches(papers, LOOKUP_BATCH):
        wanted = {chunk_id(p.arxiv_id, p.version, ABSTRACT_ORD): p for p in batch}
        present = {
            uuid.UUID(str(record.id))
            for record in client.retrieve(
                collection, ids=[str(i) for i in wanted], with_payload=False, with_vectors=False
            )
        }
        todo += [p for i, p in wanted.items() if i not in present]

    indexed = 0
    # 2. Index them in groups (bounded memory: vectors of one group at a time).
    for group in _batches(todo, EMBED_GROUP):
        # Postgres first (source of truth), committed before any money is spent.
        for paper in group:
            chunk = abstract_chunk(paper.arxiv_id, paper.version, paper.title, paper.abstract)
            save_chunks(session, paper.arxiv_id, [chunk])
        session.commit()
        # Then Qdrant: embed and upsert, then remove abstract points of older versions.
        rows = session.scalars(
            select(ChunkRow).where(
                ChunkRow.arxiv_id.in_([p.arxiv_id for p in group]), ChunkRow.ord == ABSTRACT_ORD
            )
        ).all()
        by_id = {p.arxiv_id: p for p in group}
        _upsert(client, collection, _points([(by_id[r.arxiv_id], r) for r in rows], dense, sparse))
        _delete_stale_points(
            client, collection, [p.arxiv_id for p in group], ["abstract"], keep=[r.id for r in rows]
        )
        indexed += len(rows)
    return indexed

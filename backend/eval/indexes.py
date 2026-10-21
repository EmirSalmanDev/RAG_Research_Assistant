"""The Qdrant collections of the index variants (ablations 7 and 8), built from Postgres.

    python -m eval.indexes list                                   # eval collections (free)
    python -m eval.indexes estimate --variant a7-c256             # points, tokens, cost (free)
    python -m eval.indexes build --variant a7-c256 --max-usd 0.05 # PAID: ask first
    python -m eval.indexes drop --variant a7-c256                 # removes that collection only

A chunking or embedding variant searches its own collection, named after what it holds
(`<qdrant_collection>__eval__c<tokens>o<overlap>__<model>-<dim>[-short]`, eval/variants.py),
with the same layout as the app's (named dense and BM25 vectors, the same payload and
payload indexes; app/ingest/index.ensure_collection). The app's collection and the
Postgres rows are only read: Postgres stays the source of truth, nothing there changes.

What goes in, for every paper's current chunks (the rows of its current version):
- abstract and analysis chunks are the same under any chunking (an abstract is one chunk,
  analyses are four fixed ones), so they are taken from the chunks table;
- section chunks: under the configured chunking the chunks table's rows, under another
  chunking the paper's cleaned text (`paper_texts.markdown`) chunked again with the
  variant's sizes (`app/ingest/chunk.chunk_paper`), ids as the app would give them
  (`uuid5(arxiv_id, version, ord)` of the new ords). A paper with section chunks but no
  stored text stops the build: the variant would silently lack it.
- dense vectors: with the configured embedding model and size, copied from the app's
  collection (no request); else embedded with the variant's model (and `dimensions`
  when it is shortened), feature `eval-embed-index` in llm_usage at that model's price.
  BM25 vectors are computed locally with the same model as the app's (free).
The build is resumable: points already in the variant collection with the same text are
skipped, so an interrupted (or capped) build pays only for what is missing. At the end the
collection's point count must equal the plan's.

A build spends money only for the embeddings it needs (`estimate` shows how many tokens,
counted locally with the embedding tokenizer), so it needs `--max-usd`, the hard cap of
eval/budget.py.
"""

import argparse
import math
import sys
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Chunk as ChunkRow
from app.db.models import Paper, PaperText
from app.ingest.chunk import ChunkSettings, chunk_id, chunk_paper, embedding_text
from app.ingest.embed import DenseEncoder, SparseEncoder, get_sparse_embedder
from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection, get_qdrant, payload
from app.ingest.tokens import count_tokens
from app.llm.usage import UsageRecorder, record_embedding_usage
from eval.scope import applied_settings
from eval.variants import IndexSpec, Resolved, embedding_price

FEATURE = "eval-embed-index"  # llm_usage.feature of a variant index's embeddings
BATCH = 64  # points embedded and upserted together (one embeddings request each)
EVAL_MARK = "__eval__"


class IndexBuildError(RuntimeError):
    """The variant cannot be built as asked (nothing was written for the reason given)."""


@dataclass(frozen=True)
class Item:
    """One point of the variant collection, before its vectors."""

    id: uuid.UUID
    paper: Paper
    chunk_type: str
    section_path: str | None
    text: str
    copy_from: uuid.UUID | None  # the app's point with the same text and vectors, if usable

    def embedded(self) -> str:
        return embedding_text(self.paper.title, self.section_path, self.text)


@dataclass
class IndexPlan:
    variant: str
    collection: str
    index: IndexSpec
    items: list[Item] = field(default_factory=list)
    present: set[uuid.UUID] = field(default_factory=set)  # already built with the same text

    @property
    def to_embed(self) -> list[Item]:
        return [i for i in self.items if i.id not in self.present and i.copy_from is None]

    @property
    def to_copy(self) -> list[Item]:
        return [i for i in self.items if i.id not in self.present and i.copy_from is not None]

    def tokens(self) -> int:
        return sum(count_tokens(i.embedded()) for i in self.to_embed)

    def requests(self) -> int:
        return math.ceil(len(self.to_embed) / BATCH)

    def cost_usd(self, s: Settings) -> Decimal:
        price = Decimal(str(embedding_price(self.index.embedding_model, s)))
        return (Decimal(self.tokens()) * price / 1_000_000).quantize(
            Decimal("0.000001"), ROUND_HALF_UP
        )


def _current_rows(session: Session) -> Iterator[tuple[Paper, ChunkRow]]:
    """Every chunk row of every paper's current version, papers in id order."""
    papers = {p.arxiv_id: p for p in session.scalars(select(Paper))}
    rows = session.scalars(select(ChunkRow).order_by(ChunkRow.arxiv_id, ChunkRow.ord))
    for row in rows:
        paper = papers.get(row.arxiv_id)
        if paper is not None and row.id == chunk_id(row.arxiv_id, paper.version, row.ord):
            yield paper, row


def plan_index(
    session: Session, resolved: Resolved, s: Settings, client: QdrantClient, *, present: bool = True
) -> IndexPlan:
    """What the variant's collection holds and what building it needs (no write, no request).
    `s` is the configured settings (the app's index); `present=False` skips the look at
    the variant collection (a plan for a collection that does not exist yet)."""
    index = resolved.index
    configured = IndexSpec.configured(s)
    if index == configured:
        raise IndexBuildError(f"{resolved.variant.id} searches the app's own collection")
    same_chunking = index.same_chunking(s)
    same_vectors = (index.embedding_model, index.embedding_dim, index.request_dimensions) == (
        configured.embedding_model, configured.embedding_dim, configured.request_dimensions,
    )  # fmt: skip
    cfg = ChunkSettings(index.chunk_tokens, index.chunk_overlap, s.chunk_min_tokens,
                        s.chunk_hard_max_tokens)  # fmt: skip
    plan = IndexPlan(resolved.variant.id, index.collection(s), index)
    with_sections: set[str] = set()
    for paper, row in _current_rows(session):
        if row.chunk_type == "section" and not same_chunking:
            with_sections.add(paper.arxiv_id)
            continue
        plan.items.append(Item(row.id, paper, row.chunk_type, row.section_path, row.text,
                               row.id if same_vectors else None))  # fmt: skip
    for arxiv_id in sorted(with_sections):
        text = session.get(PaperText, arxiv_id)
        held = session.get(Paper, arxiv_id)
        if text is None or held is None or not text.markdown.strip():
            raise IndexBuildError(
                f"{arxiv_id} has section chunks but no stored cleaned text to chunk again"
            )
        for chunk in chunk_paper(arxiv_id, held.version, held.title, text.markdown, cfg):
            plan.items.append(Item(chunk.id, held, "section", chunk.section_path, chunk.text, None))
    if present and client.collection_exists(plan.collection):
        plan.present = _already_built(client, plan)
    return plan


def _already_built(client: QdrantClient, plan: IndexPlan) -> set[uuid.UUID]:
    """Points of the variant collection whose text and section match the plan's."""
    wanted = {i.id: i for i in plan.items}
    found: set[uuid.UUID] = set()
    ids = list(wanted)
    for start in range(0, len(ids), 256):
        batch = [str(i) for i in ids[start : start + 256]]
        for record in client.retrieve(plan.collection, ids=batch, with_payload=True):
            item = wanted[uuid.UUID(str(record.id))]
            p = record.payload or {}
            if (p.get("text"), p.get("section_path"), p.get("title")) == (
                item.text, item.section_path, item.paper.title,
            ):  # fmt: skip
                found.add(item.id)
    return found


def _payload(item: Item) -> dict[str, Any]:
    row = ChunkRow(id=item.id, arxiv_id=item.paper.arxiv_id, chunk_type=item.chunk_type,
                   section_path=item.section_path, text=item.text)  # fmt: skip
    return payload(item.paper, row)


def build_index(
    plan: IndexPlan, *, dense: DenseEncoder, sparse: SparseEncoder, client: QdrantClient,
    source_collection: str, progress: Callable[[str], None] = lambda line: None,
    after_batch: Callable[[], None] = lambda: None,
) -> int:  # fmt: skip
    """Create the collection if needed and add every missing point; returns how many were
    embedded. `after_batch` runs after each upsert (the budget check of the CLI)."""
    if plan.collection == source_collection or EVAL_MARK not in plan.collection:
        raise IndexBuildError(f"refusing to write to {plan.collection!r}: not a variant collection")
    ensure_collection(client, plan.collection, plan.index.embedding_dim)
    copied = plan.to_copy
    for start in range(0, len(copied), BATCH):
        batch = copied[start : start + BATCH]
        records = client.retrieve(source_collection, ids=[str(i.copy_from) for i in batch],
                                  with_payload=False, with_vectors=[DENSE_VECTOR])  # fmt: skip
        vectors = {uuid.UUID(str(r.id)): r.vector for r in records}
        missing = [str(i.copy_from) for i in batch if i.copy_from not in vectors]
        if missing:
            raise IndexBuildError(f"points missing in {source_collection!r}: {missing[:5]}")
        sparse_vectors = sparse.embed([i.embedded() for i in batch])
        points = []
        for item, sv in zip(batch, sparse_vectors, strict=True):
            stored: Any = vectors[item.copy_from]  # type: ignore[index]
            dv = stored[DENSE_VECTOR] if isinstance(stored, dict) else stored
            points.append(m.PointStruct(id=str(item.id), payload=_payload(item),
                                        vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv}))  # fmt: skip
        client.upsert(plan.collection, points=points, wait=True)
        progress(f"copied {min(start + BATCH, len(copied))}/{len(copied)}")
    todo = plan.to_embed
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        texts = [i.embedded() for i in batch]
        dense_vectors = dense.embed(texts, FEATURE)
        sparse_vectors = sparse.embed(texts)
        points = [
            m.PointStruct(id=str(i.id), payload=_payload(i),
                          vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv})
            for i, dv, sv in zip(batch, dense_vectors, sparse_vectors, strict=True)
        ]  # fmt: skip
        client.upsert(plan.collection, points=points, wait=True)
        after_batch()
        progress(f"embedded {min(start + BATCH, len(todo))}/{len(todo)}")
    count = client.count(plan.collection, exact=True).count
    if count != len(plan.items):
        raise IndexBuildError(
            f"{plan.collection!r} holds {count} points, the plan {len(plan.items)} "
            "(points of another build? drop it and build again)"
        )
    return len(todo)


# --- command line -----------------------------------------------------------------------------


def _resolved(args: argparse.Namespace, s: Settings, session: Session) -> Resolved:
    """The variant whose index is built. Only the index matters here: a chunking variant's
    is the configured one with its chunk size; an embedding variant's is the chunking
    winner's (stored by `make eval-select STEP=chunking`, or `--base a7-...`) with its model."""
    from eval.ablations import base_for
    from eval.variants import find, resolve

    variant = find(args.variant, s)
    base: Resolved | None = None
    if variant.step == "embedding":
        if args.base:
            named = find(args.base, s)
            if named.step != "chunking":
                raise IndexBuildError(f"--base for an embedding variant is a7-...; got {args.base}")
            base = resolve(named, s)
        else:
            base = base_for("embedding", s, session, None)
    resolved = resolve(variant, s, base)
    if resolved.index == IndexSpec.configured(s):
        raise IndexBuildError(f"{args.variant} searches the app's own collection: nothing to build")
    return resolved


def _eval_collections(client: QdrantClient, s: Settings) -> list[str]:
    prefix = f"{s.qdrant_collection}{EVAL_MARK}"
    return sorted(c.name for c in client.get_collections().collections if c.name.startswith(prefix))


def main(argv: list[str] | None = None, *, dense: DenseEncoder | None = None,
         sparse: SparseEncoder | None = None,
         record_usage: UsageRecorder = record_embedding_usage) -> int:  # fmt: skip
    parser = argparse.ArgumentParser(prog="eval.indexes", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the eval collections and their point counts (free)")
    for name in ("estimate", "build", "drop"):
        p = sub.add_parser(name)
        p.add_argument("--variant", required=True)
        p.add_argument(
            "--base", default=None, help="the base variant, instead of the stored winner"
        )
        if name == "build":
            p.add_argument("--max-usd", default=None, help="hard spend cap for the embeddings")
    args = parser.parse_args(argv)

    from app.db.session import session_scope

    s = get_settings()
    client = get_qdrant()
    if args.command == "list":
        for name in _eval_collections(client, s):
            print(f"{name}: {client.count(name, exact=True).count} points")
        return 0
    try:
        with session_scope() as session:
            resolved = _resolved(args, s, session)
            plan = plan_index(session, resolved, s, client)
    except (IndexBuildError, LookupError, KeyError) as err:
        print(str(err), file=sys.stderr)
        return 2
    if args.command == "drop":
        if EVAL_MARK not in plan.collection or plan.collection == s.qdrant_collection:
            print(f"refusing to drop {plan.collection!r}", file=sys.stderr)
            return 2
        if client.collection_exists(plan.collection):
            client.delete_collection(plan.collection)
            print(f"dropped {plan.collection}")
        return 0
    print(
        f"{plan.collection}: {len(plan.items)} points; {len(plan.present)} already built, "
        f"{len(plan.to_copy)} copied with their vectors, {len(plan.to_embed)} to embed with "
        f"{plan.index.embedding_model}: {plan.tokens()} tokens in {plan.requests()} request(s), "
        f"about ${plan.cost_usd(s)}"
    )
    if args.command == "estimate":
        print("nothing was sent or written")
        return 0
    if plan.to_embed and not args.max_usd:
        print("this build embeds, so it needs --max-usd; nothing was written", file=sys.stderr)
        return 2

    from eval.budget import BudgetExhausted, install_budget
    from eval.runner import index_embedder

    budget = install_budget(Decimal(args.max_usd), s) if args.max_usd else None
    try:
        with applied_settings(resolved.overrides(s)):  # the variant's model, size and price
            encoder = dense or index_embedder(resolved.index, record_usage, None)
            embedded = build_index(
                plan, dense=encoder, sparse=sparse or get_sparse_embedder(), client=client,
                source_collection=s.qdrant_collection,
                progress=lambda line: print(line, file=sys.stderr, flush=True),
                after_batch=budget.check if budget is not None else lambda: None,
            )  # fmt: skip
    except IndexBuildError as err:
        print(str(err), file=sys.stderr)
        return 2
    except Exception as err:
        # A request the guard refused surfaces as a connection error of the OpenAI client.
        if isinstance(err, BudgetExhausted) or (budget is not None and budget.guard.tripped):
            print(f"stopped by the spend cap: {err}; run build again to resume", file=sys.stderr)
            return 5
        raise
    print(f"{plan.collection}: built, {embedded} points embedded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The configurations the ablation runner compares (docs/adr/0012).

Each configuration ("variant") is a set of settings overrides plus, for ablations 7 and 8,
the index it searches. Variants are grouped in steps; a step after the first builds on
the winner of the step before it (the plan's "best of the above"), chosen by the rule in
`eval/selection.py` on generated questions only:

    step        ablations  variants (the incumbent is marked *)
    retrieval   1-5        a1-dense, a2-bm25, a3-hybrid*, a4-rerank, a5-rerank-second
    rewriting   6          a6-rewrite* (the planner's rephrasings on), a6-no-rewrite
    chunking    7          a7-c256, a7-c512*, a7-c1024
    embedding   8          a8-small*, a8-large-short (large shortened to embedding_dim with
                           the `dimensions` parameter), a8-large (its own 3072 dimensions)
    reranker    (from M4)  r-reference* (a4's configuration), r-threads-4/2/1, r-arena-off,
                           r-candidates-30/20/10, r-small-model

The incumbent of a step is what the app runs today; a challenger replaces it only if the
rule finds it better. "Query rewriting" (ablation 6) is the planner's two rephrasings:
condense, the other rewriting stage, never runs for the first message of a conversation,
and every evaluation question is one.

Model names come from the settings (`reranker_model`, `eval_second_reranker_model`,
`embedding_model`, `eval_large_embedding_model`), never from this file.

An index other than the configured one (another chunk size or embedding model) lives in
its own Qdrant collection, named after what it holds (`eval/indexes.py` builds it):
`<qdrant_collection>__eval__c<tokens>o<overlap>__<model>-<dim>[-short]`.
"""

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from app.config import Settings

Step = Literal["retrieval", "rewriting", "chunking", "embedding", "reranker", "suite"]
Part = Literal["retrieval", "answers"]
STEPS: tuple[Step, ...] = ("retrieval", "rewriting", "chunking", "embedding", "reranker")
PARTS: tuple[Part, ...] = ("retrieval", "answers")
# The step whose winner a step builds on. The reranker step builds on a4 (the reranker as
# the app would run it), whatever won the retrieval step: it compares runtime settings.
PREVIOUS: dict[Step, Step] = {
    "rewriting": "retrieval",
    "chunking": "rewriting",
    "embedding": "chunking",
}
RERANKER_BASE = "a4-rerank"


@dataclass(frozen=True)
class IndexSpec:
    """What an index holds: the chunking and the dense embeddings of its points."""

    chunk_tokens: int
    chunk_overlap: int
    embedding_model: str
    embedding_dim: int
    # Sent to the embeddings API as `dimensions` (a shortened text-embedding-3 vector);
    # None sends nothing, as the app does.
    request_dimensions: int | None = None

    @classmethod
    def configured(cls, s: Settings) -> "IndexSpec":
        """The index the app searches (the `qdrant_collection`)."""
        return cls(s.chunk_tokens, s.chunk_overlap, s.embedding_model, s.embedding_dim)

    def same_chunking(self, s: Settings) -> bool:
        """Chunk ids mean the same chunks as in Postgres (so id-based metrics apply)."""
        return (self.chunk_tokens, self.chunk_overlap) == (s.chunk_tokens, s.chunk_overlap)

    def collection(self, s: Settings) -> str:
        if self == IndexSpec.configured(s):
            return s.qdrant_collection
        model = re.sub(r"[^a-z0-9]+", "-", self.embedding_model.lower()).strip("-")
        short = "-short" if self.request_dimensions else ""
        return (
            f"{s.qdrant_collection}__eval__c{self.chunk_tokens}o{self.chunk_overlap}"
            f"__{model}-{self.embedding_dim}{short}"
        )


def embedding_price(model: str, s: Settings) -> float:
    """USD per 1M tokens of an embedding model the variants use (from the settings)."""
    prices = {
        s.embedding_model: s.embedding_price_per_1m,
        s.eval_large_embedding_model: s.eval_large_embedding_price_per_1m,
    }
    if model not in prices:
        raise ValueError(f"no price configured for embedding model {model!r}")
    return prices[model]


def index_overrides(index: IndexSpec, s: Settings) -> dict[str, Any]:
    """The settings an index other than the configured one needs: its collection, and its
    embedding model, dimension and price, so that question embeddings use the same model
    as the points and llm_usage logs what they cost."""
    if index == IndexSpec.configured(s):
        return {}
    out: dict[str, Any] = {"qdrant_collection": index.collection(s)}
    if index.embedding_model != s.embedding_model:
        out["embedding_model"] = index.embedding_model
        out["embedding_price_per_1m"] = embedding_price(index.embedding_model, s)
    if index.embedding_dim != s.embedding_dim:
        out["embedding_dim"] = index.embedding_dim
    if not index.same_chunking(s):
        out["chunk_tokens"] = index.chunk_tokens
        out["chunk_overlap"] = index.chunk_overlap
    return out


@dataclass(frozen=True)
class Variant:
    id: str
    step: Step
    label: str
    ablation: int | None = None  # the plan's ablation number (1-8); None for the others
    overrides: Mapping[str, Any] = field(default_factory=dict)  # settings, on top of the base
    index: Mapping[str, Any] = field(default_factory=dict)  # IndexSpec fields, on top of the base
    incumbent: bool = False
    parts: tuple[Part, ...] = PARTS


def all_variants(s: Settings) -> tuple[Variant, ...]:
    """Every variant, in the order each step lists them (the order breaks exact ties)."""
    no_rerank = {"rerank_enabled": False}
    rerank = {"search_mode": "hybrid", "rerank_enabled": True}
    large = s.eval_large_embedding_model
    return (
        # 1-5: the retrieval stages, each on the configured settings.
        Variant("a1-dense", "retrieval", "dense only", 1, {"search_mode": "dense", **no_rerank}),
        Variant("a2-bm25", "retrieval", "BM25 only", 2, {"search_mode": "bm25", **no_rerank}),
        Variant("a3-hybrid", "retrieval", "hybrid (RRF)", 3, {"search_mode": "hybrid", **no_rerank},
                incumbent=True),
        Variant("a4-rerank", "retrieval", f"hybrid + rerank ({s.reranker_model})", 4,
                {**rerank, "reranker_model": s.reranker_model}),
        Variant("a5-rerank-second", "retrieval",
                f"hybrid + rerank ({s.eval_second_reranker_model})", 5,
                {**rerank, "reranker_model": s.eval_second_reranker_model}),
        # 6: on the winner of 1-5, with and without the planner's rephrasings.
        Variant("a6-rewrite", "rewriting", "with query rewriting", 6,
                {"plan_rephrasings_enabled": True}, incumbent=True),
        Variant("a6-no-rewrite", "rewriting", "without query rewriting", 6,
                {"plan_rephrasings_enabled": False}),
        # 7: on the winner of 6, chunk sizes; the overlap keeps the configured 1/8 share.
        Variant("a7-c256", "chunking", "chunks of 256 tokens", 7,
                index={"chunk_tokens": 256, "chunk_overlap": 32}),
        Variant("a7-c512", "chunking", "chunks of 512 tokens", 7, incumbent=True),
        Variant("a7-c1024", "chunking", "chunks of 1024 tokens", 7,
                index={"chunk_tokens": 1024, "chunk_overlap": 128}),
        # 8: on the winner of 7, the embedding model and its dimensions.
        Variant("a8-small", "embedding", f"{s.embedding_model} at {s.embedding_dim}", 8,
                incumbent=True),
        Variant("a8-large-short", "embedding", f"{large} shortened to {s.embedding_dim}", 8,
                index={"embedding_model": large, "embedding_dim": s.embedding_dim,
                       "request_dimensions": s.embedding_dim}),
        Variant("a8-large", "embedding", f"{large} at {s.eval_large_embedding_dim}", 8,
                index={"embedding_model": large, "embedding_dim": s.eval_large_embedding_dim}),
        # The reranker comparisons moved from M4, on a4's configuration. Threads and the
        # arena change speed and memory, not the ranking: their retrieval lists must equal
        # the reference's, so they need no paid answer run. Fewer candidates and the small
        # model change the ranking, so their answers are measured too.
        Variant("r-reference", "reranker", "bge, default threads, arena on, 50 candidates",
                incumbent=True),
        *(
            Variant(f"r-threads-{n}", "reranker", f"{n} ONNX thread(s)",
                    overrides={"rerank_threads": n}, parts=("retrieval",))
            for n in (4, 2, 1)
        ),
        Variant("r-arena-off", "reranker", "CPU memory arena off",
                overrides={"rerank_cpu_arena": False}, parts=("retrieval",)),
        *(
            Variant(f"r-candidates-{k}", "reranker", f"{k} candidates reranked",
                    overrides={"search_candidates": k})
            for k in (30, 20, 10)
        ),
        Variant("r-small-model", "reranker", f"{s.eval_second_reranker_model}",
                overrides={"reranker_model": s.eval_second_reranker_model}),
    )  # fmt: skip


CONFIGURED = Variant("configured", "suite", "the configured pipeline", incumbent=True)


def variants_of(step: Step, s: Settings) -> list[Variant]:
    return [v for v in all_variants(s) if v.step == step]


def find(variant_id: str, s: Settings) -> Variant:
    if variant_id == CONFIGURED.id:
        return CONFIGURED
    for v in all_variants(s):
        if v.id == variant_id:
            return v
    raise KeyError(f"unknown variant {variant_id!r}")


@dataclass(frozen=True)
class Resolved:
    """A variant on its base: the settings it runs with and the index it searches."""

    variant: Variant
    settings: dict[str, Any]  # every setting changed, except those the index implies
    index: IndexSpec
    base: str | None = None  # the variant id of the base ("best of the above")

    def overrides(self, s: Settings) -> dict[str, Any]:
        """What `eval.scope.applied_settings` gets: the settings plus the index's."""
        return {**self.settings, **index_overrides(self.index, s)}

    def describe(self) -> dict[str, Any]:
        return {
            "variant": self.variant.id,
            "step": self.variant.step,
            "ablation": self.variant.ablation,
            "label": self.variant.label,
            "settings": dict(self.settings),
            "index": asdict(self.index),
            "base": self.base,
        }


def resolve(variant: Variant, s: Settings, base: Resolved | None = None) -> Resolved:
    """`variant` on top of `base` (None: on the configured settings and index)."""
    index = replace(base.index if base else IndexSpec.configured(s), **variant.index)
    settings = {**(base.settings if base else {}), **variant.overrides}
    return Resolved(variant, settings, index, base.variant.id if base else None)


def resolved_from_record(record: Mapping[str, Any], s: Settings) -> Resolved:
    """A Resolved back from `describe()` (as stored with a selection)."""
    return Resolved(
        find(str(record["variant"]), s),
        dict(record["settings"]),
        IndexSpec(**record["index"]),
        record.get("base"),
    )

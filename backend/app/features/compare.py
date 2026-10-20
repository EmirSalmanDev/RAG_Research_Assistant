"""Paper comparison: one scoped retrieval per paper and dimension, one structured call.

    POST /compare  2-5 papers x a subset of six dimensions  ->  a table of cited cells

How it works (docs/adr/0010):
1. The dimension questions (config) are the same for every paper, so they are embedded
   in ONE batched embeddings request, however many papers are compared.
2. For each (paper, dimension) one hybrid query runs against Qdrant, filtered to that
   paper. Only `abstract` and `section` chunks are kept: generated `analysis` chunks are
   model output, and a citation must point at the paper's own words. `search()` has no
   chunk_type filter, so a few extra hits are fetched and the rest is dropped here.
3. The chunks are de-duplicated into ONE numbered source list [1]..[N], numbered paper
   by paper. If the input would exceed the token cap, the per-cell top-k is reduced
   3 -> 2 -> 1 (the same rule every time).
4. ONE Structured Outputs call fills every cell with a short text and source numbers.
5. Code, not the model, decides what is shown (`validate_fill`): a cited number must
   exist AND belong to a source of the cell's own paper; invalid numbers are dropped;
   a cell left with text but no valid citation becomes `not_stated`. An uncited claim is
   never displayed.

Nothing is stored. Every request writes its own llm_usage row.
"""

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, get_args

from pydantic import BaseModel
from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.features.deps import ExploreDeps, SearchFn, end_span, upstream
from app.features.errors import InvalidRequest, UnknownPapers, UpstreamError
from app.features.paper_facts import PaperFacts
from app.features.spend import EmbeddingMeter, Spend
from app.features.text import cut_to_tokens
from app.ids import abs_url
from app.ingest.embed import SparseEncoder
from app.ingest.tokens import count_tokens
from app.llm.client import MESSAGE_OVERHEAD_TOKENS, ChatMessage, ChatResponseError
from app.rag.citations import Source, format_sources
from app.rag.prompts import load_prompt
from app.rag.retrieve import RetrievalConfig
from app.rag.search import SearchFilters, SearchHit

Dimension = Literal["task", "method", "data", "metrics", "compute", "limitations"]
DIMENSIONS: tuple[Dimension, ...] = get_args(Dimension)
MIN_PAPERS = 2  # the API contract (config holds the tunables, not the shape of the request)
MAX_PAPERS = 5

NOT_STATED = "Not stated in the sources."  # what an unsupported cell says
EMBED_FEATURE = "embed-compare"  # llm_usage.feature of the one batched question embedding
FILL_FEATURE = "compare-fill"  # llm_usage.feature of the structured call

_CITATION_MARKER = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")

# --- the model's answer (Structured Outputs: every field required) -------------------------


class CellFill(BaseModel):
    dimension: Dimension
    text: str
    source_numbers: list[int]


class PaperFill(BaseModel):
    arxiv_id: str
    cells: list[CellFill]


class CompareFill(BaseModel):
    papers: list[PaperFill]


# --- the API response ------------------------------------------------------------------------


class CitationOut(BaseModel):
    n: int
    chunk_id: uuid.UUID
    chunk_type: str  # abstract | section
    section_path: str | None


class CellOut(BaseModel):
    dimension: Dimension
    text: str  # NOT_STATED when status is not_stated: no claim is ever shown without a citation
    status: Literal["cited", "not_stated"]
    citations: list[CitationOut]


class RowOut(BaseModel):
    arxiv_id: str
    title: str
    has_fulltext: bool  # False: the paper is known by its abstract only
    abs_url: str | None
    cells: list[CellOut]  # in the order of `dimensions`


class SourceOut(BaseModel):
    n: int
    chunk_id: uuid.UUID
    arxiv_id: str
    chunk_type: str
    section_path: str | None
    title: str
    published_at: datetime
    abs_url: str | None


class CompareNotes(BaseModel):
    invalid_citations_dropped: int  # numbers that did not exist or belonged to another paper
    uncited_cells_converted: int  # cells that had text but no valid citation
    missing_cells: int  # cells the model did not return at all
    truncated_cells: int  # texts cut to the word limit
    cut_sources: int  # sources shortened before being shown to the model
    top_k_used: int  # chunks per cell after the token cap rule


class CompareOut(BaseModel):
    dimensions: list[Dimension]
    rows: list[RowOut]
    sources: list[SourceOut]
    notes: CompareNotes
    tokens_in: int
    tokens_out: int
    cost_usd: float
    prompt_version: str


# --- validation of the request (before any paid call) ----------------------------------------


def check_request(arxiv_ids: Sequence[str], dimensions: Sequence[str]) -> list[Dimension]:
    """Raise InvalidRequest for a bad request; return the dimensions in canonical order."""
    if not MIN_PAPERS <= len(arxiv_ids) <= MAX_PAPERS:
        raise InvalidRequest(f"pick {MIN_PAPERS} to {MAX_PAPERS} papers, got {len(arxiv_ids)}")
    if len(set(arxiv_ids)) != len(arxiv_ids):
        raise InvalidRequest("a paper was picked twice")
    unknown = sorted(set(dimensions) - set(DIMENSIONS))
    if unknown:
        raise InvalidRequest("unknown dimension(s): " + ", ".join(unknown))
    if not dimensions:
        raise InvalidRequest("pick at least one dimension")
    wanted = set(dimensions)
    return [d for d in DIMENSIONS if d in wanted]


# --- retrieval ---------------------------------------------------------------------------------


class _Precomputed:
    """A DenseEncoder over vectors that were embedded already. `search()` embeds its
    question itself; handing it the vectors of the one batched request keeps the
    embedding count at one without changing the search code."""

    def __init__(self, vectors: Mapping[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


CellKey = tuple[str, Dimension]  # (arxiv_id, dimension)


def retrieve_cells(
    arxiv_ids: Sequence[str],
    dimensions: Sequence[Dimension],
    *,
    queries: Mapping[str, str],
    vectors: Mapping[str, list[float]],
    sparse: SparseEncoder | None,
    client: QdrantClient,
    config: RetrievalConfig,
    search_fn: SearchFn,
    settings: Settings,
) -> dict[CellKey, list[SearchHit]]:
    """For each cell the best evidence chunks of its own paper, best first (at most
    `compare_top_k`). Rerank is not used (docs/adr/0010)."""
    dense = _Precomputed(vectors) if vectors else None
    cells: dict[CellKey, list[SearchHit]] = {}
    for arxiv_id in arxiv_ids:
        for dimension in dimensions:
            hits = search_fn(
                queries[dimension],
                SearchFilters(arxiv_ids=(arxiv_id,)),
                mode=config.mode,
                limit=settings.compare_fetch_limit,
                prefetch_limit=config.prefetch_limit,
                rrf_k=config.rrf_k,
                dense=dense,
                sparse=sparse,
                client=client,
            )
            evidence = [
                h
                for h in hits
                if h.arxiv_id == arxiv_id and h.chunk_type in settings.compare_evidence_types
            ]
            cells[(arxiv_id, dimension)] = evidence[: settings.compare_top_k]
    return cells


# --- the numbered source list ------------------------------------------------------------------


@dataclass(frozen=True)
class SourceList:
    sources: list[Source]  # numbered 1..N, paper by paper
    by_cell: dict[CellKey, list[int]]  # which numbers each cell retrieved
    cut: int  # sources shortened to compare_source_max_tokens


def assemble_sources(
    cells: Mapping[CellKey, list[SearchHit]],
    arxiv_ids: Sequence[str],
    dimensions: Sequence[Dimension],
    *,
    top_k: int,
    source_max_tokens: int,
) -> SourceList:
    """De-duplicate the cells' chunks (the first `top_k` of each) into one numbered list.
    The order is fixed: papers as requested, dimensions in canonical order, best first."""
    number_of: dict[uuid.UUID, int] = {}
    sources: list[Source] = []
    by_cell: dict[CellKey, list[int]] = {}
    cut = 0
    for arxiv_id in arxiv_ids:
        for dimension in dimensions:
            numbers: list[int] = []
            for hit in cells.get((arxiv_id, dimension), [])[:top_k]:
                if hit.chunk_id not in number_of:
                    text, was_cut = cut_to_tokens(hit.text, source_max_tokens)
                    cut += was_cut
                    number_of[hit.chunk_id] = len(sources) + 1
                    sources.append(Source(len(sources) + 1, replace(hit, text=text)))
                numbers.append(number_of[hit.chunk_id])
            by_cell[(arxiv_id, dimension)] = numbers
    return SourceList(sources, by_cell, cut)


def build_messages(
    arxiv_ids: Sequence[str],
    dimensions: Sequence[Dimension],
    sources: Sequence[Source],
    prompt: str,
) -> list[ChatMessage]:
    # The ids and dimension names are validated (database / fixed list), not paper text.
    user = (
        f"Papers (arXiv ids): {', '.join(arxiv_ids)}\n"
        f"Dimensions: {', '.join(dimensions)}\n\n"
        f"{format_sources(sources)}"
    )
    return [
        {"role": "system", "content": load_prompt(prompt).text},
        {"role": "user", "content": user},
    ]


def input_tokens(messages: Sequence[ChatMessage]) -> int:
    return sum(count_tokens(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages)


def fit_to_cap(
    cells: Mapping[CellKey, list[SearchHit]],
    arxiv_ids: Sequence[str],
    dimensions: Sequence[Dimension],
    settings: Settings,
) -> tuple[SourceList, list[ChatMessage], int]:
    """The sources and messages for the largest top-k (3, then 2, then 1) whose input
    fits `compare_max_input_tokens`. At top-k 1 it is used whatever its size: the input
    is then bounded by (papers x dimensions) sources of `compare_source_max_tokens`."""
    k = settings.compare_top_k
    while True:
        listing = assemble_sources(
            cells,
            arxiv_ids,
            dimensions,
            top_k=k,
            source_max_tokens=settings.compare_source_max_tokens,
        )
        messages = build_messages(arxiv_ids, dimensions, listing.sources, settings.compare_prompt)
        if k == 1 or input_tokens(messages) <= settings.compare_max_input_tokens:
            return listing, messages, k
        k -= 1


# --- what the model returned -> what is shown --------------------------------------------------


def _words_cut(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words), False
    return " ".join(words[:max_words]).rstrip(" ,;:.-") + "…", True


def _is_not_stated(text: str) -> bool:
    return text.strip().rstrip(".").lower() in ("", NOT_STATED.rstrip(".").lower())


@dataclass
class _Counts:
    invalid_citations_dropped: int = 0
    uncited_cells_converted: int = 0
    missing_cells: int = 0
    truncated_cells: int = 0


def validate_fill(
    fill: CompareFill,
    arxiv_ids: Sequence[str],
    dimensions: Sequence[Dimension],
    sources: Sequence[Source],
    *,
    max_words: int,
) -> tuple[dict[CellKey, CellOut], _Counts]:
    """Turn the model's table into the cells to show. The guarantee: a cell is `cited`
    only if it has text AND at least one citation that exists and belongs to a source of
    the cell's own paper. Everything else is `not_stated` with no claim."""
    by_number = {s.n: s for s in sources}
    counts = _Counts()
    returned: dict[CellKey, CellFill] = {}
    for paper in fill.papers:
        if paper.arxiv_id not in arxiv_ids:
            continue  # an invented paper id: ignored
        for filled in paper.cells:
            returned.setdefault((paper.arxiv_id, filled.dimension), filled)  # the first wins
    out: dict[CellKey, CellOut] = {}
    for arxiv_id in arxiv_ids:
        for dimension in dimensions:
            cell = returned.get((arxiv_id, dimension))
            if cell is None:
                counts.missing_cells += 1
                out[(arxiv_id, dimension)] = _not_stated(dimension)
                continue
            valid: list[int] = []
            for n in cell.source_numbers:
                source = by_number.get(n)
                if source is None or source.hit.arxiv_id != arxiv_id:
                    counts.invalid_citations_dropped += 1
                elif n not in valid:
                    valid.append(n)
            text = _CITATION_MARKER.sub("", cell.text).strip()
            if _is_not_stated(text):
                out[(arxiv_id, dimension)] = _not_stated(dimension)
                continue
            if not valid:
                counts.uncited_cells_converted += 1
                out[(arxiv_id, dimension)] = _not_stated(dimension)
                continue
            text, was_cut = _words_cut(text, max_words)
            counts.truncated_cells += was_cut
            out[(arxiv_id, dimension)] = CellOut(
                dimension=dimension,
                text=text,
                status="cited",
                citations=[
                    CitationOut(
                        n=n,
                        chunk_id=by_number[n].hit.chunk_id,
                        chunk_type=by_number[n].hit.chunk_type,
                        section_path=by_number[n].hit.section_path,
                    )
                    for n in valid
                ],
            )
    return out, counts


def _not_stated(dimension: Dimension) -> CellOut:
    return CellOut(dimension=dimension, text=NOT_STATED, status="not_stated", citations=[])


# --- the whole request ---------------------------------------------------------------------------


CompareDeps = ExploreDeps  # the same collaborators serve compare and timeline


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), ROUND_HALF_UP))


def run_compare(
    arxiv_ids: Sequence[str], dimensions: Sequence[str], *, deps: CompareDeps
) -> CompareOut:
    """Compare the papers. Raises InvalidRequest / UnknownPapers before any paid call,
    OpenAIKeyMissing when there is no key, UpstreamError when OpenAI or Qdrant fails."""
    settings = get_settings()
    wanted = check_request(arxiv_ids, dimensions)
    facts = deps.facts.get_facts(list(arxiv_ids))
    missing = [a for a in arxiv_ids if a not in facts]
    if missing:
        raise UnknownPapers(missing)
    missing_queries = [d for d in wanted if d not in settings.compare_dimension_queries]
    if missing_queries:
        raise InvalidRequest("no question configured for: " + ", ".join(missing_queries))
    queries: dict[str, str] = {d: settings.compare_dimension_queries[d] for d in wanted}

    spend = Spend()
    trace = deps.tracer.start_trace(
        "compare", input={"arxiv_ids": list(arxiv_ids), "dimensions": wanted}, metadata={}
    )
    prompt = load_prompt(settings.compare_prompt)
    # --- retrieve
    span = trace.span("compare-retrieve", input={"queries": queries})
    try:
        cells = _retrieve(arxiv_ids, wanted, queries, deps, settings, spend)
    except Exception as err:
        end_span(span, error=err)
        trace.end(output="error")
        mapped = upstream(err)
        if mapped is err:
            raise
        raise mapped from err
    listing, messages, top_k = fit_to_cap(cells, arxiv_ids, wanted, settings)
    end_span(
        span,
        embedding_requests=spend.embedding_requests,
        cells=len(cells),
        sources=len(listing.sources),
        top_k_used=top_k,
        input_tokens=input_tokens(messages),
    )
    # --- fill
    span = trace.span("compare-fill", input={"prompt": prompt.id, "sources": len(listing.sources)})
    try:
        fill, usage = deps.chat.parse(
            messages,
            CompareFill,
            feature=FILL_FEATURE,
            max_tokens=settings.compare_max_output_tokens,
        )
    except ChatResponseError as err:
        if err.usage is not None:  # billed even though the output was unusable
            spend.add_chat(err.usage)
        end_span(span, error=err)
        trace.end(output="error")
        raise UpstreamError(f"the model did not return a usable table: {err}") from err
    except Exception as err:
        end_span(span, error=err)
        trace.end(output="error")
        mapped = upstream(err)
        if mapped is err:
            raise
        raise mapped from err
    spend.add_chat(usage)
    shown, counts = validate_fill(
        fill, arxiv_ids, wanted, listing.sources, max_words=settings.compare_cell_max_words
    )
    span.end(
        usage={"input": usage.tokens_in, "output": usage.tokens_out},
        metadata={
            "invalid_citations_dropped": counts.invalid_citations_dropped,
            "uncited_cells_converted": counts.uncited_cells_converted,
            "missing_cells": counts.missing_cells,
            "cited_cells": sum(c.status == "cited" for c in shown.values()),
        },
    )
    trace.end(output="ok", metadata={"prompt": prompt.id})
    return CompareOut(
        dimensions=wanted,
        rows=[_row(facts[a], [shown[(a, d)] for d in wanted]) for a in arxiv_ids],
        sources=[_source_out(s) for s in listing.sources],
        notes=CompareNotes(
            invalid_citations_dropped=counts.invalid_citations_dropped,
            uncited_cells_converted=counts.uncited_cells_converted,
            missing_cells=counts.missing_cells,
            truncated_cells=counts.truncated_cells,
            cut_sources=listing.cut,
            top_k_used=top_k,
        ),
        tokens_in=spend.tokens_in,
        tokens_out=spend.tokens_out,
        cost_usd=_money(spend.cost_usd),
        prompt_version=prompt.id,
    )


def _retrieve(
    arxiv_ids: Sequence[str],
    dimensions: Sequence[Dimension],
    queries: Mapping[str, str],
    deps: CompareDeps,
    settings: Settings,
    spend: Spend,
) -> dict[CellKey, list[SearchHit]]:
    config = RetrievalConfig.from_settings()
    vectors: dict[str, list[float]] = {}
    if config.mode != "bm25":  # ONE request for all dimension questions
        texts = [queries[d] for d in dimensions]
        embedded = deps.dense(EmbeddingMeter(spend, deps.record_embedding)).embed(
            texts, EMBED_FEATURE
        )
        vectors = dict(zip(texts, embedded, strict=True))
    sparse = deps.sparse() if config.mode != "dense" else None
    return retrieve_cells(
        arxiv_ids,
        dimensions,
        queries=queries,
        vectors=vectors,
        sparse=sparse,
        client=deps.qdrant(),
        config=config,
        search_fn=deps.search_fn,
        settings=settings,
    )


def _row(facts: PaperFacts, cells: list[CellOut]) -> RowOut:
    return RowOut(
        arxiv_id=facts.arxiv_id,
        title=facts.title,
        has_fulltext=facts.has_fulltext,
        abs_url=abs_url(facts.arxiv_id),
        cells=cells,
    )


def _source_out(source: Source) -> SourceOut:
    h = source.hit
    return SourceOut(
        n=source.n,
        chunk_id=h.chunk_id,
        arxiv_id=h.arxiv_id,
        chunk_type=h.chunk_type,
        section_path=h.section_path,
        title=h.title,
        published_at=datetime.fromisoformat(h.published_at),
        abs_url=abs_url(h.arxiv_id),
    )

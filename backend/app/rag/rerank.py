"""Rerank stage: rescore the search candidates with a cross-encoder, then cap per paper.

Why a second model? Search embeds the question and each chunk SEPARATELY and compares
two vectors, a summary of each. A cross-encoder reads the question and one chunk
TOGETHER and outputs one relevance score. That is slower (one model run per pair), but
much better at telling which of 50 candidates actually answers the question. So search
finds ~50 candidates fast, and the cross-encoder orders them carefully.

The diversity cap then keeps at most N chunks per paper, so one long paper cannot fill
every slot of the answer's context. Generated `analysis` chunks are capped on their own
(docs/adr/0012): they are a model's summary of a paper, not the paper's words, so they
never use up the paper's slots for its own text, and one is enough.

The model (`reranker_model`, ~1 GB for bge-reranker-base) is NEVER downloaded at
runtime: it is always loaded with `local_files_only=True` from `fastembed_cache_dir`.
Only `python -m app.rag.rerank fetch` (`make fetch-reranker`) downloads it, a deliberate
step that needs network access. FastEmbed 0.8.1's cache check for this model only needs
`onnx/model.onnx`, so a complete cache loads without any network access (unlike BM25,
whose check asks for a file the Hub repo lacks; see docs/adr/0005).
"""

import argparse
import json
import sys
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from app.config import get_settings
from app.ingest.chunk import embedding_text
from app.rag.search import SearchHit


class CrossEncoder(Protocol):
    """The one method of FastEmbed's TextCrossEncoder we use (tests pass fakes)."""

    def rerank(
        self, query: str, documents: Iterable[str], batch_size: int = ...
    ) -> Iterable[float]: ...


class RerankerUnavailable(RuntimeError):
    """The reranker's files are not in the local cache (run `make fetch-reranker`)."""


def rerank(
    query: str, hits: Sequence[SearchHit], encoder: CrossEncoder, *, batch_size: int
) -> list[SearchHit]:
    """Rescore `hits` with the cross-encoder; best first.

    Each hit's score becomes the cross-encoder's score (a raw relevance logit: higher is
    more relevant; only the order matters). The model reads the same 'Title > Section'
    + text that was embedded, so a chunk keeps its context. Equal scores keep the search
    order (Python's sort is stable).
    """
    if not hits:
        return []
    documents = [embedding_text(h.title, h.section_path, h.text) for h in hits]
    scores = list(encoder.rerank(query, documents, batch_size=batch_size))
    if len(scores) != len(hits):
        raise RuntimeError(f"the reranker returned {len(scores)} scores for {len(hits)} chunks")
    rescored = [replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)]
    return sorted(rescored, key=lambda h: h.score, reverse=True)


ANALYSIS = "analysis"  # the chunk_type of a generated paper analysis (docs/adr/0008)


def cap_per_paper(
    hits: Sequence[SearchHit], max_per_paper: int, *, max_analysis_per_paper: int
) -> list[SearchHit]:
    """Keep at most `max_per_paper` evidence chunks and at most `max_analysis_per_paper`
    analysis chunks of each paper, in the given order.

    Evidence chunks are the paper's own text (`abstract`, `section`); `max_per_paper` 0
    means no cap on them. Analysis chunks are counted apart, so they never take a slot
    of the paper's own text: 0 drops them all, 1 keeps the best one, and a negative
    value turns the separate count off, so that they count toward `max_per_paper` like
    any chunk (the behaviour before docs/adr/0012).

    Walks the list best first and skips a chunk once its paper already has that many
    chunks of its kind above it, so the best chunks of every paper survive. Nothing is
    reordered.
    """
    separate = max_analysis_per_paper >= 0
    evidence: Counter[str] = Counter()
    analysis: Counter[str] = Counter()
    kept: list[SearchHit] = []
    for hit in hits:
        if separate and hit.chunk_type == ANALYSIS:
            if analysis[hit.arxiv_id] < max_analysis_per_paper:
                analysis[hit.arxiv_id] += 1
                kept.append(hit)
        elif max_per_paper <= 0 or evidence[hit.arxiv_id] < max_per_paper:
            evidence[hit.arxiv_id] += 1
            kept.append(hit)
    return kept


# --- loading the model ------------------------------------------------------------------------


def model_cache_dir() -> Path:
    """The configured FastEmbed cache directory ('~' expanded)."""
    return Path(get_settings().fastembed_cache_dir).expanduser()


def load_cross_encoder(
    model_name: str, cache_dir: Path, *, threads: int | None = None, cpu_arena: bool = True
) -> CrossEncoder:
    """Load the model from the local cache only; never downloads.

    `threads` limits ONNX Runtime's threads (None: its default, one per core) and
    `cpu_arena=False` switches its CPU memory arena off. Both are passed to FastEmbed only
    when they differ from its defaults, so the default load is exactly the one measured
    in docs/adr/0005.

    A missing or incomplete cache raises RerankerUnavailable at once (FastEmbed raises
    ValueError; with local_files_only=True it does not try the network first).
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # loads onnxruntime

    if threads is not None and threads < 1:
        raise ValueError(f"rerank threads must be at least 1, got {threads}")
    options: dict[str, Any] = {}
    if threads is not None:
        options["threads"] = threads
    if not cpu_arena:
        options["enable_cpu_mem_arena"] = False  # an ONNX session option FastEmbed exposes
    try:
        return TextCrossEncoder(
            model_name, cache_dir=str(cache_dir), local_files_only=True, **options
        )
    except ValueError as err:
        raise RerankerUnavailable(
            f"reranker {model_name!r} is not in {cache_dir}. Download it once with "
            "`make fetch-reranker` (network access, about 1 GB; ask first)."
        ) from err


@dataclass(frozen=True)
class RerankerKey:
    """What a loaded cross-encoder was built from: the model and its runtime settings."""

    model: str
    cache_dir: str
    threads: int | None
    cpu_arena: bool


def reranker_key() -> RerankerKey:
    """The cross-encoder the current settings ask for."""
    s = get_settings()
    return RerankerKey(
        s.reranker_model, str(model_cache_dir()), s.rerank_threads, s.rerank_cpu_arena
    )


_lock = threading.Lock()
_loaded: tuple[RerankerKey, CrossEncoder] | None = None


def get_cross_encoder() -> CrossEncoder:
    """One model per process, loaded on first use.

    The lock matters in the API: requests run in a thread pool, and without it two
    first requests at the same moment would each load a copy (about 1 GB of RAM each).

    The model is kept together with the settings it was loaded with. The API never
    changes them, but an evaluation runs several configurations in one process: when the
    settings ask for another model (or other runtime settings), the old one is released
    before the new one is loaded, so a configuration never reranks with the model of the
    one before, and two models never sit in memory together.
    """
    global _loaded
    key = reranker_key()
    with _lock:
        if _loaded is None or _loaded[0] != key:
            _loaded = None  # drop the reference first, so the old model can be freed
            encoder = load_cross_encoder(
                key.model, Path(key.cache_dir), threads=key.threads, cpu_arena=key.cpu_arena
            )
            _loaded = (key, encoder)
        return _loaded[1]


def loaded_reranker() -> RerankerKey | None:
    """The key of the cross-encoder this process holds, if any (the evaluation records it)."""
    with _lock:
        return None if _loaded is None else _loaded[0]


# --- the one download -------------------------------------------------------------------------


def fetch(model_name: str, cache_dir: Path) -> dict[str, object]:
    """Download the model files into `cache_dir`. NETWORK: the only function here that
    goes online. Returns what is now in the cache (file names and sizes in bytes)."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    supported = {d["model"] for d in TextCrossEncoder.list_supported_models()}
    if model_name not in supported:
        raise ValueError(f"{model_name!r} is not a FastEmbed cross-encoder: {sorted(supported)}")
    # Constructing it downloads the files; lazy_load=True skips loading them into RAM.
    TextCrossEncoder(model_name, cache_dir=str(cache_dir), lazy_load=True)
    # Hugging Face cache layout: models--<org>--<name>/snapshots/<commit>/<files>, where
    # the files are links into blobs/. Sizes are read through the links.
    repo = cache_dir / f"models--{model_name.replace('/', '--')}"
    files = {
        str(p.relative_to(repo)): p.stat().st_size
        for p in sorted((repo / "snapshots").rglob("*"))
        if p.is_file()
    }
    return {"model": model_name, "cache_dir": str(cache_dir), "files": files,
            "total_bytes": sum(files.values())}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.rag.rerank")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="download the configured reranker (network, ~1 GB)")
    parser.parse_args(argv)
    try:
        summary = fetch(get_settings().reranker_model, model_cache_dir())
    except Exception as err:
        print(f"fetch failed: {type(err).__name__}: {err}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

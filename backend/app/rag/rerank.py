"""Rerank stage: rescore the search candidates with a cross-encoder, then cap per paper.

Why a second model? Search embeds the question and each chunk SEPARATELY and compares
two vectors, a summary of each. A cross-encoder reads the question and one chunk
TOGETHER and outputs one relevance score. That is slower (one model run per pair), but
much better at telling which of 50 candidates actually answers the question. So search
finds ~50 candidates fast, and the cross-encoder orders them carefully.

The diversity cap then keeps at most N chunks per paper, so one long paper cannot fill
every slot of the answer's context.

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
from dataclasses import replace
from pathlib import Path
from typing import Protocol

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


def cap_per_paper(hits: Sequence[SearchHit], max_per_paper: int) -> list[SearchHit]:
    """Keep at most `max_per_paper` chunks of each paper, in the given order (0 = no cap).

    Walks the list best first and skips a chunk once its paper already has
    `max_per_paper` chunks above it, so the best chunks of every paper survive.
    """
    if max_per_paper <= 0:
        return list(hits)
    seen: Counter[str] = Counter()
    kept: list[SearchHit] = []
    for hit in hits:
        if seen[hit.arxiv_id] < max_per_paper:
            seen[hit.arxiv_id] += 1
            kept.append(hit)
    return kept


# --- loading the model ------------------------------------------------------------------------


def model_cache_dir() -> Path:
    """The configured FastEmbed cache directory ('~' expanded)."""
    return Path(get_settings().fastembed_cache_dir).expanduser()


def load_cross_encoder(model_name: str, cache_dir: Path) -> CrossEncoder:
    """Load the model from the local cache only; never downloads.

    A missing or incomplete cache raises RerankerUnavailable at once (FastEmbed raises
    ValueError; with local_files_only=True it does not try the network first).
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # loads onnxruntime

    try:
        return TextCrossEncoder(model_name, cache_dir=str(cache_dir), local_files_only=True)
    except ValueError as err:
        raise RerankerUnavailable(
            f"reranker {model_name!r} is not in {cache_dir}. Download it once with "
            "`make fetch-reranker` (network access, about 1 GB; ask first)."
        ) from err


_lock = threading.Lock()
_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    """One model per process, loaded on first use.

    The lock matters in the API: requests run in a thread pool, and without it two
    first requests at the same moment would each load a copy (about 1 GB of RAM each).
    """
    global _encoder
    with _lock:
        if _encoder is None:
            _encoder = load_cross_encoder(get_settings().reranker_model, model_cache_dir())
        return _encoder


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

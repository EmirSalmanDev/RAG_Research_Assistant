"""Command line entry, used by `make ingest-one` (which runs it in the worker container):

    python -m app.ingest.cli one <arxiv_id> [--force-source hf|html|pdf] [--metadata-only]

`one` ingests a paper end to end: metadata and enrichment, the abstract chunk, the full
text (Hugging Face .md, then arXiv HTML, then the PDF; or only the forced source),
cleaning, chunking, embedding and indexing. `--metadata-only` keeps the old behaviour
(metadata and enrichment only: no full text, no OpenAI calls).
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from typing import Any

from app.db.session import session_scope
from app.ingest.fulltext import FulltextError
from app.ingest.index import close_qdrant
from app.ingest.paper import IngestError, default_clients, ingest_one


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.ingest.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("one", help="ingest a single paper (metadata, abstract, full text)")
    one.add_argument("arxiv_id")
    one.add_argument(
        "--force-source",
        choices=["hf", "html", "pdf"],
        help="use only this full-text source instead of the fallback chain",
    )
    one.add_argument(
        "--metadata-only",
        action="store_true",
        help="only metadata and enrichment (no full text, no embeddings)",
    )
    args = parser.parse_args(argv)
    if args.metadata_only and args.force_source:
        parser.error("--force-source has no effect with --metadata-only")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    output: dict[str, Any]
    try:
        with session_scope() as session:
            if args.metadata_only:
                output = ingest_one(session, default_clients(), args.arxiv_id).model_dump()
            else:
                output = _run_full(session, args.arxiv_id, args.force_source)
    except (IngestError, FulltextError, ValueError) as err:
        print(f"ingest-one failed: {err}", file=sys.stderr)
        return 1
    finally:
        close_qdrant()  # close the shared Qdrant client (if any) before the process exits
    print(json.dumps(output, indent=2, default=str))
    return 0  # optional-source failures are in the output's errors but do not fail the command


def _run_full(session: Any, arxiv_id: str, force_source: str | None) -> dict[str, Any]:
    # Imported here: the full pipeline loads the embedders (OpenAI SDK, FastEmbed BM25),
    # which --metadata-only does not need.
    from app.ingest.embed import get_dense_embedder, get_sparse_embedder
    from app.ingest.fulltext import default_sources
    from app.ingest.index import ensure_collection
    from app.ingest.pipeline import run_paper

    ensure_collection()
    result = run_paper(
        session,
        arxiv_id,
        clients=default_clients(),
        sources=default_sources(),
        dense=get_dense_embedder(),
        sparse=get_sparse_embedder(),
        force_source=force_source,  # type: ignore[arg-type]  # argparse restricts the choices
    )
    data = asdict(result)
    data["metadata"] = result.metadata.model_dump()
    return data


if __name__ == "__main__":
    raise SystemExit(main())

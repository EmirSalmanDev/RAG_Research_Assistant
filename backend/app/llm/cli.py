"""Command line entry for paper analyses (used by `make analyze-submit` / `analyze-collect`,
which run it in the worker container):

    python -m app.llm.cli submit [--arxiv-id ID ...] [--max-papers N] [--dry-run]
    python -m app.llm.cli collect

`submit --dry-run` selects the papers and prints the token counts and the upper bound of
the cost; it writes nothing and makes no API call. Without it, `submit` uploads the
input file and creates one Batch API job (paid when OpenAI runs it). `collect` checks the
open batches and stores and indexes the results of finished ones (the indexing embeds
the analysis chunks: small embedding requests).
"""

import argparse
import json
import logging
from typing import Any

from app.db.session import session_scope
from app.ingest.index import close_qdrant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.llm.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    submit = sub.add_parser("submit", help="submit one Batch API job of paper analyses")
    submit.add_argument("--arxiv-id", action="append", help="only these papers (repeatable)")
    submit.add_argument("--max-papers", type=int, help="at most this many papers")
    submit.add_argument("--dry-run", action="store_true", help="select and estimate only")
    sub.add_parser("collect", help="store and index the results of finished batches")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        output = _submit(args) if args.command == "submit" else _collect()
    finally:
        close_qdrant()
    print(json.dumps(output, indent=2, default=str))
    return 0


def _client() -> Any:
    from app.ingest.embed import default_openai_client

    return default_openai_client()


def _submit(args: argparse.Namespace) -> dict[str, Any]:
    from app.llm.batch import submit
    from app.sources.arxiv import validate_arxiv_id

    ids = [validate_arxiv_id(i) for i in args.arxiv_id] if args.arxiv_id else None
    with session_scope() as session:
        # The client is only created for a real submission (a dry run needs no key).
        client = None if args.dry_run else _client()
        return submit(
            session, client, arxiv_ids=ids, max_papers=args.max_papers, dry_run=args.dry_run
        )


def _collect() -> dict[str, Any]:
    from app.ingest.embed import get_dense_embedder, get_sparse_embedder
    from app.ingest.index import ensure_collection
    from app.llm.batch import collect, open_batch_count

    with session_scope() as session:
        if open_batch_count(session) == 0:
            return {"open": 0}
        ensure_collection()
        return collect(session, _client(), dense=get_dense_embedder(), sparse=get_sparse_embedder())


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry: `python -m app.ingest.cli one <arxiv_id>` (used by `make ingest-one`)."""

import argparse
import json
import logging
import sys

from app.db.session import session_scope
from app.ingest.paper import IngestError, default_clients, ingest_one


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.ingest.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("one", help="ingest a single paper's metadata")
    one.add_argument("arxiv_id")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        with session_scope() as session:
            result = ingest_one(session, default_clients(), args.arxiv_id)
    except (IngestError, ValueError) as err:
        print(f"ingest-one failed: {err}", file=sys.stderr)
        return 1
    print(json.dumps(result.model_dump(), indent=2))
    return 0  # optional-source failures are in result.errors but do not fail the command


if __name__ == "__main__":
    raise SystemExit(main())

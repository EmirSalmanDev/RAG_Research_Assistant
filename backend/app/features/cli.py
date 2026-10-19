"""Command line for the digest, used by `make digest`. Runs natively against the stores
on localhost (like `make migrate` and `make eval-retrieval`):

    python -m app.features.cli [--dry-run] [--date YYYY-MM-DD] [--rebuild] [--no-embed]

One run does what the nightly chain does for the digest: embed the interests whose text
changed (the only paid call, one embeddings request of a few hundred tokens at most),
score the papers of the window, and build the digest of the date (default: today in
Istanbul). It then prints the digest and which papers the nightly full-text selection
WOULD pick (selection only: nothing is downloaded).

`--dry-run` writes nothing and makes no request: it ranks with the interest vectors that
are already stored. `--no-embed` skips the embedding request. `--rebuild` replaces the
entries of an existing digest that nobody saved or dismissed.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, date, datetime

from app.db.session import session_scope
from app.features import app_settings, digest
from app.ingest.fulltext import select_for_fulltext
from app.ingest.index import close_qdrant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.features.cli")
    parser.add_argument("--dry-run", action="store_true", help="write nothing, make no request")
    parser.add_argument("--date", type=date.fromisoformat, help="digest date (YYYY-MM-DD)")
    parser.add_argument("--rebuild", action="store_true", help="replace the entries still new")
    parser.add_argument("--no-embed", action="store_true", help="do not embed interests")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        return _run(args)
    finally:
        close_qdrant()


def _run(args: argparse.Namespace) -> int:
    now = datetime.now(UTC)
    day = args.date or digest.today_in_zone(now)
    # For a past date the score step and the selection preview look at the window that
    # date could see, like the digest build itself.
    reference = digest.reference_time(day, now)
    vectors = digest.QdrantAbstractVectors()
    with session_scope() as session:
        report: dict[str, object] = {}
        if not args.dry_run and not args.no_embed:
            from app.ingest.embed import get_dense_embedder

            report["interests_embedded"] = digest.ensure_interest_embeddings(
                session, get_dense_embedder()
            )
            session.commit()  # the embedding request is paid: keep its result
        if not args.dry_run:
            report["score"] = digest.score_window(session, vectors, reference)
        result = digest.build_digest(
            session, vectors, digest_date=day, now=now, rebuild=args.rebuild,
            dry_run=args.dry_run,
        )  # fmt: skip
        report["digest"] = result.counts()
        top_n = app_settings.load(session).fulltext_top_n
        selection = list(select_for_fulltext(session, top_n=top_n, now=reference))
        report["fulltext_selection_would_pick"] = {"top_n": top_n, "arxiv_ids": selection}
        print(json.dumps(report, indent=2, default=str))
        for entry in result.entries[:20]:
            print(
                f"{entry.rank:>3}  {entry.score:.4f}  interest={entry.best_interest_id}  "
                f"{entry.arxiv_id}  {entry.title[:70]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

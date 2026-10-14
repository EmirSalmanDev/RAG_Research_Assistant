"""Command line entry, used by `make ingest-one` (which runs it in the worker container):

    python -m app.ingest.cli one <arxiv_id> [--force-source hf|html|pdf] [--metadata-only]
    python -m app.ingest.cli backfill --days N [--confirm] [--max-papers M]
    python -m app.ingest.cli enrich <arxiv_id>

`one` ingests a paper end to end: metadata and enrichment, the abstract chunk, the full
text (Hugging Face .md, then arXiv HTML, then the PDF; or only the forced source),
cleaning, chunking, embedding and indexing. `--metadata-only` keeps the old behaviour
(metadata and enrichment only: no full text, no OpenAI calls).

`enrich` refreshes GitHub, Hugging Face and Semantic Scholar data for a paper we already
hold (no arXiv call, no full text, no OpenAI call; used by `make enrich-one`).

`backfill` fetches the last N days of the configured categories (new papers and new
versions) and indexes their abstracts only. More than 7 days needs --confirm.
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
    back = sub.add_parser("backfill", help="fetch the last N days and index abstracts only")
    back.add_argument("--days", type=int, required=True)
    back.add_argument("--confirm", action="store_true", help="required for more than 7 days")
    back.add_argument("--max-papers", type=int, default=20_000, help="stop paging after this many")
    enrich = sub.add_parser("enrich", help="GitHub, HF and S2 data for a paper we hold")
    enrich.add_argument("arxiv_id")
    args = parser.parse_args(argv)
    if args.command == "enrich":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        return _enrich(args.arxiv_id)
    if args.command == "backfill":
        if args.days < 1 or (args.days > MAX_DAYS_WITHOUT_CONFIRM and not args.confirm):
            parser.error(f"--days must be 1..{MAX_DAYS_WITHOUT_CONFIRM} (more needs --confirm)")
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        try:
            print(json.dumps(_backfill(args.days, args.max_papers), indent=2, default=str))
        finally:
            close_qdrant()
        return 0
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


MAX_DAYS_WITHOUT_CONFIRM = 7
AVG_ABSTRACT_TOKENS = 300  # measured on the first indexed abstracts (~130-350 tokens)


def embedding_requests(papers: int, batch_size: int) -> int:
    """Requests index_abstracts makes for `papers` abstracts: it embeds them in groups of
    EMBED_GROUP (1000), each sent in requests of `batch_size` (100)."""
    from app.ingest.index import EMBED_GROUP

    full, rest = divmod(papers, EMBED_GROUP)
    return full * -(-EMBED_GROUP // batch_size) + -(-rest // batch_size)


def _backfill(days: int, max_papers: int) -> dict[str, Any]:
    from app.config import get_settings
    from app.ingest import daily
    from app.ingest.embed import get_dense_embedder, get_sparse_embedder
    from app.ingest.index import ensure_collection, index_abstracts
    from app.llm.usage import embedding_cost
    from app.sources.arxiv import ArxivClient

    plan = daily.plan_backfill(days)
    settings = get_settings()
    print(
        f"backfill: {days} days [{plan.since:%Y-%m-%d %H:%M} .. {plan.until:%Y-%m-%d %H:%M} UTC], "
        f"categories {', '.join(settings.arxiv_categories)}, at most {max_papers} papers, "
        "abstracts only",
        flush=True,
    )

    def estimate(fetched: dict[str, Any]) -> None:
        todo = int(fetched["fetched"])
        tokens = todo * AVG_ABSTRACT_TOKENS
        requests = embedding_requests(todo, settings.embedding_batch_size)
        print(
            f"fetched {todo} papers ({fetched['new']} new, {fetched['new_version']} new versions)"
            f"; at most {todo} abstracts to embed: ~{tokens} tokens, <= {requests} requests, "
            f"~${embedding_cost(tokens)}",
            flush=True,
        )

    ensure_collection()
    dense, sparse = get_dense_embedder(), get_sparse_embedder()
    with session_scope() as session:
        return daily.backfill(
            session,
            ArxivClient(),
            plan,
            lambda s: index_abstracts(s, dense=dense, sparse=sparse),
            max_papers=max_papers,
            before_index=estimate,
        )


def _enrich(raw_arxiv_id: str) -> int:
    from app.ingest.enrich import NotHeldError, default_enrich_clients, enrich_paper
    from app.sources.arxiv import validate_arxiv_id

    try:
        arxiv_id = validate_arxiv_id(raw_arxiv_id)
        with session_scope() as session:
            result = enrich_paper(session, default_enrich_clients(), arxiv_id)
    except (NotHeldError, ValueError) as err:
        print(f"enrich-one failed: {err}", file=sys.stderr)
        return 1
    print(json.dumps({"arxiv_id": arxiv_id, **asdict(result)}, indent=2, default=str))
    return 0  # a source that failed is listed in "errors" but does not fail the command


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

"""Command line for adding a PDF that is not on arXiv, used by `make ingest-pdf`. It runs
natively against the stores on localhost, through the same service function as the upload
route (`app.ingest.upload.ingest_pdf`):

    python -m app.ingest.upload_cli FILE --title TITLE [--year Y] [--abstract TEXT]
        [--collection NAME] [--estimate]

Without --estimate this makes REAL paid OpenAI embeddings requests (about $0.0002 per 10
pages; ask first) and writes the document to Postgres and Qdrant. `--estimate` parses the PDF
in memory, cleans and chunks it, and prints how many chunks and tokens would be embedded and
the cost bound: no network, no key, nothing stored.
"""

import argparse
import json
import logging
import sys
from decimal import Decimal
from pathlib import Path

from app.config import get_settings
from app.db.session import session_scope
from app.ingest.chunk import abstract_chunk, chunk_paper
from app.ingest.clean import clean_markdown
from app.ingest.embed import OpenAIKeyMissing
from app.ingest.fulltext import pdf_to_markdown
from app.ingest.index import close_qdrant
from app.ingest.tokens import count_tokens
from app.ingest.upload import (
    NoTextLayer,
    UploadBusy,
    UploadDeps,
    UploadError,
    abstract_from_body,
    check_body,
    check_metadata,
    ingest_pdf,
    local_id_for,
    upload_slot,
)
from app.llm.usage import embedding_cost


def estimate_upload(
    pdf: bytes, title: str, year: int | None, abstract: str | None
) -> dict[str, object]:
    """What an upload would embed, from the parsed text. Nothing is stored or sent."""
    settings = get_settings()
    check_body(pdf)
    clean_title, upload_year, clean_abstract = check_metadata(title, year, abstract)
    raw = pdf_to_markdown(pdf)
    if len(raw.strip()) < settings.upload_min_text_chars:
        raise NoTextLayer("the PDF has no text layer (a scan?); OCR is not supported")
    markdown = clean_markdown(raw).markdown
    arxiv_id = local_id_for(pdf)
    body_abstract = clean_abstract or abstract_from_body(markdown, settings.upload_abstract_chars)
    chunks = [
        abstract_chunk(arxiv_id, 1, clean_title, body_abstract),
        *chunk_paper(arxiv_id, 1, clean_title, markdown),
    ]
    tokens = sum(count_tokens(c.embedding_text) for c in chunks)
    # Indexing sends one request per chunk type (abstract, then sections), in batches.
    sections = len(chunks) - 1
    requests = 1 + (-(-sections // settings.embedding_batch_size) if sections else 0)  # ceil
    return {
        "estimate": "upper bound; nothing was sent or stored",
        "id": arxiv_id,
        "year": upload_year,
        "chunks": len(chunks),
        "embedding_tokens": tokens,
        "embedding_requests": f"{requests} (one for the abstract, then the sections in batches)",
        "cost_usd_max": float(embedding_cost(tokens).quantize(Decimal("0.000001"))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.ingest.upload_cli")
    parser.add_argument("file", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--abstract", default=None)
    parser.add_argument("--collection", default=None)
    parser.add_argument(
        "--estimate", action="store_true", help="chunks, tokens and cost bound only: no key"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        pdf = args.file.read_bytes()
    except OSError as err:
        print(f"error: cannot read {args.file}: {err}", file=sys.stderr)
        return 2
    try:
        if args.estimate:
            print(json.dumps(estimate_upload(pdf, args.title, args.year, args.abstract), indent=2))
            return 0
        if get_settings().openai_api_key is None:
            raise OpenAIKeyMissing("OPENAI_API_KEY is not set (put it in .env)")
        with upload_slot(), session_scope() as session:
            result = ingest_pdf(
                session, pdf, title=args.title, year=args.year, abstract=args.abstract,
                collection=args.collection, deps=UploadDeps.default(),
            )  # fmt: skip
        state = "added" if result.created else "already there (nothing changed)"
        print(f"{result.arxiv_id}: {state}; {result.title!r} ({result.year})")
        print(f"chunks {result.chunks}, embedded {result.embedded}")
        spend = result.spend
        print(
            f"embedding requests {spend.embedding_requests}, tokens {spend.tokens_in}, "
            f"cost ${spend.cost_usd:.6f}"
        )
        return 0
    except OpenAIKeyMissing as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    except (UploadError, UploadBusy) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    finally:
        close_qdrant()


if __name__ == "__main__":
    sys.exit(main())

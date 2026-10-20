"""Command line for the literature review and the .bib export, used by `make review`.
Runs natively against Postgres on localhost (like `make digest` and `make compare`):

    python -m app.features.review_cli COLLECTION [--focus TEXT] [--max-papers N]
        [--estimate] [--bib] [--json]

Without --estimate and --bib this makes REAL paid OpenAI chat calls: one per group of eight
papers above eight papers, plus one writing call. `--estimate` makes no network call and needs
no key: it reads Postgres only and prints an UPPER BOUND of the input tokens and of the cost
from the stored text. The bound is not a prediction: the real answer is shorter than the
output cap. `--bib` prints the collection's .bib file (no model, no key).
"""

import argparse
import json
import logging
import sys
from decimal import Decimal

from app.config import get_settings
from app.features import library, review
from app.features.bibtex import build_entries, render_bibtex
from app.features.errors import InvalidRequest, UnknownCollection, UpstreamError
from app.ingest.embed import OpenAIKeyMissing
from app.ingest.tokens import count_tokens
from app.llm.client import MESSAGE_OVERHEAD_TOKENS
from app.llm.usage import chat_cost
from app.rag.prompts import load_prompt

NOTE_FORMAT_TOKENS = 20  # the <note> wrapper and the labels around one theme note
NOTES_PER_GROUP_MAX = 10  # review_map_v1 asks for 4 to 10 notes


def estimate_review(
    collection: str, focus: str | None, max_papers: int | None, deps: review.ReviewDeps
) -> dict[str, object]:
    """An upper bound of the tokens and the cost of a review, from the stored text."""
    settings = get_settings()
    wanted = settings.review_default_papers if max_papers is None else max_papers
    selection, cleaned = review.prepare(deps, collection, focus, wanted)
    sources = selection.sources
    tokens_in = 0
    tokens_out = settings.review_write_max_output_tokens
    map_calls = 0
    if review.uses_map(sources, settings):
        groups = review.groups_of(sources, settings.review_map_group_size)
        map_calls = len(groups)
        for group in groups:
            tokens_in += review.input_tokens(review.map_messages(group, cleaned, settings))
        tokens_out += map_calls * settings.review_map_max_output_tokens
        # The writer reads the notes, which cannot be longer than the map calls' output (the
        # JSON around them is dropped), plus the numbered list of papers.
        notes_bound = map_calls * (
            settings.review_map_max_output_tokens + NOTES_PER_GROUP_MAX * NOTE_FORMAT_TOKENS
        )
        listing = review.format_source_list(sources)
        write_in = (
            count_tokens(load_prompt(settings.review_write_prompt).text)
            + count_tokens(listing)
            + notes_bound
            + 2 * MESSAGE_OVERHEAD_TOKENS
            + count_tokens(cleaned or "")
        )
    else:
        write_in = review.input_tokens(review.write_messages(sources, None, cleaned, settings))
    tokens_in += write_in
    cost = chat_cost(tokens_in, tokens_out)
    return {
        "estimate": "upper bound; no request was made",
        "papers": len(sources),
        "papers_without_analysis": sum(s.evidence == "abstract" for s in sources),
        "papers_omitted": selection.omitted,
        "requests": f"{map_calls + 1} chat (map {map_calls}, write 1), 0 embeddings",
        "chat_input_tokens_max": tokens_in,
        "chat_output_tokens_max": tokens_out,
        "cost_usd_max": float(cost.quantize(Decimal("0.000001"))),
    }


def _print_review(out: review.ReviewOut) -> None:
    print(f"collection: {out.collection}" + (f"   focus: {out.focus}" if out.focus else ""))
    for section in out.sections:
        print(f"\n## {section.heading}")
        for paragraph in section.paragraphs:
            print()
            print(
                " ".join(
                    s.text + "".join(f"[{n}]" for n in s.source_numbers)
                    for s in paragraph.sentences
                )
            )
    print("\nReferences:")
    for r in out.references:
        link = r.abs_url or "uploaded document"
        used = f"cited {r.cited_count}x" if r.cited_count else "not cited"
        print(f"  [{r.n}] {r.bibtex_key}  ({r.evidence}, {used})  {r.title}  -> {link}")
    print(f"\n{out.counts.model_dump_json()}")
    print(
        f"tokens in/out {out.tokens_in}/{out.tokens_out}, cost ${out.cost_usd:.6f}, "
        + ", ".join(f"{k} {v}" for k, v in out.prompt_versions.items())
    )
    print(
        "NOTE: the draft rests on generated analyses and abstracts, not on full text, and a "
        "citation that resolves does not prove the cited text supports the sentence."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.features.review_cli")
    parser.add_argument("collection")
    parser.add_argument("--focus", default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument(
        "--estimate", action="store_true", help="upper bound only: no request, no key"
    )
    parser.add_argument("--bib", action="store_true", help="print the collection's .bib file")
    parser.add_argument("--json", action="store_true", help="print the full response as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        deps = review.ReviewDeps.default()
        if args.bib:
            from app.db.session import session_scope

            with session_scope() as session:
                papers = library.collection_bib_papers(session, args.collection)
            print(render_bibtex(build_entries(papers)), end="")
        elif args.estimate:
            print(
                json.dumps(
                    estimate_review(args.collection, args.focus, args.max_papers, deps), indent=2
                )
            )
        else:
            out = review.run_review(args.collection, args.focus, args.max_papers, deps=deps)
            if args.json:
                print(out.model_dump_json(indent=2))
            else:
                _print_review(out)
        return 0
    except (InvalidRequest, UnknownCollection, library.CollectionNotFound) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OpenAIKeyMissing as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    except UpstreamError as err:
        print(f"error: {err}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())

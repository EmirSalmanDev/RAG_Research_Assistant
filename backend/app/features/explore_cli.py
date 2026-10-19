"""Command line for the comparison and the timeline, used by `make compare` and
`make timeline`. Runs natively against the stores on localhost (like `make digest`):

    python -m app.features.explore_cli compare ID ID [ID ...] [--dimensions a,b]
        [--estimate] [--json]
    python -m app.features.explore_cli timeline "topic" [--date-from D] [--date-to D]
        [--category C] [--max-papers N] [--key-papers N] [--estimate] [--json]

Without --estimate these make REAL paid OpenAI calls (one embeddings request and one chat
request each). `--estimate` makes no network call and needs no key: it reads Postgres only
and prints an UPPER BOUND of the input tokens and of the cost, from the token counts stored
with the chunks. The estimate is a bound, not a prediction: de-duplication, shorter chunks
and a shorter answer all make the real number smaller.
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select

from app.config import Settings, get_settings
from app.db.models import Chunk, Paper
from app.db.session import session_scope
from app.features import compare, timeline
from app.features.deps import ExploreDeps
from app.features.errors import InvalidRequest, UnknownPapers, UpstreamError
from app.ingest.embed import OpenAIKeyMissing
from app.ingest.index import close_qdrant
from app.ingest.tokens import count_tokens
from app.llm.client import MESSAGE_OVERHEAD_TOKENS
from app.llm.usage import chat_cost, embedding_cost
from app.rag.prompts import load_prompt

SOURCE_HEADER = '<source n="99" evidence="section">\nTitle: {title}\narXiv: {id}\nSection: {path}\nEvidence: section\nText:\n\n</source>'  # noqa: E501
CUT_MARK_TOKENS = 5


def _bounds(
    per_cell: list[list[int]], overheads: list[list[int]], settings: Settings, fixed: int
) -> int:
    """Upper bound of the fill call's input tokens for the largest top-k that fits the cap
    (the same 3 -> 2 -> 1 rule the real run uses; see compare.fit_to_cap)."""

    def bound(k: int) -> int:
        return fixed + sum(
            sum(sorted((t + o for t, o in zip(sizes, over, strict=True)), reverse=True)[:k])
            for sizes, over in zip(per_cell, overheads, strict=True)
        )

    top = bound(settings.compare_top_k)
    if top <= settings.compare_max_input_tokens:
        return top
    return min(top, max(settings.compare_max_input_tokens, bound(1)))


def estimate_compare(arxiv_ids: Sequence[str], dimensions: Sequence[str]) -> dict[str, object]:
    settings = get_settings()
    wanted = compare.check_request(arxiv_ids, dimensions)
    with session_scope() as session:
        papers = {
            p.arxiv_id: p
            for p in session.scalars(select(Paper).where(Paper.arxiv_id.in_(list(arxiv_ids))))
        }
        missing = [a for a in arxiv_ids if a not in papers]
        if missing:
            raise UnknownPapers(missing)
        per_cell: list[list[int]] = []
        overheads: list[list[int]] = []
        for arxiv_id in arxiv_ids:
            rows = session.execute(
                select(Chunk.token_count, Chunk.section_path)
                .where(
                    Chunk.arxiv_id == arxiv_id,
                    Chunk.chunk_type.in_(settings.compare_evidence_types),
                )
                .order_by(Chunk.token_count.desc())
                .limit(settings.compare_top_k)
            ).all()
            overhead = [
                count_tokens(
                    SOURCE_HEADER.format(
                        title=papers[arxiv_id].title, id=arxiv_id, path=path or "Abstract"
                    )
                )
                for _, path in rows
            ]
            sizes = [
                min(t, settings.compare_source_max_tokens)
                + (CUT_MARK_TOKENS if t > settings.compare_source_max_tokens else 0)
                for t, _ in rows
            ]
            for _ in wanted:  # every dimension of the paper may retrieve its largest chunks
                per_cell.append(sizes)
                overheads.append(overhead)
    prompt_tokens = count_tokens(load_prompt(settings.compare_prompt).text)
    fixed = prompt_tokens + 2 * MESSAGE_OVERHEAD_TOKENS + 40 + 12 * len(arxiv_ids)
    tokens_in = _bounds(per_cell, overheads, settings, fixed)
    queries = sum(count_tokens(settings.compare_dimension_queries[d]) for d in wanted)
    return _estimate(tokens_in, settings.compare_max_output_tokens, queries, requests=2)


def estimate_timeline(req: timeline.TimelineRequest) -> dict[str, object]:
    settings = get_settings()
    req = timeline.check_request(req)
    with session_scope() as session:
        sizes = session.scalars(
            select(Chunk.token_count)
            .where(Chunk.chunk_type == "abstract")
            .order_by(Chunk.token_count.desc())
            .limit(req.key_papers)
        ).all()
        longest = (
            session.scalar(select(Paper.title).order_by(func.length(Paper.title).desc()).limit(1))
            or ""
        )
    header = (
        count_tokens(SOURCE_HEADER.format(title=longest, id="0000.00000", path="Abstract")) + 12
    )
    per_paper = [
        min(t, settings.timeline_abstract_max_tokens) + CUT_MARK_TOKENS + header for t in sizes
    ]
    fixed = (
        count_tokens(load_prompt(settings.timeline_prompt).text) + 2 * MESSAGE_OVERHEAD_TOKENS + 30
    )
    tokens_in = fixed + sum(per_paper)
    return _estimate(
        tokens_in, settings.timeline_max_output_tokens, count_tokens(req.topic), requests=2
    )


def _estimate(
    tokens_in: int, tokens_out: int, embed_tokens: int, requests: int
) -> dict[str, object]:
    cost = chat_cost(tokens_in, tokens_out) + embedding_cost(embed_tokens)
    return {
        "estimate": "upper bound; no request was made",
        "requests": f"1 embeddings + {requests - 1} chat",
        "embedding_tokens": embed_tokens,
        "chat_input_tokens_max": tokens_in,
        "chat_output_tokens_max": tokens_out,
        "cost_usd_max": float(cost.quantize(Decimal("0.000001"))),
    }


def _print_totals(tokens_in: int, tokens_out: int, cost: float, prompt: str) -> None:
    print(f"tokens in/out {tokens_in}/{tokens_out}, cost ${cost:.6f}, {prompt}")


def _print_compare(out: compare.CompareOut) -> None:
    titles = {r.arxiv_id: r.title for r in out.rows}
    for row in out.rows:
        badge = "full text" if row.has_fulltext else "abstract only"
        print(f"\n== {row.arxiv_id} ({badge}) {titles[row.arxiv_id]}")
        for cell in row.cells:
            refs = "".join(f"[{c.n}]" for c in cell.citations)
            print(f"  {cell.dimension:12} {cell.status:10} {cell.text} {refs}")
    print("\nSources:")
    for s in out.sources:
        print(
            f"  [{s.n}] {s.arxiv_id} {s.chunk_type} {s.section_path or 'Abstract'} -> {s.abs_url}"
        )
    print(f"\n{out.notes.model_dump_json()}")
    _print_totals(out.tokens_in, out.tokens_out, out.cost_usd, out.prompt_version)


def _print_timeline(out: timeline.TimelineOut) -> None:
    print(f"topic: {out.topic}  range: {out.range.first} .. {out.range.last}")
    for p in out.points:
        mark = "KEY" if p.is_key else "   "
        cites = "?" if p.citation_count is None else p.citation_count
        scores = f"rel={p.topic_relevance:.2f} score={p.key_score:.3f} cites={cites}"
        print(f"{mark} {p.published_at.date()} {p.arxiv_id} {scores}  {p.title}")
        if p.line:
            print(f"       ({p.line_source}) {p.line}")
    print(f"\n{out.notes.model_dump_json()}")
    _print_totals(out.tokens_in, out.tokens_out, out.cost_usd, out.prompt_version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.features.explore_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    cmp_ = sub.add_parser("compare", help="compare 2-5 papers (paid unless --estimate)")
    cmp_.add_argument("arxiv_ids", nargs="+")
    cmp_.add_argument("--dimensions", default=",".join(compare.DIMENSIONS))
    tl = sub.add_parser("timeline", help="a topic timeline (paid unless --estimate)")
    tl.add_argument("topic")
    tl.add_argument("--date-from", type=date.fromisoformat)
    tl.add_argument("--date-to", type=date.fromisoformat)
    tl.add_argument("--category", action="append", default=[])
    tl.add_argument("--max-papers", type=int, default=timeline.DEFAULT_MAX_PAPERS)
    tl.add_argument("--key-papers", type=int, default=timeline.DEFAULT_KEY_PAPERS)
    for p in (cmp_, tl):
        p.add_argument(
            "--estimate", action="store_true", help="upper bound only: no request, no key"
        )
        p.add_argument("--json", action="store_true", help="print the full response as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return _run(args)
    except (InvalidRequest, UnknownPapers) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    except OpenAIKeyMissing as err:
        print(f"error: {err}", file=sys.stderr)
        return 3
    except UpstreamError as err:
        print(f"error: {err}", file=sys.stderr)
        return 4
    finally:
        close_qdrant()


def _run(args: argparse.Namespace) -> int:
    if args.command == "compare":
        dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        if args.estimate:
            print(json.dumps(estimate_compare(args.arxiv_ids, dimensions), indent=2))
            return 0
        out = compare.run_compare(args.arxiv_ids, dimensions, deps=ExploreDeps.default())
        print(out.model_dump_json(indent=2) if args.json else "", end="")
        if not args.json:
            _print_compare(out)
        return 0
    request = timeline.TimelineRequest(
        topic=args.topic, date_from=args.date_from, date_to=args.date_to,
        categories=tuple(args.category), max_papers=args.max_papers, key_papers=args.key_papers,
    )  # fmt: skip
    if args.estimate:
        print(json.dumps(estimate_timeline(request), indent=2))
        return 0
    out_t = timeline.run_timeline(request, deps=ExploreDeps.default())
    if args.json:
        print(out_t.model_dump_json(indent=2))
    else:
        _print_timeline(out_t)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Test-set generation and review: the model writes candidate questions from the stored
full-text papers, and the owner accepts or rejects each one.

    python -m eval.make_testset plan                  # papers available vs needed (read-only)
    python -m eval.make_testset generate --estimate   # seeds, requests, cost bound (free)
    python -m eval.make_testset generate              # PAID: one model call per seed
    python -m eval.make_testset review                # accept or reject each candidate
    python -m eval.make_testset status                # counts by type and decision

How a candidate is made (docs/adr/0012):
1. The pool: every paper with full text (it has `section` chunks), at least `--min-papers`
   (30) of them. Its section chunks of at least MIN_SEED_TOKENS tokens are seeds, except
   reference lists and acknowledgements, and except every reference chunk of the golden
   questions, so a generated question is never built on a held-out question's evidence.
2. Seeds are drawn with a fixed random seed, spread over the papers in turns:
   - fact: one section chunk -> a question that chunk alone answers;
   - compare: two section chunks of two different papers -> a comparison;
   - filter: one paper's abstract chunk with its date and categories -> "which paper in
     <period> did X";
   - unanswerable: one section chunk -> a question about that paper it does not answer.
3. One Structured Outputs call per seed (`testset_<type>_v1`, the evaluation model at
   temperature 0) returns the question and a reference answer from the seed text, or
   "not usable". The seed chunks become the reference chunk ids (none for unanswerable).
4. Every candidate is appended to the candidates file at once, so an interrupted run
   keeps what it paid for and a re-run skips seeds already done.

Review shows each pending candidate with its seed text and warns when it resembles a
golden or an already accepted question. Accept, reject (with a reason), skip or quit;
every decision is saved at once. Accepted questions are written to `generated.jsonl` in
the golden format, numbered gen-001, gen-002, ...: the evaluation loads them as the
"generated" split. `golden.jsonl` is only ever read.

Files: the candidates file holds questions, reference answers, ids and decisions, never
chunk text (review reads the text from Postgres), so it can be committed.
"""

import argparse
import hashlib
import random
import re
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, Paper
from app.ids import is_local_id
from app.ingest.tokens import count_tokens
from app.llm.client import MESSAGE_OVERHEAD_TOKENS, ChatClient, ChatMessage
from app.rag.prompts import escape_block, load_prompt
from eval.golden import GENERATED_PATH, GOLDEN_PATH, GoldenItem, load_golden
from eval.llm import eval_cost

CANDIDATES_PATH = Path(__file__).parent / "testset_candidates.jsonl"
CandidateType = Literal["fact", "compare", "filter", "unanswerable"]
TYPES: tuple[CandidateType, ...] = ("fact", "compare", "filter", "unanswerable")
PROMPTS: dict[CandidateType, str] = {t: f"testset_{t}_v1" for t in TYPES}
# 60 candidates for the 40 the plan keeps after review; 8 unanswerable questions in all are
# planned and the golden set has 3, so 10 unanswerable candidates leave room to reject some.
DEFAULT_COUNTS: dict[CandidateType, int] = {
    "fact": 30,
    "compare": 12,
    "filter": 8,
    "unanswerable": 10,
}
MIN_PAPERS = 30  # "40 generated with RAGAS test-set generation from 30 full-text papers"
MIN_SEED_TOKENS = 120  # shorter chunks rarely hold a fact worth a question
SKIP_SECTIONS = re.compile(r"acknowledg|references|bibliography", re.IGNORECASE)
DEFAULT_SEED = 20261021
FEATURE = "eval-testset"  # llm_usage.feature is "eval-testset-<type>"
SIMILAR = 0.6  # word overlap (Jaccard) from which review warns about a near duplicate


# --- the pool of seed chunks ------------------------------------------------------------------


@dataclass(frozen=True)
class SeedChunk:
    chunk_id: uuid.UUID
    arxiv_id: str
    title: str
    chunk_type: str  # section | abstract
    section_path: str | None
    text: str
    token_count: int
    published_at: datetime
    categories: tuple[str, ...]


def load_pool(session: Session) -> list[SeedChunk]:
    """Section and abstract chunks of every full-text paper (read-only), ordered by
    paper and position. Uploaded documents (local- ids) have no arXiv date or category
    and are left out."""
    fulltext = select(Chunk.arxiv_id).where(Chunk.chunk_type == "section").distinct()
    rows = session.execute(
        select(Chunk, Paper.title, Paper.published_at, Paper.categories)
        .join(Paper, Paper.arxiv_id == Chunk.arxiv_id)
        .where(Chunk.arxiv_id.in_(fulltext), Chunk.chunk_type.in_(("section", "abstract")))
        .order_by(Chunk.arxiv_id, Chunk.ord)
    ).all()
    return [
        SeedChunk(
            chunk.id,
            chunk.arxiv_id,
            title,
            chunk.chunk_type,
            chunk.section_path,
            chunk.text,
            chunk.token_count,
            published_at,
            tuple(categories or ()),
        )  # fmt: skip
        for chunk, title, published_at, categories in rows
        if not is_local_id(chunk.arxiv_id)
    ]


def papers_of(pool: Iterable[SeedChunk]) -> list[str]:
    return sorted({c.arxiv_id for c in pool})


def eligible_sections(pool: Iterable[SeedChunk], exclude: set[uuid.UUID]) -> list[SeedChunk]:
    return [
        c
        for c in pool
        if c.chunk_type == "section"
        and c.token_count >= MIN_SEED_TOKENS
        and c.chunk_id not in exclude
        and not SKIP_SECTIONS.search(c.section_path or "")
    ]


def ingest_suggestions(session: Session, n: int) -> list[tuple[str, str, int | None]]:
    """Papers to add full text for, by a rule fixed in advance: the most Hugging Face
    upvotes first (papers the community picked out), then the newest, then the id.
    (arxiv_id, title, upvotes); read-only."""
    fulltext = select(Chunk.arxiv_id).where(Chunk.chunk_type == "section").distinct()
    rows = session.execute(
        select(Paper.arxiv_id, Paper.title, Paper.hf_upvotes)
        .where(Paper.arxiv_id.not_in(fulltext), Paper.arxiv_id.not_like("local-%"))
        .order_by(Paper.hf_upvotes.desc().nulls_last(), Paper.published_at.desc(), Paper.arxiv_id)
        .limit(n)
    ).all()
    return [(a, t, u) for a, t, u in rows]


# --- picking seeds ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    type: CandidateType
    chunks: tuple[SeedChunk, ...]

    @property
    def id(self) -> str:
        """Stable: the same chunks and prompt always give the same candidate id."""
        key = "|".join([self.type, PROMPTS[self.type], *(str(c.chunk_id) for c in self.chunks)])
        return "cand-" + hashlib.sha256(key.encode()).hexdigest()[:10]


def _in_turns(groups: dict[str, list[SeedChunk]], rng: random.Random) -> list[SeedChunk]:
    """One chunk of each paper in turn (papers and chunks shuffled by `rng`), so seeds
    spread over the papers instead of piling up in the longest one."""
    order = sorted(groups)
    rng.shuffle(order)
    queues = {paper: list(groups[paper]) for paper in order}
    for queue in queues.values():
        rng.shuffle(queue)
    out: list[SeedChunk] = []
    while any(queues.values()):
        for paper in order:
            if queues[paper]:
                out.append(queues[paper].pop())
    return out


def pick_seeds(
    pool: Sequence[SeedChunk],
    counts: dict[CandidateType, int],
    *,
    seed: int,
    exclude: set[uuid.UUID],
) -> list[Seed]:
    """Seeds for every type, the same for the same pool, counts and seed. A chunk seeds
    at most one fact or unanswerable question; a type gets fewer seeds than asked when the
    pool runs out (the caller reports it)."""
    rng = random.Random(seed)
    sections = eligible_sections(pool, exclude)
    by_paper: dict[str, list[SeedChunk]] = defaultdict(list)
    for chunk in sections:
        by_paper[chunk.arxiv_id].append(chunk)
    turns = _in_turns(by_paper, rng)
    seeds: list[Seed] = []

    singles = iter(turns)
    single_types: tuple[CandidateType, ...] = ("fact", "unanswerable")
    for kind in single_types:
        for _ in range(counts.get(kind, 0)):
            single = next(singles, None)
            if single is None:
                break
            seeds.append(Seed(kind, (single,)))

    papers = sorted(by_paper)
    pairs = [(a, b) for i, a in enumerate(papers) for b in papers[i + 1 :]]
    rng.shuffle(pairs)
    for k in range(counts.get("compare", 0)):
        if not pairs:
            break
        a, b = pairs[k % len(pairs)]
        seeds.append(Seed("compare", (rng.choice(by_paper[a]), rng.choice(by_paper[b]))))

    abstracts = sorted((c for c in pool if c.chunk_type == "abstract"), key=lambda c: c.arxiv_id)
    rng.shuffle(abstracts)
    for chunk in abstracts[: counts.get("filter", 0)]:
        seeds.append(Seed("filter", (chunk,)))
    return list({s.id: s for s in seeds}.values())  # a repeated compare pair counts once


# --- one model call per seed ------------------------------------------------------------------


class GeneratedQuestion(BaseModel):
    """The JSON shape the model fills."""

    usable: bool
    question: str
    reference_answer: str


Status = Literal["pending", "accepted", "rejected", "unusable"]


class Candidate(BaseModel):
    """One line of the candidates file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: CandidateType
    question: str
    reference_answer: str | None
    reference_chunk_ids: list[uuid.UUID]
    seed_chunk_ids: list[uuid.UUID]
    arxiv_ids: list[str]
    prompt: str
    model: str
    status: Status
    item_id: str | None = None  # gen-NNN once accepted
    reason: str | None = None  # why rejected (optional)
    reviewed_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def _source_block(n: int, chunk: SeedChunk, *, with_dates: bool) -> str:
    """One seed as a data block; everything taken from the paper is escaped (rule 8)."""
    lines = [
        f'<source n="{n}">',
        f"Title: {escape_block(chunk.title)}",
        f"arXiv: {chunk.arxiv_id}",
    ]
    if with_dates:
        lines += [
            f"Published: {chunk.published_at:%Y-%m-%d}",
            f"Categories: {', '.join(chunk.categories)}",
        ]
    lines += [
        f"Section: {escape_block(chunk.section_path or 'Abstract')}",
        f"Text:\n{escape_block(chunk.text)}",
        "</source>",
    ]
    return "\n".join(lines)


def build_messages(seed: Seed) -> list[ChatMessage]:
    blocks = "\n\n".join(
        _source_block(n, c, with_dates=seed.type == "filter")
        for n, c in enumerate(seed.chunks, start=1)
    )
    return [
        {"role": "system", "content": load_prompt(PROMPTS[seed.type]).text},
        {"role": "user", "content": f"<sources>\n{blocks}\n</sources>"},
    ]


def candidate_from(seed: Seed, out: GeneratedQuestion, model: str) -> Candidate:
    usable = out.usable and bool(out.question.strip())
    if seed.type != "unanswerable":
        usable = usable and bool(out.reference_answer.strip())
    return Candidate(
        id=seed.id,
        type=seed.type,
        question=out.question.strip(),
        reference_answer=(out.reference_answer.strip() or None)
        if seed.type != "unanswerable"
        else None,
        reference_chunk_ids=[]
        if seed.type == "unanswerable"
        else [c.chunk_id for c in seed.chunks],
        seed_chunk_ids=[c.chunk_id for c in seed.chunks],
        arxiv_ids=sorted({c.arxiv_id for c in seed.chunks}),
        prompt=PROMPTS[seed.type],
        model=model,
        status="pending" if usable else "unusable",
    )


def load_candidates(path: Path) -> list[Candidate]:
    if not path.is_file():
        return []
    return [
        Candidate.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def save_candidates(path: Path, candidates: Sequence[Candidate]) -> None:
    """Write to a temporary file, then rename: a crash never leaves half a file."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(c.model_dump_json() + "\n" for c in candidates))
    tmp.replace(path)


def generate(
    seeds: Sequence[Seed],
    client: ChatClient,
    path: Path,
    *,
    max_tokens: int,
    progress: Callable[[str], None] = lambda line: None,
) -> list[Candidate]:
    """One call per seed not yet in the file; each candidate is appended at once."""
    done = {c.id for c in load_candidates(path)}
    made: list[Candidate] = []
    todo = [s for s in seeds if s.id not in done]
    for k, seed in enumerate(todo, start=1):
        out, _ = client.parse(
            build_messages(seed), GeneratedQuestion, feature=f"{FEATURE}-{seed.type}",
            max_tokens=max_tokens,
        )  # fmt: skip
        candidate = candidate_from(seed, out, client.model)
        with path.open("a") as f:
            f.write(candidate.model_dump_json() + "\n")
        made.append(candidate)
        progress(f"[{k}/{len(todo)}] {seed.type} {candidate.id}: {candidate.status}")
    return made


@dataclass(frozen=True)
class GenerationEstimate:
    requests: int
    tokens_in: int
    bound_usd: Decimal


def estimate_generation(seeds: Sequence[Seed], *, max_tokens: int) -> GenerationEstimate:
    """Input tokens counted locally (cl100k_base, close to the model's own tokenizer)
    for every message, plus a few per message; output bounded by the cap on every call."""
    tokens_in = sum(
        count_tokens(m["content"]) + MESSAGE_OVERHEAD_TOKENS
        for s in seeds
        for m in build_messages(s)
    )
    return GenerationEstimate(len(seeds), tokens_in, eval_cost(tokens_in, max_tokens * len(seeds)))


# --- review -----------------------------------------------------------------------------------


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(a: str, b: str) -> float:
    """Word overlap (Jaccard): shared words / all words of the two questions."""
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def to_item(candidate: Candidate) -> GoldenItem:
    """An accepted candidate in the golden format (validated by the same model)."""
    assert candidate.item_id is not None
    when = f"{candidate.reviewed_at:%Y-%m-%d}" if candidate.reviewed_at else "unknown date"
    return GoldenItem(
        id=candidate.item_id,
        question=candidate.question,
        reference_answer=candidate.reference_answer,
        reference_chunk_ids=list(candidate.reference_chunk_ids),
        type=candidate.type,
        notes=(
            f"generated by {candidate.prompt} ({candidate.model}) from "
            f"{', '.join(candidate.arxiv_ids)}; candidate {candidate.id}; accepted {when}"
        ),
    )


def write_generated(path: Path, candidates: Sequence[Candidate]) -> None:
    accepted = sorted(
        (c for c in candidates if c.status == "accepted"), key=lambda c: c.item_id or ""
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(to_item(c).model_dump_json(exclude_none=True) + "\n" for c in accepted))
    tmp.replace(path)


ChunkText = tuple[str, str | None, str]  # (title, section_path, text)


def _next_item_id(candidates: Sequence[Candidate]) -> str:
    used = [int(c.item_id[4:]) for c in candidates if c.item_id and c.item_id.startswith("gen-")]
    return f"gen-{max(used, default=0) + 1:03d}"


def review(
    candidates_path: Path,
    generated_path: Path,
    golden_questions: Sequence[str],
    texts: Callable[[Sequence[uuid.UUID]], dict[uuid.UUID, ChunkText]],
    *,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    excerpt_chars: int = 900,
) -> Counter[str]:
    """Walk the pending candidates in file order; return what was decided this session.

    a = accept (gets the next gen-NNN id), r = reject (asks for an optional reason),
    s = skip (stays pending), q = quit. Each decision is saved before the next question.
    A question whose text equals a golden or an accepted one cannot be accepted.
    """
    candidates = load_candidates(candidates_path)
    decided: Counter[str] = Counter()
    pending = [c for c in candidates if c.status == "pending"]
    evidence = texts([cid for c in pending for cid in c.seed_chunk_ids])
    for k, cand in enumerate(pending, start=1):
        accepted = [c.question for c in candidates if c.status == "accepted"]
        papers = ", ".join(cand.arxiv_ids)
        show(f"\n[{k}/{len(pending)}] {cand.id}  type: {cand.type}  papers: {papers}")
        show(f"Q: {cand.question}")
        show(f"Reference answer: {cand.reference_answer or '(none: should be unanswerable)'}")
        for cid in cand.seed_chunk_ids:
            title, section, text = evidence.get(cid, ("(chunk not found)", None, ""))
            show(f"--- seed {cid} | {title} | {section or 'Abstract'}\n{text[:excerpt_chars]}")
        exact = cand.question.strip().lower() in {
            q.strip().lower() for q in [*golden_questions, *accepted]
        }
        for other, label in [(q, "golden") for q in golden_questions] + [
            (q, "accepted") for q in accepted
        ]:
            score = similarity(cand.question, other)
            if score >= SIMILAR:
                show(f"WARNING: {score:.2f} word overlap with a {label} question: {other}")
        while True:
            choice = ask("[a]ccept  [r]eject  [s]kip  [q]uit > ").strip().lower()[:1]
            if choice == "a" and exact:
                show("Not accepted: the same question is already in the set.")
                continue
            if choice in ("a", "r", "s", "q"):
                break
        if choice == "q":
            break
        if choice == "s":
            decided["skipped"] += 1
            continue
        if choice == "a":
            cand.status, cand.item_id = "accepted", _next_item_id(candidates)
        else:
            cand.status = "rejected"
            cand.reason = ask("Reason (optional) > ").strip() or None
        cand.reviewed_at = now()
        decided[cand.status] += 1
        save_candidates(candidates_path, candidates)
        write_generated(generated_path, candidates)
    return decided


def status_table(candidates: Sequence[Candidate]) -> str:
    counts = Counter((c.type, c.status) for c in candidates)
    statuses: tuple[Status, ...] = ("pending", "accepted", "rejected", "unusable")
    lines = ["| type | " + " | ".join(statuses) + " |", "|" + " --- |" * (len(statuses) + 1)]
    for t in TYPES:
        lines.append(f"| {t} | " + " | ".join(str(counts[(t, s)]) for s in statuses) + " |")
    return "\n".join(lines)


# --- command line -----------------------------------------------------------------------------


def _golden_reference_ids(path: Path) -> set[uuid.UUID]:
    return {cid for item in load_golden(path) for cid in item.reference_chunk_ids}


def _chunk_texts(ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, ChunkText]:
    from app.db.session import session_scope

    with session_scope() as session:
        rows = session.execute(
            select(Chunk.id, Paper.title, Chunk.section_path, Chunk.text)
            .join(Paper, Paper.arxiv_id == Chunk.arxiv_id)
            .where(Chunk.id.in_(list(ids)))
        ).all()
    return {cid: (title, section, text) for cid, title, section, text in rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.make_testset", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    plan_p = sub.add_parser("plan", help="full-text papers available and needed (read-only)")
    plan_p.add_argument("--min-papers", type=int, default=MIN_PAPERS)
    gen = sub.add_parser("generate", help="write candidates (PAID unless --estimate)")
    gen.add_argument("--estimate", action="store_true", help="seeds, requests and cost only")
    gen.add_argument("--min-papers", type=int, default=MIN_PAPERS)
    gen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    for t in TYPES:
        gen.add_argument(f"--{t}", type=int, default=DEFAULT_COUNTS[t], help=f"{t} seeds")
    for p in (gen, sub.add_parser("review"), sub.add_parser("status")):
        p.add_argument("--candidates", type=Path, default=CANDIDATES_PATH)
    args = parser.parse_args(argv)

    if args.command == "status":
        print(status_table(load_candidates(args.candidates)))
        return 0
    if args.command == "review":
        golden = [i.question for i in load_golden(GOLDEN_PATH)]
        decided = review(
            args.candidates, GENERATED_PATH, golden, _chunk_texts, ask=input, show=print
        )
        print(f"\nthis session: {dict(decided)}; accepted questions are in {GENERATED_PATH}")
        return 0

    from app.db.session import session_scope

    with session_scope() as session:
        pool = load_pool(session)
        papers = papers_of(pool)
        missing = max(0, args.min_papers - len(papers))
        suggestions = ingest_suggestions(session, missing) if missing else []
    if args.command == "plan" or missing:
        print(f"full-text papers: {len(papers)} ({', '.join(papers) or 'none'}); "
              f"needed: {args.min_papers}")  # fmt: skip
        if missing:
            print(f"missing {missing}. Suggested full-text ingests (most HF upvotes first):")
            for arxiv_id, title, upvotes in suggestions:
                print(f"  {arxiv_id}  upvotes {upvotes}  {title}")
        if args.command == "generate":
            print("Nothing was generated (fewer papers than --min-papers).", file=sys.stderr)
            return 3
        return 0

    counts: dict[CandidateType, int] = {t: getattr(args, t) for t in TYPES}
    seeds = pick_seeds(pool, counts, seed=args.seed, exclude=_golden_reference_ids(GOLDEN_PATH))
    got = Counter(s.type for s in seeds)
    short = {t: f"{got[t]}/{n}" for t, n in counts.items() if got[t] < n}
    settings = get_settings()
    est = estimate_generation(seeds, max_tokens=settings.eval_testset_max_output_tokens)
    print(
        f"{len(seeds)} seeds from {len(papers)} papers ({dict(got)}"
        + (f"; fewer than asked: {short}" if short else "")
        + f"): {est.requests} requests to {settings.eval_model}, {est.tokens_in} input tokens, "
        f"cost bound ${est.bound_usd}."
    )
    if args.estimate:
        print("Nothing was sent (--estimate).")
        return 0
    from eval.llm import eval_chat_client

    made = generate(
        seeds, eval_chat_client(), args.candidates,
        max_tokens=settings.eval_testset_max_output_tokens,
        progress=lambda line: print(line, file=sys.stderr, flush=True),
    )  # fmt: skip
    print(
        f"{len(made)} new candidates in {args.candidates}; next: python -m eval.make_testset review"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

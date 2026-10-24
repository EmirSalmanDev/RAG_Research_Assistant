"""Test-set generation and review: the model writes candidate questions from the stored
full-text papers, and the owner accepts or rejects each one.

    python -m eval.make_testset plan                  # papers available vs needed (read-only)
    python -m eval.make_testset generate --estimate   # seeds, requests, cost bound (free)
    python -m eval.make_testset generate              # PAID: one model call per seed
    python -m eval.make_testset review                # accept or reject each candidate
    python -m eval.make_testset status                # counts by type and decision

How a candidate is made (docs/adr/0012):
1. The pool: every paper with full text (it has `section` chunks) EXCEPT the papers the
   golden questions refer to, at least `--min-papers` (30) of them. A generated question
   is never built on a held-out question's paper (its abstract, its sections, or a
   neighbouring window that repeats a golden passage), so the golden set stays an
   independent check. Section chunks of at least MIN_SEED_TOKENS tokens are seeds, except
   reference lists and acknowledgements.
2. Seeds are drawn with a fixed random seed, spread over the papers in turns:
   - fact: one section chunk -> a question that chunk alone answers;
   - compare: two papers on the same topic (same primary category first, then the most
     similar abstract vectors) and, of their sections, the two most similar ones -> a
     comparison (no random draw; the stored vectors of the local index, no request);
   - filter: one paper's abstract chunk with its date and categories -> "which paper in
     <period> did X";
   - unanswerable: one paper's abstract and the headings of its sections -> a natural
     question about that paper that nothing in it suggests the paper answers.
3. One Structured Outputs call per seed (`PROMPTS`, the evaluation model at temperature 0)
   returns the question and a reference answer from the seed text, or "not usable". The
   seed chunks become the reference chunk ids (none for unanswerable).
4. Every candidate is appended to the candidates file at once, so an interrupted run
   keeps what it paid for and a re-run skips seeds already done.

Review shows each pending candidate with the full text of its seed chunks (and, for an
unanswerable one, the closest passages of the same paper by a free local BM25 search, to
check the paper really does not answer it) and warns when it resembles a golden, an
accepted or an already generated question. Accept, reject (with a reason), skip or quit;
every decision is saved at once. Accepted questions are written to `generated.jsonl` in
the golden format, numbered gen-001, gen-002, ... after the highest number in that file
or in the candidates file; questions in `generated.jsonl` that the candidates file does
not know are kept. The evaluation loads them as the "generated" split. `golden.jsonl` is
only ever read.

Files: the candidates file holds questions, reference answers, ids and decisions, never
chunk text (review reads the text from Postgres), so it can be committed.
"""

import argparse
import hashlib
import math
import operator
import random
import re
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, Paper
from app.ids import is_local_id
from app.ingest.tokens import count_tokens
from app.llm.client import MESSAGE_OVERHEAD_TOKENS, ChatClient, ChatMessage
from app.rag.prompts import EVAL_TAGS, escape_block, load_prompt
from eval.golden import GENERATED_PATH, GOLDEN_PATH, GoldenItem, load_golden
from eval.llm import eval_cost

CANDIDATES_PATH = Path(__file__).parent / "testset_candidates.jsonl"
CandidateType = Literal["fact", "compare", "filter", "unanswerable"]
TYPES: tuple[CandidateType, ...] = ("fact", "compare", "filter", "unanswerable")
# The prompt of each type. v1 of compare and unanswerable gave no usable candidate on the
# first real round (docs/adr/0012, decisions 21-23); their v1 files stay for the record.
PROMPTS: dict[CandidateType, str] = {
    "fact": "testset_fact_v1",
    "compare": "testset_compare_v2",
    "filter": "testset_filter_v1",
    "unanswerable": "testset_unanswerable_v2",
}
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
RELATED_PASSAGES = 3  # same-paper passages shown with an unanswerable candidate


class MakeTestsetError(RuntimeError):
    """The files or the database are not in a state the tool can work from; the message
    says what to fix."""


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
    primary_category: str | None = None


def load_pool(session: Session) -> list[SeedChunk]:
    """Section and abstract chunks of every full-text paper (read-only), ordered by
    paper and position (so a paper's sections are in reading order). Uploaded documents
    (local- ids) have no arXiv date or category and are left out."""
    fulltext = select(Chunk.arxiv_id).where(Chunk.chunk_type == "section").distinct()
    rows = session.execute(
        select(Chunk, Paper.title, Paper.published_at, Paper.categories, Paper.primary_category)
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
            primary,
        )  # fmt: skip
        for chunk, title, published_at, categories, primary in rows
        if not is_local_id(chunk.arxiv_id)
    ]


def golden_papers(session: Session, golden: Path = GOLDEN_PATH) -> set[str]:
    """The papers the golden questions refer to, through their reference chunks
    (read-only). A reference chunk missing from the database is an error: the golden set
    and the stored chunks disagree (a new paper version or re-chunking; eval/golden.py)."""
    refs = {cid for item in load_golden(golden) for cid in item.reference_chunk_ids}
    if not refs:
        return set()
    rows = session.execute(select(Chunk.id, Chunk.arxiv_id).where(Chunk.id.in_(refs))).all()
    found = {cid: arxiv_id for cid, arxiv_id in rows}
    missing = sorted(str(cid) for cid in refs - set(found))
    if missing:
        raise MakeTestsetError(f"golden reference chunks not in the database: {missing}")
    return set(found.values())


def seedable(pool: Iterable[SeedChunk], exclude_papers: set[str]) -> list[SeedChunk]:
    """The pool without the excluded (golden) papers: no chunk of theirs is ever a seed."""
    return [c for c in pool if c.arxiv_id not in exclude_papers]


def papers_of(pool: Iterable[SeedChunk]) -> list[str]:
    return sorted({c.arxiv_id for c in pool})


def eligible_sections(pool: Iterable[SeedChunk], exclude_papers: set[str]) -> list[SeedChunk]:
    return [
        c
        for c in seedable(pool, exclude_papers)
        if c.chunk_type == "section"
        and c.token_count >= MIN_SEED_TOKENS
        and not SKIP_SECTIONS.search(c.section_path or "")
    ]


def ingest_suggestions(
    session: Session, n: int, exclude_papers: set[str]
) -> list[tuple[str, str, int | None]]:
    """Papers to add full text for, by a rule fixed in advance: the most Hugging Face
    upvotes first (papers the community picked out), then the newest, then the id. A
    paper a golden question refers to is never suggested: it could not be a seed.
    (arxiv_id, title, upvotes); read-only."""
    fulltext = select(Chunk.arxiv_id).where(Chunk.chunk_type == "section").distinct()
    rows = session.execute(
        select(Paper.arxiv_id, Paper.title, Paper.hf_upvotes)
        .where(
            Paper.arxiv_id.not_in(fulltext),
            Paper.arxiv_id.not_like("local-%"),
            Paper.arxiv_id.not_in(sorted(exclude_papers)),
        )
        .order_by(Paper.hf_upvotes.desc().nulls_last(), Paper.published_at.desc(), Paper.arxiv_id)
        .limit(n)
    ).all()
    return [(a, t, u) for a, t, u in rows]


# --- picking seeds ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    type: CandidateType
    chunks: tuple[SeedChunk, ...]
    # unanswerable: the headings of the paper's sections, shown with its abstract (they
    # follow from the paper, so they are not part of the id)
    outline: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Stable: the same chunks and prompt always give the same candidate id."""
        key = "|".join([self.type, PROMPTS[self.type], *(str(c.chunk_id) for c in self.chunks)])
        return "cand-" + hashlib.sha256(key.encode()).hexdigest()[:10]


Vectors = Mapping[uuid.UUID, Sequence[float]]  # a chunk's dense vector from the local index


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(map(operator.mul, a, b)))


def _vector(vectors: Vectors, chunk: SeedChunk) -> Sequence[float]:
    if chunk.chunk_id not in vectors:
        raise MakeTestsetError(
            f"no vector for chunk {chunk.chunk_id} of {chunk.arxiv_id} in the local index: "
            "compare seeds need the stored vectors of every abstract and eligible section"
        )
    return vectors[chunk.chunk_id]


def topic_pairs(
    abstracts: Mapping[str, SeedChunk], vectors: Vectors, n: int
) -> list[tuple[str, str]]:
    """Up to `n` pairs of papers on the same topic, each paper in one pair at most.

    Every pair of papers is ranked: the same primary category first, then the cosine
    similarity of their abstract vectors (highest first), then the two ids. Pairs are taken
    in that order unless a paper is already in a pair taken before. Nothing is random."""
    unit = {p: _unit(_vector(vectors, c)) for p, c in abstracts.items()}
    papers = sorted(abstracts)

    def rank(pair: tuple[str, str]) -> tuple[bool, float, tuple[str, str]]:
        a, b = (abstracts[p].primary_category for p in pair)
        return (a is None or a != b, -_dot(unit[pair[0]], unit[pair[1]]), pair)

    ranked = sorted(((a, b) for i, a in enumerate(papers) for b in papers[i + 1 :]), key=rank)
    taken: list[tuple[str, str]] = []
    used: set[str] = set()
    for a, b in ranked:
        if len(taken) == n:
            break
        if a not in used and b not in used:
            taken.append((a, b))
            used |= {a, b}
    return taken


def closest_excerpts(
    a: Sequence[SeedChunk], b: Sequence[SeedChunk], vectors: Vectors
) -> tuple[SeedChunk, SeedChunk] | None:
    """The section of `a` and the section of `b` whose vectors are the most similar (ties:
    the smaller chunk ids), so that both excerpts are likely to be about the same thing."""
    best: tuple[tuple[float, str, str], SeedChunk, SeedChunk] | None = None
    unit_b = [(cb, _unit(_vector(vectors, cb))) for cb in b]
    for ca in a:
        ua = _unit(_vector(vectors, ca))
        for cb, ub in unit_b:
            key = (-_dot(ua, ub), str(ca.chunk_id), str(cb.chunk_id))
            if best is None or key < best[0]:
                best = (key, ca, cb)
    return None if best is None else (best[1], best[2])


def outline_of(pool: Iterable[SeedChunk], arxiv_id: str) -> tuple[str, ...]:
    """The distinct section headings of a paper in reading order (pool order), without
    the reference list and acknowledgements."""
    seen: dict[str, None] = {}
    for c in pool:
        path = c.section_path
        if c.arxiv_id == arxiv_id and c.chunk_type == "section" and path:
            if not SKIP_SECTIONS.search(path):
                seen.setdefault(path, None)
    return tuple(seen)


def _first_rule_compare_draws(
    rng: random.Random, by_paper: Mapping[str, Sequence[SeedChunk]], n: int
) -> None:
    """The random draws the first compare rule (testset_compare_v1) made at this point. They
    are still made, and not used, so that the filter seeds drawn after them stay exactly the
    ones already generated with that rule."""
    papers = sorted(by_paper)
    pairs = [(a, b) for i, a in enumerate(papers) for b in papers[i + 1 :]]
    rng.shuffle(pairs)
    for k in range(n):
        if not pairs:
            break
        a, b = pairs[k % len(pairs)]
        rng.choice(by_paper[a])
        rng.choice(by_paper[b])


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
    exclude_papers: set[str],
    vectors: Vectors | None = None,
) -> list[Seed]:
    """Seeds for every type, the same for the same pool, counts, seed and vectors. No chunk
    of an excluded paper is used, section or abstract. A chunk seeds at most one fact
    question, and a fact seed is never a compare excerpt; a type gets fewer seeds than
    asked when the pool runs out (the caller reports it).

    - fact: the papers' eligible sections in turns (random order from `seed`);
    - unanswerable: the papers the turns give next, each with its abstract and outline;
    - compare: `topic_pairs`, each with its `closest_excerpts` (needs `vectors`);
    - filter: abstracts in a random order from `seed`."""
    rng = random.Random(seed)
    sections = eligible_sections(pool, exclude_papers)
    by_paper: dict[str, list[SeedChunk]] = defaultdict(list)
    for chunk in sections:
        by_paper[chunk.arxiv_id].append(chunk)
    abstract_of = {
        c.arxiv_id: c for c in seedable(pool, exclude_papers) if c.chunk_type == "abstract"
    }
    turns = _in_turns(by_paper, rng)
    seeds: list[Seed] = []

    singles = iter(turns)
    for _ in range(counts.get("fact", 0)):
        single = next(singles, None)
        if single is None:
            break
        seeds.append(Seed("fact", (single,)))
    # The next papers in the turns get the unanswerable questions (the papers the first
    # rule chose too); the question is written from the abstract and the outline.
    for _ in range(counts.get("unanswerable", 0)):
        single = next(singles, None)
        if single is None:
            break
        abstract = abstract_of.get(single.arxiv_id)
        if abstract is not None:
            outline = outline_of(pool, single.arxiv_id)
            seeds.append(Seed("unanswerable", (abstract,), outline))

    _first_rule_compare_draws(rng, by_paper, counts.get("compare", 0))
    facts = {s.chunks[0].chunk_id for s in seeds if s.type == "fact"}
    pairable = {p: c for p, c in abstract_of.items() if p in by_paper}
    if counts.get("compare", 0) and len(pairable) > 1:
        if vectors is None:
            raise MakeTestsetError("compare seeds need the chunk vectors of the local index")
        for a, b in topic_pairs(pairable, vectors, counts["compare"]):
            free_a = [c for c in by_paper[a] if c.chunk_id not in facts]
            free_b = [c for c in by_paper[b] if c.chunk_id not in facts]
            excerpts = closest_excerpts(free_a, free_b, vectors)
            if excerpts is not None:
                seeds.append(Seed("compare", excerpts))

    abstracts = sorted(
        (c for c in seedable(pool, exclude_papers) if c.chunk_type == "abstract"),
        key=lambda c: c.arxiv_id,
    )
    rng.shuffle(abstracts)
    for chunk in abstracts[: counts.get("filter", 0)]:
        seeds.append(Seed("filter", (chunk,)))
    return list({s.id: s for s in seeds}.values())


def pending_seeds(seeds: Sequence[Seed], candidates_path: Path) -> list[Seed]:
    """The seeds the candidates file does not hold yet: what `generate` will send."""
    done = {c.id for c in load_candidates(candidates_path)}
    return [s for s in seeds if s.id not in done]


# --- one model call per seed ------------------------------------------------------------------


class GeneratedQuestion(BaseModel):
    """The JSON shape the v1 prompts ask for: the verdict first."""

    usable: bool
    question: str
    reference_answer: str


class DraftedQuestion(BaseModel):
    """The JSON shape the v2 prompts ask for: the question and the answer first, the verdict
    last, so the model writes before it judges (docs/adr/0012, decision 23)."""

    question: str
    reference_answer: str
    usable: bool


Output = GeneratedQuestion | DraftedQuestion
OUTPUTS: dict[str, type[GeneratedQuestion] | type[DraftedQuestion]] = {
    "testset_fact_v1": GeneratedQuestion,
    "testset_filter_v1": GeneratedQuestion,
    "testset_compare_v2": DraftedQuestion,
    "testset_unanswerable_v2": DraftedQuestion,
}


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


def _escape(text: str) -> str:
    """Paper text for a test-set prompt, with every block tag of the evaluation escaped."""
    return escape_block(text, EVAL_TAGS)


def _source_block(
    n: int, chunk: SeedChunk, *, with_dates: bool, outline: Sequence[str] = ()
) -> str:
    """One seed as a data block; everything taken from the paper is escaped (rule 8)."""
    lines = [f'<source n="{n}">', f"Title: {_escape(chunk.title)}", f"arXiv: {chunk.arxiv_id}"]
    if with_dates:
        lines += [
            f"Published: {chunk.published_at:%Y-%m-%d}",
            f"Categories: {', '.join(chunk.categories)}",
        ]
    lines += [
        f"Section: {_escape(chunk.section_path or 'Abstract')}",
        f"Text:\n{_escape(chunk.text)}",
    ]
    if outline:
        lines += ["Sections of the paper:", *(f"- {_escape(h)}" for h in outline)]
    lines.append("</source>")
    return "\n".join(lines)


def build_messages(seed: Seed) -> list[ChatMessage]:
    blocks = "\n\n".join(
        _source_block(n, c, with_dates=seed.type == "filter", outline=seed.outline)
        for n, c in enumerate(seed.chunks, start=1)
    )
    return [
        {"role": "system", "content": load_prompt(PROMPTS[seed.type]).text},
        {"role": "user", "content": f"<sources>\n{blocks}\n</sources>"},
    ]


def candidate_from(seed: Seed, out: Output, model: str) -> Candidate:
    unanswerable = seed.type == "unanswerable"
    usable = out.usable and bool(out.question.strip())
    if not unanswerable:
        usable = usable and bool(out.reference_answer.strip())
    return Candidate(
        id=seed.id,
        type=seed.type,
        question=out.question.strip(),
        reference_answer=None if unanswerable else (out.reference_answer.strip() or None),
        reference_chunk_ids=[] if unanswerable else [c.chunk_id for c in seed.chunks],
        seed_chunk_ids=[c.chunk_id for c in seed.chunks],
        arxiv_ids=sorted({c.arxiv_id for c in seed.chunks}),
        prompt=PROMPTS[seed.type],
        model=model,
        status="pending" if usable else "unusable",
    )


def load_candidates(path: Path) -> list[Candidate]:
    if not path.is_file():
        return []
    out: list[Candidate] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(Candidate.model_validate_json(line))
        except ValueError as err:  # pydantic's ValidationError is a ValueError
            raise MakeTestsetError(
                f"{path.name} line {number} is not a valid candidate ({type(err).__name__}). "
                "If it is the last line, an interrupted run left it half-written: delete that "
                "line and run generate again (that one seed is made again, one call)."
            ) from err
    return out


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
    made: list[Candidate] = []
    todo = pending_seeds(seeds, path)
    for k, seed in enumerate(todo, start=1):
        out: Output
        out, _ = client.parse(
            build_messages(seed), OUTPUTS[PROMPTS[seed.type]], feature=f"{FEATURE}-{seed.type}",
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


def load_generated(path: Path) -> list[GoldenItem]:
    """The accepted questions already on disk (none if the file is missing or empty)."""
    if not path.is_file() or not path.read_text().strip():
        return []
    return load_golden(path)


def check_generated(existing: Sequence[GoldenItem], candidates: Sequence[Candidate]) -> None:
    """Refuse to go on when generated.jsonl and the candidates file disagree about an
    accepted question (an id with another text, for example after a hand edit), instead
    of silently overwriting either of them."""
    on_disk = {i.id: i for i in existing}
    for c in candidates:
        if c.status == "accepted" and c.item_id in on_disk:
            if on_disk[c.item_id].question != c.question:
                raise MakeTestsetError(
                    f"{c.item_id} in generated.jsonl is not the question of {c.id} in the "
                    "candidates file; make the two agree before reviewing on"
                )


def write_generated(
    path: Path, candidates: Sequence[Candidate], existing: Sequence[GoldenItem]
) -> None:
    """generated.jsonl = the questions already there that the candidates file does not
    hold, plus the candidates file's accepted ones, ordered by id. Written to a temporary
    file first, then renamed."""
    items = {i.id: i for i in existing}
    items |= {c.item_id: to_item(c) for c in candidates if c.status == "accepted" and c.item_id}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(items[k].model_dump_json(exclude_none=True) + "\n" for k in sorted(items))
    )
    tmp.replace(path)


ChunkText = tuple[str, str | None, str]  # (title, section_path, text)


def _next_item_id(candidates: Sequence[Candidate], existing: Sequence[GoldenItem]) -> str:
    """One after the highest gen-NNN in either file, so an id is never given twice."""
    ids = [c.item_id for c in candidates if c.item_id] + [i.id for i in existing]
    used = [int(i[4:]) for i in ids if re.fullmatch(r"gen-\d+", i)]
    return f"gen-{max(used, default=0) + 1:03d}"


def review(
    candidates_path: Path,
    generated_path: Path,
    golden_questions: Sequence[str],
    texts: Callable[[Sequence[uuid.UUID]], dict[uuid.UUID, ChunkText]],
    *,
    ask: Callable[[str], str],
    show: Callable[[str], None],
    related: Callable[[Candidate], list[ChunkText]] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Counter[str]:
    """Walk the pending candidates in file order; return what was decided this session.

    a = accept (gets the next gen-NNN id), r = reject (asks for an optional reason),
    s = skip (stays pending), q = quit. Each decision is saved before the next question.
    A question whose text equals a golden, a generated or an accepted one cannot be
    accepted. `related` (for unanswerable candidates) gives the closest passages of the
    same paper, so the reviewer can see whether the paper answers it after all.
    """
    candidates = load_candidates(candidates_path)
    existing = load_generated(generated_path)
    check_generated(existing, candidates)
    known = {c.item_id for c in candidates if c.item_id}
    elsewhere = [i.question for i in existing if i.id not in known]  # e.g. another file's
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
            show(f"--- seed {cid} | {title} | {section or 'Abstract'}\n{text}")
        if cand.type == "unanswerable" and related is not None:
            show("--- closest passages of the same paper (does one answer it after all?)")
            for title, section, text in related(cand):
                show(f"--- {title} | {section or 'Abstract'}\n{text}")
        others = [(q, "golden") for q in golden_questions] + [
            (q, "generated") for q in elsewhere
        ] + [(q, "accepted") for q in accepted]  # fmt: skip
        exact = cand.question.strip().lower() in {q.strip().lower() for q, _ in others}
        for other, label in others:
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
            cand.status, cand.item_id = "accepted", _next_item_id(candidates, existing)
        else:
            cand.status = "rejected"
            cand.reason = ask("Reason (optional) > ").strip() or None
        cand.reviewed_at = now()
        decided[cand.status] += 1
        save_candidates(candidates_path, candidates)
        write_generated(generated_path, candidates, existing)
    return decided


def status_table(candidates: Sequence[Candidate]) -> str:
    counts = Counter((c.type, c.status) for c in candidates)
    statuses: tuple[Status, ...] = ("pending", "accepted", "rejected", "unusable")
    lines = ["| type | " + " | ".join(statuses) + " |", "|" + " --- |" * (len(statuses) + 1)]
    for t in TYPES:
        lines.append(f"| {t} | " + " | ".join(str(counts[(t, s)]) for s in statuses) + " |")
    return "\n".join(lines)


# --- command line -----------------------------------------------------------------------------


def _chunk_texts(ids: Sequence[uuid.UUID]) -> dict[uuid.UUID, ChunkText]:
    from app.db.session import session_scope

    with session_scope() as session:
        rows = session.execute(
            select(Chunk.id, Paper.title, Chunk.section_path, Chunk.text)
            .join(Paper, Paper.arxiv_id == Chunk.arxiv_id)
            .where(Chunk.id.in_(list(ids)))
        ).all()
    return {cid: (title, section, text) for cid, title, section, text in rows}


def pool_vectors(
    chunks: Sequence[SeedChunk], *, client: Any = None, collection: str | None = None
) -> dict[uuid.UUID, list[float]]:
    """The stored dense vectors of `chunks`, read from the local index (Qdrant only, no
    embedding request): what compare seeds are paired by. A chunk the index lacks is left
    out here and reported when a seed needs it."""
    from app.ingest.index import DENSE_VECTOR, get_qdrant

    qdrant = client if client is not None else get_qdrant()
    name = collection or get_settings().qdrant_collection
    ids = [str(c.chunk_id) for c in chunks]
    out: dict[uuid.UUID, list[float]] = {}
    for start in range(0, len(ids), 256):
        points = qdrant.retrieve(
            name, ids=ids[start : start + 256], with_vectors=[DENSE_VECTOR], with_payload=False
        )
        for point in points:
            vector: Any = (
                point.vector.get(DENSE_VECTOR) if isinstance(point.vector, dict) else point.vector
            )
            if isinstance(vector, list):  # a named dense vector is a list of floats
                out[uuid.UUID(str(point.id))] = [float(x) for x in vector]
    return out


def _same_paper_passages(candidate: Candidate) -> list[ChunkText]:
    """The candidate's question as a BM25 search inside its own paper: local and free
    (the sparse encoder and Qdrant on this machine; no embedding request)."""
    from app.ingest.embed import get_sparse_embedder
    from app.rag.search import SearchFilters, search

    s = get_settings()
    hits = search(
        candidate.question, SearchFilters(arxiv_ids=tuple(candidate.arxiv_ids)), mode="bm25",
        limit=RELATED_PASSAGES, prefetch_limit=s.search_prefetch_limit, rrf_k=s.search_rrf_k,
        sparse=get_sparse_embedder(),
    )  # fmt: skip
    return [(h.title, h.section_path, h.text) for h in hits]


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

    try:
        return _run(args)
    except MakeTestsetError as err:
        print(f"make_testset: {err}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.command == "status":
        print(status_table(load_candidates(args.candidates)))
        return 0
    if args.command == "review":
        from app.ingest.index import close_qdrant

        golden = [i.question for i in load_golden(GOLDEN_PATH)]
        try:
            decided = review(
                args.candidates, GENERATED_PATH, golden, _chunk_texts, ask=input, show=print,
                related=_same_paper_passages,
            )  # fmt: skip
        finally:
            close_qdrant()
        print(f"\nthis session: {dict(decided)}; accepted questions are in {GENERATED_PATH}")
        return 0

    from app.db.session import session_scope

    with session_scope() as session:
        excluded = golden_papers(session, GOLDEN_PATH)
        pool = seedable(load_pool(session), excluded)
        papers = papers_of(pool)
        missing = max(0, args.min_papers - len(papers))
        suggestions = ingest_suggestions(session, missing, excluded) if missing else []
    if args.command == "plan" or missing:
        print(
            f"full-text papers that can seed questions: {len(papers)} "
            f"({', '.join(papers) or 'none'}); needed: {args.min_papers}. Left out because "
            f"golden questions refer to them: {', '.join(sorted(excluded)) or 'none'}."
        )
        if missing:
            print(f"missing {missing}. Suggested full-text ingests (most HF upvotes first):")
            for arxiv_id, title, upvotes in suggestions:
                print(f"  {arxiv_id}  upvotes {upvotes}  {title}")
        if args.command == "generate":
            print("Nothing was generated (fewer papers than --min-papers).", file=sys.stderr)
            return 3
        return 0

    counts: dict[CandidateType, int] = {t: getattr(args, t) for t in TYPES}
    vectors: dict[uuid.UUID, list[float]] | None = None
    if counts["compare"]:
        from app.ingest.index import close_qdrant

        try:  # only abstracts and eligible sections can be paired
            wanted = [c for c in pool if c.chunk_type == "abstract"]
            vectors = pool_vectors(wanted + eligible_sections(pool, excluded))
        finally:
            close_qdrant()
    seeds = pick_seeds(pool, counts, seed=args.seed, exclude_papers=excluded, vectors=vectors)
    got = Counter(s.type for s in seeds)
    short = {t: f"{got[t]}/{n}" for t, n in counts.items() if got[t] < n}
    settings = get_settings()
    todo = pending_seeds(seeds, args.candidates)
    est = estimate_generation(todo, max_tokens=settings.eval_testset_max_output_tokens)
    print(
        f"{len(seeds)} seeds from {len(papers)} papers ({dict(got)}"
        + (f"; fewer than asked: {short}" if short else "")
        + f"); {len(seeds) - len(todo)} already in {args.candidates.name}, to send: "
        f"{dict(Counter(s.type for s in todo))}: {est.requests} requests to "
        f"{settings.eval_model}, {est.tokens_in} input tokens, cost bound ${est.bound_usd}."
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

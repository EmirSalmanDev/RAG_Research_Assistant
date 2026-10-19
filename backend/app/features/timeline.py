"""Topic timeline: papers on a topic by date, key papers marked, one line on what changed.

    POST /timeline  topic (+ optional dates, categories)  ->  dots on a time axis

How it works (docs/adr/0010):
1. The topic is embedded (one request) and searched once, hybrid, over the whole index
   with the filters applied inside Qdrant. Only `abstract` chunks count, so each paper
   appears once with its abstract. `search()` has no chunk_type filter, so the hits are
   filtered here; a paper whose abstract is not among the hits is left out.
2. Papers are ordered by their best fused position (ties broken by arXiv id, so the
   order is the same on every run) and cut to `max_papers`.
3. Key papers: `key_score = 0.7 * topic_relevance + 0.3 * citation_norm` (weights in
   config). Both parts are in [0, 1]; see `score_papers`. "Relevance" means relevance to
   the TOPIC, not the interest-based score of the digest.
4. ONE structured call writes the "what changed" line of every key paper from its own
   abstract. Code checks each line; a missing, empty, too long or off-list line is
   replaced by the abstract's first sentence (no second paid call).

Nothing is stored. Every request writes its own llm_usage rows.
"""

import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from typing import Literal

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.features.deps import ExploreDeps, end_span, upstream
from app.features.digest import first_sentence
from app.features.errors import InvalidRequest, UpstreamError
from app.features.paper_facts import PaperFacts
from app.features.spend import EmbeddingMeter, Spend
from app.features.text import cut_to_tokens
from app.llm.client import ChatMessage, ChatResponseError
from app.rag.citations import ARXIV_ABS_URL, format_sources, number_sources
from app.rag.prompts import load_prompt
from app.rag.retrieve import RetrievalConfig
from app.rag.search import SearchFilters, SearchHit

TOPIC_MIN_CHARS = 3  # the API contract
TOPIC_MAX_CHARS = 200
DEFAULT_MAX_PAPERS, MAX_PAPERS_CAP = 30, 50
DEFAULT_KEY_PAPERS, KEY_PAPERS_CAP = 8, 15

EMBED_FEATURE = "embed-timeline"
LINES_FEATURE = "timeline-lines"

# Other arXiv ids in a line (new-style 1706.03762 and old-style cs/0112017).
_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b")


# --- the model's answer ---------------------------------------------------------------------


class LineFill(BaseModel):
    arxiv_id: str
    line: str


class TimelineLines(BaseModel):
    lines: list[LineFill]


# --- the API response ---------------------------------------------------------------------------


class TimelinePoint(BaseModel):
    arxiv_id: str
    title: str
    published_at: datetime
    citation_count: int | None  # None: Semantic Scholar has not answered for this paper
    topic_relevance: float  # 1.0 for the best match down to 0.0 for the last
    key_score: float
    is_key: bool
    line: str | None  # key papers only: what this paper changed, from its abstract
    line_source: Literal["model", "abstract_fallback"] | None
    chunk_id: uuid.UUID  # the abstract chunk the line comes from
    abs_url: str


class DateRange(BaseModel):
    first: datetime | None
    last: datetime | None


class TimelineNotes(BaseModel):
    papers: int
    key_papers: int
    known_citation_counts: int  # papers with a citation count (Semantic Scholar is thin)
    narrow_range: bool  # the papers span fewer than `timeline_narrow_days` days
    range_days: int
    fallback_lines: int  # key papers whose line is the abstract's first sentence
    lines_call_failed: bool


class TimelineOut(BaseModel):
    topic: str
    range: DateRange
    points: list[TimelinePoint]  # oldest first
    notes: TimelineNotes
    prompt_version: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


# --- validation (before any paid call) --------------------------------------------------------


@dataclass(frozen=True)
class TimelineRequest:
    topic: str
    date_from: date | None = None
    date_to: date | None = None
    categories: tuple[str, ...] = ()
    max_papers: int = DEFAULT_MAX_PAPERS
    key_papers: int = DEFAULT_KEY_PAPERS


def check_request(req: TimelineRequest) -> TimelineRequest:
    """The request with a trimmed topic, or InvalidRequest."""
    topic = " ".join(req.topic.split())
    if not TOPIC_MIN_CHARS <= len(topic) <= TOPIC_MAX_CHARS:
        raise InvalidRequest(f"the topic needs {TOPIC_MIN_CHARS} to {TOPIC_MAX_CHARS} characters")
    if req.date_from and req.date_to and req.date_from > req.date_to:
        raise InvalidRequest("date_from is after date_to")
    if not 1 <= req.max_papers <= MAX_PAPERS_CAP:
        raise InvalidRequest(f"max_papers must be 1 to {MAX_PAPERS_CAP}")
    if not 1 <= req.key_papers <= KEY_PAPERS_CAP:
        raise InvalidRequest(f"key_papers must be 1 to {KEY_PAPERS_CAP}")
    return replace(req, topic=topic)


def search_filters(req: TimelineRequest) -> SearchFilters:
    """Whole days in UTC, both ends inclusive (like GET /search)."""
    return SearchFilters(
        categories=req.categories,
        published_from=datetime.combine(req.date_from, time.min, UTC) if req.date_from else None,
        published_to=datetime.combine(req.date_to, time.max, UTC) if req.date_to else None,
    )


# --- candidates and the key-paper ranking (pure) --------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One paper found by the topic search. A list of candidates is in fused order."""

    facts: PaperFacts
    chunk_id: uuid.UUID
    abstract: str  # the text of its abstract chunk


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    topic_relevance: float
    key_score: float


def group_by_paper(
    hits: Sequence[SearchHit], facts: dict[str, PaperFacts], max_papers: int
) -> list[Candidate]:
    """One candidate per paper from abstract hits, best fused position first.

    Hits are put in a fixed order (score, then arXiv id) before grouping: equal RRF
    scores come back from Qdrant in no guaranteed order, and the ranking must not
    change between runs. A hit whose paper is not in Postgres is dropped."""
    ordered = sorted(
        (h for h in hits if h.chunk_type == "abstract" and h.arxiv_id in facts),
        key=lambda h: (-h.score, h.arxiv_id),
    )
    seen: dict[str, Candidate] = {}
    for hit in ordered:
        if hit.arxiv_id not in seen:
            seen[hit.arxiv_id] = Candidate(facts[hit.arxiv_id], hit.chunk_id, hit.text)
    return list(seen.values())[:max_papers]


def score_papers(
    candidates: Sequence[Candidate], *, relevance_weight: float, citation_weight: float
) -> list[Scored]:
    """key_score for every candidate (same order as the input, which is fused order).

    topic_relevance = 1 - (rank - 1) / (n - 1): 1.0 for rank 1, 0.0 for the last, and 1.0
        when there is a single paper.
    citation_norm = ln(1 + citations) / ln(1 + the largest count in the set); an unknown
        count counts as 0, and everything is 0 when the largest count is 0.
    key_score = relevance_weight * topic_relevance + citation_weight * citation_norm.
    """
    n = len(candidates)
    counts = [c.facts.citation_count or 0 for c in candidates]
    top = math.log1p(max(counts, default=0))
    scored = []
    for rank, (candidate, count) in enumerate(zip(candidates, counts, strict=True), start=1):
        relevance = 1.0 if n == 1 else 1 - (rank - 1) / (n - 1)
        citations = math.log1p(count) / top if top > 0 else 0.0
        score = relevance_weight * relevance + citation_weight * citations
        scored.append(Scored(candidate, round(relevance, 9), round(score, 9)))
    return scored


def pick_key_papers(scored: Sequence[Scored], count: int) -> list[Scored]:
    """The `count` best by key_score. Ties: higher topic_relevance, then the earlier
    publication date, then the smaller arXiv id, so the choice is deterministic."""
    ranked = sorted(
        scored,
        key=lambda s: (
            -s.key_score,
            -s.topic_relevance,
            s.candidate.facts.published_at,
            s.candidate.facts.arxiv_id,
        ),
    )
    return ranked[:count]


# --- the "what changed" lines ---------------------------------------------------------------


def _by_date(s: Scored) -> tuple[datetime, str]:
    return s.candidate.facts.published_at, s.candidate.facts.arxiv_id


def build_messages(key: Sequence[Scored], settings: Settings) -> list[ChatMessage]:
    """Key papers in date order, each as a data block holding its abstract."""
    ordered = sorted(
        key, key=lambda s: (s.candidate.facts.published_at, s.candidate.facts.arxiv_id)
    )
    hits = []
    for s in ordered:
        facts = s.candidate.facts
        text, _ = cut_to_tokens(s.candidate.abstract, settings.timeline_abstract_max_tokens)
        hits.append(
            SearchHit(
                chunk_id=s.candidate.chunk_id,
                arxiv_id=facts.arxiv_id,
                title=facts.title,
                section_path=None,
                chunk_type="abstract",
                published_at=facts.published_at.isoformat(),
                text=text,
                score=0.0,
            )
        )
    dates = "\n".join(
        f"- {s.candidate.facts.arxiv_id}: {s.candidate.facts.published_at.date().isoformat()}"
        for s in ordered
    )
    blocks = format_sources(number_sources(hits))
    user = f"Key papers in date order (arXiv id: date):\n{dates}\n\n{blocks}"
    return [
        {"role": "system", "content": load_prompt(settings.timeline_prompt).text},
        {"role": "user", "content": user},
    ]


def check_lines(
    answer: TimelineLines | None, key_ids: Sequence[str], *, max_words: int
) -> dict[str, str]:
    """The lines that pass: exactly one entry for the paper, non-empty, at most
    `max_words` words, and no mention of an arXiv id that is not a key paper."""
    if answer is None:
        return {}
    entries: dict[str, list[str]] = {}
    for item in answer.lines:
        entries.setdefault(item.arxiv_id, []).append(item.line)
    good: dict[str, str] = {}
    for arxiv_id in key_ids:
        found = entries.get(arxiv_id, [])
        if len(found) != 1:
            continue
        line = " ".join(found[0].split())
        mentioned = {m.group(1) for m in _ARXIV_ID.finditer(line)}
        if line and len(line.split()) <= max_words and mentioned <= set(key_ids):
            good[arxiv_id] = line
    return good


# --- the whole request ---------------------------------------------------------------------------

TimelineDeps = ExploreDeps  # the same collaborators serve compare and timeline


class _Precomputed:
    """The topic's vector, embedded already (see the same class in compare.py)."""

    def __init__(self, text: str, vector: list[float]) -> None:
        self._text, self._vector = text, vector

    def embed(self, texts: Sequence[str], feature: str) -> list[list[float]]:
        return [self._vector for _ in texts]


def _search_topic(
    req: TimelineRequest, deps: ExploreDeps, settings: Settings, spend: Spend
) -> list[SearchHit]:
    config = RetrievalConfig.from_settings()
    dense = None
    if config.mode != "bm25":  # ONE embeddings request, for the topic
        vector = deps.dense(EmbeddingMeter(spend, deps.record_embedding)).embed(
            [req.topic], EMBED_FEATURE
        )[0]
        dense = _Precomputed(req.topic, vector)
    return deps.search_fn(
        req.topic,
        search_filters(req),
        mode=config.mode,
        limit=settings.timeline_search_limit,
        prefetch_limit=config.prefetch_limit,
        rrf_k=config.rrf_k,
        dense=dense,
        sparse=deps.sparse() if config.mode != "dense" else None,
        client=deps.qdrant(),
    )


def _money(spend: Spend) -> float:
    return round(float(spend.cost_usd), 6)


def run_timeline(request: TimelineRequest, *, deps: ExploreDeps) -> TimelineOut:
    """Build the timeline. Raises InvalidRequest before any paid call, OpenAIKeyMissing when
    there is no key, UpstreamError when the search fails. A failed "what changed" call is
    not an error: the lines fall back to the abstracts' first sentences."""
    settings = get_settings()
    req = check_request(request)
    spend = Spend()
    prompt = load_prompt(settings.timeline_prompt)
    trace = deps.tracer.start_trace(
        "timeline", input={"topic": req.topic, "max_papers": req.max_papers}, metadata={}
    )
    # --- search
    span = trace.span("timeline-search", input={"topic": req.topic})
    try:
        hits = _search_topic(req, deps, settings, spend)
        facts = deps.facts.get_facts(sorted({h.arxiv_id for h in hits}))
    except Exception as err:
        end_span(span, error=err)
        trace.end(output="error")
        mapped = upstream(err)
        if mapped is err:
            raise
        raise mapped from err
    candidates = group_by_paper(hits, facts, req.max_papers)
    scored = score_papers(
        candidates,
        relevance_weight=settings.timeline_relevance_weight,
        citation_weight=settings.timeline_citation_weight,
    )
    key = pick_key_papers(scored, req.key_papers)
    end_span(span, hits=len(hits), papers=len(candidates), key_papers=len(key))
    # --- lines
    span = trace.span("timeline-lines", input={"prompt": prompt.id, "key_papers": len(key)})
    good: dict[str, str] = {}
    call_failed = False
    if key:
        key_ids = [s.candidate.facts.arxiv_id for s in key]
        try:
            answer, usage = deps.chat.parse(
                build_messages(key, settings),
                TimelineLines,
                feature=LINES_FEATURE,
                max_tokens=settings.timeline_max_output_tokens,
            )
        except ChatResponseError as err:
            if err.usage is not None:  # billed even though the output was unusable
                spend.add_chat(err.usage)
            call_failed = True
            end_span(span, error=err)
        except Exception as err:
            mapped = upstream(err)
            if not isinstance(mapped, UpstreamError):
                end_span(span, error=err)
                trace.end(output="error")
                raise
            call_failed = True
            end_span(span, error=err)
        else:
            spend.add_chat(usage)
            good = check_lines(answer, key_ids, max_words=settings.timeline_line_max_words)
            span.end(
                usage={"input": usage.tokens_in, "output": usage.tokens_out},
                metadata={"lines_ok": len(good), "lines_fallback": len(key_ids) - len(good)},
            )
    else:
        span.end(skipped="no papers found")
    trace.end(output="ok", metadata={"prompt": prompt.id})
    return _response(req, scored, key, good, call_failed, spend, prompt.id, settings)


def _response(
    req: TimelineRequest,
    scored: Sequence[Scored],
    key: Sequence[Scored],
    good: dict[str, str],
    call_failed: bool,
    spend: Spend,
    prompt_id: str,
    settings: Settings,
) -> TimelineOut:
    key_ids = {s.candidate.facts.arxiv_id for s in key}
    points = []
    for s in sorted(
        scored, key=lambda s: (s.candidate.facts.published_at, s.candidate.facts.arxiv_id)
    ):
        facts = s.candidate.facts
        is_key = facts.arxiv_id in key_ids
        line: str | None = None
        source: Literal["model", "abstract_fallback"] | None = None
        if is_key:
            if facts.arxiv_id in good:
                line, source = good[facts.arxiv_id], "model"
            else:
                line = first_sentence(s.candidate.abstract, settings.timeline_line_fallback_chars)
                source = "abstract_fallback"
        points.append(
            TimelinePoint(
                arxiv_id=facts.arxiv_id,
                title=facts.title,
                published_at=facts.published_at,
                citation_count=facts.citation_count,
                topic_relevance=s.topic_relevance,
                key_score=s.key_score,
                is_key=is_key,
                line=line,
                line_source=source,
                chunk_id=s.candidate.chunk_id,
                abs_url=ARXIV_ABS_URL.format(arxiv_id=facts.arxiv_id),
            )
        )
    first = points[0].published_at if points else None
    last = points[-1].published_at if points else None
    days = (last - first).days if first and last else 0
    return TimelineOut(
        topic=req.topic,
        range=DateRange(first=first, last=last),
        points=points,
        notes=TimelineNotes(
            papers=len(points),
            key_papers=len(key_ids),
            known_citation_counts=sum(p.citation_count is not None for p in points),
            narrow_range=bool(points) and days < settings.timeline_narrow_days,
            range_days=days,
            fallback_lines=sum(p.line_source == "abstract_fallback" for p in points),
            lines_call_failed=call_failed,
        ),
        prompt_version=prompt_id,
        tokens_in=spend.tokens_in,
        tokens_out=spend.tokens_out,
        cost_usd=_money(spend),
    )

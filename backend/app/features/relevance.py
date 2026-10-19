"""Relevance of a paper to the owner's interests: a pure function, no database, no API.

For each active interest i with a vector e_i:

    s_i = w_i * (0.8 * cos(abstract_vector, e_i) + 0.2 * kw_i) + 0.05 * category_match_i

- `cos` is the cosine similarity of the paper's abstract vector (read from Qdrant) and
  the interest's vector; a missing vector counts as 0.
- `kw_i` is the share of the interest's keywords found in the paper's title and
  abstract (case-insensitive, whole words or whole phrases), at most 1.
- `category_match_i` is 1 when the interest lists categories and the paper shares one
  with it, else 0. There is no penalty for a mismatch.
- `w_i` is the interest's weight, nudged by Save and Dismiss.

The paper's `relevance_score` is the largest s_i plus a small boost for Hugging Face
upvotes, `0.05 * min(1, ln(1 + upvotes) / ln(1 + 100))`, which grows on a log scale and
stops at 100 upvotes. With no active interest the score is the boost alone.

The "best interest" of a paper (named on its digest card, and the one Save and Dismiss
nudge) is the interest with the largest UNWEIGHTED part `0.8 * cos + 0.2 * kw`, so a
feedback-boosted interest does not name itself best just by being heavy. Ties go to the
smaller interest id. If no interest has any evidence (that part is 0 or less and no
category matches) there is no best interest.

All the constants come from `app.config.Settings`; nothing is hard-coded here.
Ordering is deterministic: score, then newer first, then arXiv id (a tie must never be
broken by the order the database happens to return rows in).
"""

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.config import Settings, get_settings


@dataclass(frozen=True)
class InterestVector:
    """An interest as the scorer sees it."""

    id: int
    weight: float
    keywords: Sequence[str]
    categories: Sequence[str]
    embedding: Sequence[float] | None  # None until the scoring task has embedded it


@dataclass(frozen=True)
class PaperFeatures:
    """A paper as the scorer sees it."""

    arxiv_id: str
    title: str
    abstract: str
    categories: Sequence[str]
    hf_upvotes: int | None
    published_at: datetime
    vector: Sequence[float] | None  # the abstract's dense vector; None when not indexed


@dataclass(frozen=True)
class Score:
    score: float
    best_interest_id: int | None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector is empty, all zeros or of another size."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm > 0 else 0.0


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    # Whole word or phrase: not preceded or followed by a letter, digit or underscore
    # ("rag" is not found in "storage"; "C++" works because the lookarounds do not need
    # a word character next to the plus signs). Spaces in a phrase match any whitespace.
    words = [re.escape(w) for w in keyword.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", re.IGNORECASE)


def keyword_share(keywords: Iterable[str], text: str) -> float:
    """Share of the distinct, non-blank keywords that occur in `text` (0 when there are
    none), between 0 and 1."""
    distinct = {" ".join(k.split()).casefold(): k for k in keywords if k.strip()}
    if not distinct:
        return 0.0
    hits = sum(1 for k in distinct.values() if _keyword_pattern(k).search(text))
    return min(1.0, hits / len(distinct))


def upvote_boost(upvotes: int | None, settings: Settings | None = None) -> float:
    s = settings or get_settings()
    if not upvotes or upvotes <= 0:
        return 0.0
    share = min(1.0, math.log1p(upvotes) / math.log1p(s.relevance_upvote_cap))
    return s.relevance_upvote_boost * share


def score_paper(
    paper: PaperFeatures, interests: Sequence[InterestVector], settings: Settings | None = None
) -> Score:
    """The paper's relevance score and best interest. `interests` are the ACTIVE ones."""
    s = settings or get_settings()
    boost = upvote_boost(paper.hf_upvotes, s)
    text = f"{paper.title}\n{paper.abstract}"
    paper_categories = set(paper.categories)
    values: list[float] = []
    # (unweighted part, category match, id) per interest, to pick the best interest
    parts: list[tuple[float, float, int]] = []
    for interest in interests:
        cos = (
            cosine(paper.vector, interest.embedding)
            if paper.vector is not None and interest.embedding is not None
            else 0.0
        )
        kw = keyword_share(interest.keywords, text)
        match = 1.0 if interest.categories and paper_categories & set(interest.categories) else 0.0
        inner = s.relevance_cosine_weight * cos + s.relevance_keyword_weight * kw
        values.append(interest.weight * inner + s.relevance_category_bonus * match)
        parts.append((inner, match, interest.id))
    if not values:  # no active interest: the boost alone
        return Score(score=boost, best_interest_id=None)
    inner, match, interest_id = min(parts, key=lambda p: (-p[0], -p[1], p[2]))
    has_evidence = inner > 0 or match > 0
    return Score(score=max(values) + boost, best_interest_id=interest_id if has_evidence else None)


def rank_key(arxiv_id: str, score: float, published_at: datetime) -> tuple[float, float, str]:
    """Sort key for `sorted(...)`: best score first, then the newer paper, then the
    smaller arXiv id."""
    return (-score, -published_at.timestamp(), arxiv_id)

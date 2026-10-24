"""Evaluation metrics. Retrieval metrics first, answer metrics below.

Retrieval metrics: how well a ranked list of chunk ids matches the reference chunks.

All three use binary relevance: a retrieved chunk is relevant if its id is one of the
question's `reference_chunk_ids`, else it is not (no partial credit). Each is computed
per question; the runner averages them over the questions that have references.

- Recall@k: of the reference chunks, what share is in the top k?
  "Did we find what is needed?" Order inside the top k does not matter.
- MRR (mean reciprocal rank): 1 / rank of the FIRST relevant chunk (1, 1/2, 1/3, ...),
  0 if none is retrieved. "How soon does something useful appear?"
- nDCG@k (normalised discounted cumulative gain): every relevant chunk in the top k
  earns 1 / log2(rank + 1), so rank 1 earns 1, rank 2 0.63, rank 3 0.5, ...; the sum is
  divided by the best possible sum (all reference chunks at the top), giving 0..1.
  "Are ALL the relevant chunks near the top?"
"""

import math
import re
from collections.abc import Collection, Hashable, Sequence


def _dedupe[T: Hashable](retrieved: Sequence[T]) -> list[T]:
    """A repeated item counts once, at its first (best) rank."""
    return list(dict.fromkeys(retrieved))


def recall_at_k[T: Hashable](retrieved: Sequence[T], relevant: Collection[T], k: int) -> float:
    """Share of the relevant items found in the first k retrieved."""
    if not relevant:
        raise ValueError("recall is undefined without relevant items")
    found = set(_dedupe(retrieved)[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def reciprocal_rank[T: Hashable](retrieved: Sequence[T], relevant: Collection[T]) -> float:
    """1 / rank of the first relevant item (ranks start at 1); 0 if none is retrieved.

    The mean of this over questions is the MRR. It looks at the whole list passed in,
    so the runner passes the top 10.
    """
    wanted = set(relevant)
    for rank, item in enumerate(_dedupe(retrieved), start=1):
        if item in wanted:
            return 1.0 / rank
    return 0.0


def ndcg_at_k[T: Hashable](retrieved: Sequence[T], relevant: Collection[T], k: int) -> float:
    """DCG of the first k retrieved, divided by the DCG of an ideal ranking."""
    wanted = set(relevant)
    if not wanted:
        raise ValueError("nDCG is undefined without relevant items")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(_dedupe(retrieved)[:k], start=1)
        if item in wanted
    )
    # The ideal list puts every relevant item first (at most k of them count).
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(wanted), k) + 1))
    return dcg / ideal


def mean(values: Sequence[float]) -> float | None:
    """Plain average; None for an empty list (e.g. no question had references)."""
    return sum(values) / len(values) if values else None


# Reference coverage (docs/adr/0012). Chunk ids only mean the same text under one chunking:
# with chunks of 256 tokens a reference passage of the 512-token index is two or three other
# chunks, with 1024 part of a bigger one. So ablation 7 needs a retrieval metric that looks at
# text: which share of the reference passages' words reached the top k.
SHINGLE_WORDS = 8  # long enough that a match is the same passage, short enough to survive a cut
_WORD = re.compile(r"\w+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _shingles(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def reference_coverage(
    retrieved: Sequence[str], references: Sequence[str], k: int, *, n: int = SHINGLE_WORDS
) -> float | None:
    """Share of the reference texts' word n-grams (8 words, lower case) that appear in the
    first k retrieved texts; None without reference text.

    1.0 when every reference passage is in the top k, whole or in pieces (any chunking);
    0.5 when half of it is. A reference shorter than n words counts as one n-gram of its
    own length. Only a few n-grams that straddle a cut between two retrieved chunks are
    lost, so the value is close to, not exactly, 1.0 for a passage found in pieces.
    Larger chunks carry more text into the top k, so compare chunk sizes with this
    together with the number of source tokens they cost.
    """
    found: dict[int, set[tuple[str, ...]]] = {}  # n-gram length -> n-grams of the top k
    total = matched = 0
    for reference in references:
        words = _words(reference)
        if not words:
            continue
        size = min(n, len(words))
        wanted = _shingles(words, size)
        if size not in found:
            found[size] = set().union(*(_shingles(_words(t), size) for t in retrieved[:k]))
        total += len(wanted)
        matched += len(wanted & found[size])
    return matched / total if total else None


# --- answer metrics (docs/adr/0012) -------------------------------------------------------------
#
# The judge model gives one yes/no verdict per item (a statement, a source, a citation);
# these functions turn verdicts into scores. They follow the definitions of the RAGAS
# metrics with the same names (docs.ragas.io, "available metrics"), so the numbers mean
# what those names mean elsewhere. Each is per question; runs average them over the
# questions where the metric is defined.


def share(verdicts: Sequence[bool]) -> float | None:
    """Share of True verdicts; None for no verdicts (undefined, which is not the same as 0).

    Faithfulness: supported statements / statements of the answer.
    Context recall: reference sentences found in the sources / reference sentences.
    Citation accuracy: citations whose source supports the sentence / citations.
    """
    return sum(verdicts) / len(verdicts) if verdicts else None


def average_precision(useful: Sequence[bool]) -> float:
    """Context precision: are the useful sources ranked at the top?

    `useful[i]` is the judge's verdict for the source at rank i + 1. At every rank k that
    holds a useful source, precision@k = (useful sources in the top k) / k; the score is
    the mean of these precisions over the useful ranks, so it is 1.0 when all useful
    sources come first and lower the further down they are. No useful source gives 0.0.

    Example: [True, False, True] -> (1/1 + 2/3) / 2 = 0.833.
    """
    hits = 0
    total = 0.0
    for k, ok in enumerate(useful, start=1):
        if ok:
            hits += 1
            total += hits / k
    return total / hits if hits else 0.0


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors (1 = same direction). A zero vector gives 0.0."""
    if len(a) != len(b):
        raise ValueError(f"vectors of different length: {len(a)} and {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def response_relevancy(
    question: Sequence[float], generated: Sequence[Sequence[float]], *, noncommittal: bool
) -> float | None:
    """Response relevancy: does the answer address the question that was asked?

    The judge writes a few questions that the answer would answer; the score is the mean
    cosine similarity between the embedding of the real question and the embeddings of
    those questions. An answer that drifts off topic produces questions unlike the real
    one. A noncommittal answer ("the sources do not say") scores 0 whatever its words,
    also when the judge wrote no question for it. Otherwise None when the judge wrote no
    question.
    """
    if noncommittal:
        return 0.0
    if not generated:
        return None
    return sum(cosine(question, g) for g in generated) / len(generated)

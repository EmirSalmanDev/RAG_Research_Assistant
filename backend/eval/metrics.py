"""Retrieval metrics: how well a ranked list of chunk ids matches the reference chunks.

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

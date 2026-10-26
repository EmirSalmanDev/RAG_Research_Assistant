# 0014: Hybrid search: dense and BM25 vectors fused with RRF

## Context
Questions about papers mix two kinds of text. Some are about meaning ("how do adapters avoid
forgetting?") and are answered by passages that use other words. Others hinge on an exact
token (a model name, an acronym, a dataset) that an embedding blurs. One kind of search is
good at each. The corpus is small (thousands of papers and under ten thousand chunks), runs
locally, and every question must also be checked against a measured baseline.

## Decision
Every chunk is one Qdrant point with two named vectors: `text-dense` (OpenAI
`text-embedding-3-small`, 1536 dimensions, cosine) and `text-sparse` (BM25 term weights from
FastEmbed, with the IDF part applied by Qdrant at query time). A search is one
`query_points` call with two prefetches, 50 candidates each, fused by Reciprocal Rank Fusion
with k = 60. Filters (arXiv ids, categories, publication dates) are applied inside each
prefetch, so the BM25 branch cannot bring back a point the filter excludes. Equal fused
scores are ordered by chunk id, so the same question gives the same list every time (0012).
The mode is a setting (`search_mode`: `dense`, `bm25`, `hybrid`), so the evaluation can switch
it. The details are in 0005.

Why RRF: cosine similarity and BM25 scores are on different scales, so adding or weighting
them needs a tuned constant. RRF uses ranks only. k = 60 is the value from the RRF paper and
was fixed before any measurement; it was never tuned on the evaluation questions.

## Alternatives considered
- **Dense only.** Simplest, but weakest on exact terms (below).
- **BM25 only.** Free (no embedding request) and strong on these questions (below); it cannot
  match a question to a passage that uses other words. It stays available as a mode.
- **Weighted sum of normalised scores.** Needs a weight and a normalisation, both tuned on the
  data it would then be judged on.
- **Qdrant and a second engine** (Elasticsearch, Postgres full-text search). Two systems to
  run on an 8 GB machine for something one Qdrant collection does.
- **A retrieval framework** (LlamaIndex). The vector names, payload fields and filters are
  already defined by the indexing code; one call covers the search.

## Evidence
Hybrid was the best or tied for best on both question sets, and not by a wide or certain margin.
The sets are small, so one question moves a mean by 0.04 to 0.06.
Measured on 2026-10-22 on the corpus of that day (6,653 papers, 8,912 chunks, 33 papers with
full text; 0012). Later ingests, from the live check of the daily pipeline on 2026-10-26 on,
add papers to that corpus; the numbers were not measured again.

| Mode | Golden, 20 held-out (17 scored): R@8 / MRR | Generated, 27: R@8 / MRR / context recall |
| --- | --- | --- |
| dense | 0.824 / 0.667 | 0.540 / 0.513 / 0.667 |
| BM25 | 0.912 / 0.779 | 0.620 / 0.613 / 0.767 |
| hybrid (RRF) | 0.912 / 0.775 | 0.660 / 0.627 / 0.807 |

On the held-out golden questions hybrid and BM25 are the same (MRR 0.775 against 0.779). On the
generated questions, hybrid is ahead of BM25 by 0.04 in R@8 and context recall, which is one
question. The pre-registered rule (0012, decision 13) asked whether any challenger beat hybrid
and found neither dense nor BM25 better; that is not a finding that hybrid is better than BM25.
The generated questions were written and reviewed by a model, not by the owner (0012, decision
25). The honest reading: BM25 carries most of the retrieval quality on this corpus, dense adds
a little where wording differs, and hybrid pays one embedding request per question (well under
$0.00001) for that.

## Consequences
- A hybrid or dense search sends one embedding request per question, logged as `embed-query`.
  BM25 mode needs no network.
- Changing `search_rrf_k`, the candidate counts or the embedding model changes what is
  retrieved; compare with `make eval-retrieval` before and after.
- Changing the embedding model or its dimensions means embedding everything again. The
  evaluation compared `text-embedding-3-large` (shortened to 1536 and at 3072) and found no
  gain over `text-embedding-3-small` (0012, ablation 8), so the small model stays.

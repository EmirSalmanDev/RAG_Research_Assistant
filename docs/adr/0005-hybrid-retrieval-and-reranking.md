# 0005: Hybrid retrieval, reranking and retrieval evaluation

## Context
Every chunk is a Qdrant point with two named vectors: `text-dense` (OpenAI
`text-embedding-3-small`, cosine) and `text-sparse` (BM25 term weights, with the IDF part
applied by Qdrant at query time). See 0003. Questions now have to find the right chunks.
Recall comes from combining both vectors, and precision from a cross-encoder that reads
the question and the chunk together. The machine has 8 GB of RAM, so a large model may be
loaded in one process only. Retrieval quality must be measurable from the start, on a
hand-written golden set, without spending money on every run.

## Decision

**Client.** Retrieval calls qdrant-client directly (1.19, same as the server), like
indexing does. LlamaIndex is not used: the vector names, payload fields and filters are
already defined in `app/ingest/index.py`, and one `query_points` call covers the whole
search.

**Search stage (`app/rag/search.py`).** One `query_points` call per question, in one of
three modes (`search_mode`):
- `dense`: nearest neighbours on `text-dense` (the question is embedded by OpenAI);
- `bm25`: nearest neighbours on `text-sparse` (BM25 runs locally, no API call);
- `hybrid`: two prefetches, dense and BM25 with 50 candidates each
  (`search_prefetch_limit`), fused by Reciprocal Rank Fusion.

RRF scores each point by `sum(1 / (k + rank))` over the lists it appears in. It uses
ranks, not raw scores, because cosine and BM25 scores are on different scales.
`k = 60` (`search_rrf_k`) is the value from the original RRF paper. Qdrant's default is
k = 2, which weights the top ranks much more heavily. The value was chosen on principle,
not tuned on the golden set.

Filters (arXiv ids, categories, a `published_at` date range) are applied inside Qdrant,
in each prefetch and not only on the fused query. That way the BM25 branch can never
bring back a point the filter excludes. All three fields have payload indexes.

**Rerank stage (`app/rag/rerank.py`).** The fused candidates (`search_candidates`, 50)
are rescored by a FastEmbed `TextCrossEncoder`, `BAAI/bge-reranker-base`
(`reranker_model`). It reads the same "Title > Section" text that was embedded. Then a
diversity cap keeps at most 3 chunks per paper (`max_chunks_per_paper`; 0 switches it
off), so one paper cannot crowd out the rest. Each stage has its own switch
(`search_mode`, `rerank_enabled`, `max_chunks_per_paper`), and `app/rag/retrieve.py`
runs them in order. The API and the evaluation use the same pipeline.

**The reranker model is never downloaded at runtime.** It is always loaded with
`local_files_only=True` from `fastembed_cache_dir` (on the host: `~/.cache/fastembed`).
`make fetch-reranker` is the only code that downloads it, once and deliberately. The API
container mounts the host cache read-only, with `HF_HUB_OFFLINE=1`. The model is not in
the Docker image, git or CI; tests use a fake cross-encoder.

Checked in the fastembed 0.8.1 source, with invented cache files and all network
blocked:
- The cross-encoder's cache check needs only `onnx/model.onnx` (no extra files, unlike
  BM25's `mock.file`), so a complete cache loads with no network access.
- With `local_files_only=True`, an incomplete cache fails at once, also with no network
  access.

The API loads the model lazily, behind a lock, so two first requests cannot load two
copies. The worker never loads it.

**Measured cost, and reranking off by default in `GET /search`.** Measured on 2026-10-02
in a native process on an Apple M1 (8 cores, 8 GB RAM), with the same modules the API
imports, BM25 candidates from the real collection and the configured batch size of 16:
- loading the model from the local cache took 2.09 s;
- RSS was 168 MiB with the API modules and BM25 loaded, 758 MiB with the reranker
  loaded, 841 MiB after reranking, and 1,375 MiB at the peak;
- a rerank of 50 candidates (about 394 model tokens per question–chunk pair) took a
  median of 22.0 s (mean 21.9 s, range 19.0 to 25.3 s over 15 timed runs).

That is too slow for an interactive search, so `GET /search` reranks only when the
request asks for it (`rerank=true`); `rerank_enabled` (default false) sets the default
for requests that do not say. The evaluation is unaffected: it sets rerank per
configuration, keeps `bge-reranker-base` and 50 candidates, so `hybrid-rerank` takes
about 20 questions × 22 s per run. The api container's own numbers are measured when it
is rebuilt with the model mount.

Options not taken now, to re-measure in M4 when answers depend on the reranker:
- a lower `search_candidates` (fewer pairs to score; the cost grows with their number);
- `Xenova/ms-marco-MiniLM-L-12-v2` (0.12 GB, a much smaller model; needs a download);
- limiting ONNX Runtime threads to the M1's 4 performance cores.

**Retrieval evaluation (`backend/eval/`).**
- **Golden set.** `backend/eval/golden.jsonl` has one JSON object per line: `id`,
  `question`, `reference_answer`, `reference_chunk_ids`, and optionally `type` and
  `notes`. An empty `reference_chunk_ids` marks an unanswerable question: it is stored
  but not scored. Only the owner writes this file. The 20 hand-written questions are a
  held-out check and are never tuned against.
- **Syncing questions.** Questions go into `eval_items`, matched on their exact text, so
  re-runs add no duplicates and no migration is needed.
- **Stale references.** A run stops if a reference chunk id is missing from Postgres.
  That happens when a paper got a new version or was re-chunked, because chunk ids
  depend on both.
- **Metrics** (binary relevance; computed per question, then averaged over scored
  questions):
  - Recall@8: the share of the reference chunks found in the top 8.
  - MRR: 1 / rank of the first reference chunk in the top 10 (0 if none).
  - nDCG@10: rank-discounted gain (`1 / log2(rank + 1)` per hit), divided by the best
    possible ordering.
- **Four configurations:** `dense`, `bm25`, `hybrid` and `hybrid-rerank`. Each run writes
  one `eval_runs` row (config, git commit, metrics, p50/p95 latency) and one
  `eval_results` row per question.
- **Query embeddings.** Dense and hybrid search need each question embedded by OpenAI,
  which costs money. The vectors are cached in a gitignored file, keyed by model,
  dimension and question text. `make eval-retrieval` never calls OpenAI: if a vector is
  missing, it stops and prints how many are missing, with their token count and cost.
  `make eval-retrieval EMBED=1` embeds only the missing ones in batched requests and
  logs them to `llm_usage`. So repeated runs are free.

**Departures from the original design**
1. **`bge-reranker-v2-m3` cannot be loaded.** FastEmbed 0.8.1 supports only
   `ms-marco-MiniLM-L-6-v2`, `ms-marco-MiniLM-L-12-v2`, `bge-reranker-base`,
   `jina-reranker-v1-tiny-en`, `jina-reranker-v1-turbo-en` and
   `jina-reranker-v2-base-multilingual`. The planned reranker ablation with v2-m3 is
   therefore changed at the full evaluation milestone. Candidates:
   - an ONNX export of v2-m3 registered with `TextCrossEncoder.add_custom_model`;
   - `jina-reranker-v2-base-multilingual`, which is licensed CC BY-NC 4.0
     (non-commercial);
   - `ms-marco-MiniLM-L-12-v2`.
2. **No Langfuse spans yet.** Each stage is meant to have its own Langfuse span. The
   Langfuse SDK is not a dependency yet, so the spans are added in the milestone that
   adds it. The stages are already separate typed functions, so a span can wrap each
   one.
3. **No author filter.** The Qdrant payload has no authors. Author filters come with the
   query planner, which can add the field to the payload.
4. **No LlamaIndex** (see Client above). The original stack listed it for documents,
   nodes and the Qdrant store.
5. **The diversity cap is applied in all four evaluation configurations**, not only
   after reranking. The original design describes it as part of the rerank stage.
   Applying it everywhere means "hybrid" versus "hybrid-rerank" measures the reranker
   alone. It is recorded in each run's `eval_runs.config`.

## Consequences
- Search with the `dense` or `hybrid` mode costs one embedding request per question
  (logged as `embed-query`). The `bm25` mode costs nothing.
- The reranker adds about 590 MiB of resident memory once loaded, with peaks near
  1.4 GiB while it runs (measured above), to the one process that uses it. Evaluation
  runs natively, so `/search` with `rerank=true` should not be called while an
  evaluation runs, or two copies are loaded.
- `GET /search` returns fusion-ranked results unless `rerank=true` is passed; a reranked
  request takes about 20 s on a CPU like the one measured.
- A new paper version changes its chunk ids, so golden reference ids can go stale. The
  evaluation refuses to run until the owner updates them.
- Changing `search_rrf_k`, the candidate counts or the reranker changes retrieval;
  compare `make eval-retrieval` before and after.

# 0015: The reranker: `bge-reranker-base` configured, off by default; a smaller model fits

## Context
A cross-encoder reads the question and a candidate chunk together and scores them, which is
more precise than comparing two separately made vectors. It is also the slowest and
largest part of the pipeline: a model of about 1 GB, run on the CPU, on an 8 GB Mac (a 24 GB
Mac since 2026-10-11, with a Docker VM of 7.65 GiB). The reranker was added in 0005 and
measured for quality in 0012. This record keeps the choice and the evidence in one place.

## Decision
- The reranker is a FastEmbed `TextCrossEncoder`, named by the `reranker_model` setting. The
  default name is `BAAI/bge-reranker-base`.
- **It is off by default** (`rerank_enabled = false`). A search or a chat turn reranks only if
  asked to. The default pipeline is hybrid search (0014) and the per-paper diversity cap.
- The model is never downloaded by the app. `make fetch-reranker` downloads the model named by
  `reranker_model` once; the API container mounts the host cache read-only with
  `HF_HUB_OFFLINE=1`. Only the API process loads it, behind a lock.
- **The recommendation, if reranking is turned on: `Xenova/ms-marco-MiniLM-L-12-v2`.** It is
  the only variant measured that met the bar set for this machine (at least 20 candidates,
  median rerank time per question at most 5 s, peak memory at most 2,048 MiB, retrieval quality
  kept). To use it: set `RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-12-v2` and
  `RERANK_ENABLED=true` in `.env`, run `make fetch-reranker` (about 134 MB) and restart the API.
  The change of default is the owner's decision and has not been made.

## Evidence (0012; run on this machine's CPU, each variant in its own process)
Quality with the reranker, 27 generated questions, and the same on the 20 held-out golden
questions (R@8): the first rows are hybrid search without a reranker.
Measured on 2026-10-22 on the corpus of that day (6,653 papers, 8,912 chunks, 33 papers with
full text; 0012). Later ingests, from the live check of the daily pipeline on 2026-10-26 on,
add papers to that corpus; the numbers were not measured again.

| Configuration | Generated: R@8 / context recall | Golden: R@8 | Median rerank | Peak memory |
| --- | --- | --- | --- | --- |
| hybrid, no reranker | 0.660 / 0.807 | 0.912 | - | - |
| + `bge-reranker-base`, 50 candidates | 0.900 / 0.987 | 0.794 | 5.24 s | 2,130 MiB |
| + `bge-reranker-base`, 30 candidates | 0.860 / 0.947 | - | 3.90 s | 2,822 MiB |
| + `bge-reranker-base`, 20 candidates | 0.820 / 0.867 | - | 1.92 s | 2,822 MiB |
| + MiniLM-L-12, 50 candidates | 0.880 / 0.973 (comparison run: 0.933) | 0.912 | 2.33 s | 1,500 MiB |

- On the generated questions the reranker is a large gain over hybrid search alone (context
  recall +0.18, interval [0.02, 0.38] at the corrected 98.75 % level).
- **On the held-out golden questions the `bge` reranker is worse than no reranker** (R@8 0.794
  against 0.912). The golden questions come from three papers whose reference passages hybrid
  search already ranks high; the generated ones come from 30 other, longer papers. Whether
  that explains the difference was not tested. Nothing was chosen on the held-out numbers.
- `bge-reranker-base` misses the speed and memory bar at 50 candidates; fewer candidates
  lost quality, and ONNX thread and memory-arena settings changed speed and memory only
  (identical lists) without making it fit.
- MiniLM-L-12 meets the bar, with quality close to `bge` on the generated questions and equal
  to no reranker on the golden ones.
- The two sets disagree and are small (one question moves a mean by 0.04 to 0.06); the same
  configuration run three times moved answer metrics by 0.02 to 0.04. The generated set was
  written and reviewed by a model (0012, decision 25).

## Alternatives considered
- **Reranker on by default with `bge-reranker-base`.** Best on the generated questions, worst
  on the held-out ones, and over the memory and time bar.
- **A hosted reranking API.** It would send every candidate chunk, which is paper text, to a
  third party on every question, and add a cost and a dependency to a local-first tool.
- **No reranker at all.** Simplest, and the held-out result supports it; but the generated
  result suggests a gain on longer papers. The model stays available as a setting.

## Consequences
- v1.0 ships with reranking off. Search latency is the latency of hybrid search plus one
  embedding request.
- With `bge-reranker-base` and a reranked search the API container holds about 2.1 to 2.9 GB
  and a rerank of 50 candidates takes about 5 s on this machine (first load much longer).
- Never load the model in more than one process: an evaluation run natively and a reranked
  `/search` in the API container would hold two copies.
- Any change to the reranker or to `rerank_enabled` is a retrieval change: compare
  `make eval-retrieval` before and after.

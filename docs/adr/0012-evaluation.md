# 0012: Evaluation: deterministic ties, analysis chunks, answer metrics and the test set

## Context
The evaluation has to produce numbers that can be repeated and trusted: retrieval metrics
on a fixed question set, answer metrics with a judge model, and later an ablation table.
Two things in the retrieval stage made the numbers move without any change in the code or
the data, or for reasons the golden set cannot judge:

- **Ties.** Qdrant returns equal scores in no fixed order. In hybrid mode exact ties are
  common: a chunk that is dense #1 and BM25 #2 scores exactly what a chunk that is dense #2
  and BM25 #1 scores (1/60 + 1/61). Two identical runs of the retrieval evaluation gave
  hybrid MRR 0.833 and 0.804 because one such pair swapped (docs/adr/0006).
- **Analysis chunks.** Since docs/adr/0008 each analysed paper has up to four generated
  `analysis` chunks (summary, method, results, limitations) in the same index as the
  paper's own text. They count toward the per-paper cap of 3, so a paper's three slots can
  all be filled with model output, and four views of one generated analysis can crowd the
  sources of one answer. The golden questions were written before analysis chunks existed
  and none of their reference chunks is an analysis chunk, so the golden set cannot say
  whether analysis chunks help an answer; it only counts them as misses.

Both were decided before any evaluation run of this change, on the principles below, and
never from golden-set results (the golden set is the held-out check). Each decision is
written as decision / alternative / how it is tested.

## Decision

### 1. A fixed order for equal scores, also at the cut (`app/rag/search.py`)
- **Decision.** Every result list of the search stage (dense, BM25 and hybrid) is ordered
  by score, best first, and among equal scores by chunk id, smallest first. A tie at the
  cut is settled by the same rule: the search asks Qdrant for one point more than it needs;
  while the last point fetched still has the score of the last point kept, the request is
  repeated with twice the size, until the whole tied group is in hand (`complete_ties`).
  Without a tie at the cut this is one request for `limit + 1` points, as before plus one.
  The question is embedded once, however many requests follow.
- **Why chunk id.** The goal is repeatability, not quality: nothing says which of two
  exactly tied chunks is better, and any "smarter" key (dense rank first, newer paper
  first) is a modelling choice that would need its own evaluation. The id order is
  arbitrary but stable, and the ids themselves are stable (`uuid5(arxiv_id, version, ord)`).
- **Alternative.** Fuse in our own code: two queries (dense and BM25, tie-complete at
  their own cuts) and RRF in Python. That would also fix ties at the prefetch cut (see
  Consequences), but it replaces the single Qdrant Query API call of the plan and ADR 0005
  with our own fusion and its float arithmetic. Not taken; the residual case is rare.
- **Tested by.** Unit tests with a fake Qdrant that returns ties in a chosen order: the
  same list comes out for either order in all three modes, a tie at the cut is fetched
  again (sizes 3 then 6) and the smaller id wins, a tied group longer than the window is
  fetched until it ends (3, 6, 12), the question is embedded once. An integration test
  builds an exact RRF tie in a real Qdrant (two chunks the dense and BM25 lists rank in
  opposite orders; both fused scores equal 1/60 + 1/61) and checks ten runs come back in id
  order. With the sort removed, the real Qdrant returned the pair in the other order and
  that test failed; with the widening removed, the two cut tests failed.

### 2. Analysis chunks are a secondary source (`app/rag/rerank.py`, `cap_per_paper`)
- **Decision.** The per-paper cap counts only a paper's own text (`abstract`, `section`):
  at most `max_chunks_per_paper` (3) of those. Generated `analysis` chunks are counted
  apart: at most `max_analysis_chunks_per_paper` (default 1) per paper, the best-ranked
  one. The ranking itself is not changed (no weight, no demotion): an analysis chunk keeps
  its place in the list. The setting is a stage switch for the ablations: 0 leaves
  analysis chunks out, -1 counts them like any chunk (the behaviour of ADR 0008).
- **Why.** (a) The plan keeps `analysis` as an evidence level a chat source can have, so
  leaving them out by default would contradict it. (b) The cap exists so that one paper
  cannot crowd out the others; counting model output against it lets a summary push the
  paper's own words out of the answer's sources, which nothing intended. (c) The four
  analysis chunks of a paper are four views of one generated text: one is enough, by the
  same reasoning as the diversity cap. (d) A down-weighting factor would be a constant with
  nothing to tune it on except the golden set, which is not allowed; this rule has no
  tunable constant. Comparison (ADR 0010) already keeps analysis chunks out of its cells,
  because a cell must cite the paper's own words; chat and `/search` keep one.
- **Alternatives.** (i) Leave analysis chunks out of retrieval altogether (setting 0):
  simplest, but contradicts the plan's evidence levels and hides the paper analysis from
  chat. (ii) Keep the ADR 0008 behaviour (-1). (iii) Rank all analysis chunks after the
  paper's own text: in practice the same as (i) at 8 sources, and a hidden weight.
  Which of 0, 1 and -1 gives better answers is measured with the answer metrics on
  questions that are not the golden 20 (the ablation runner, a later step).
- **Tested by.** Unit tests of `cap_per_paper`: an analysis chunk never takes a slot of
  the paper's text (three sections survive next to one analysis chunk), each paper keeps
  its own best analysis chunk and the order is unchanged, 0 removes them, -1 reproduces the
  old behaviour, and the analysis limit holds with the cap off. `retrieve` passes the
  setting through (all three values). With the separate count switched off in the code,
  six of these tests fail.

### Measured: retrieval before and after (rule-10 comparison)
Decisions 1 and 2 were committed (4a34a4d) before this run; nothing was changed after it.
`make eval-retrieval` on the 20 golden questions (17 scored), cached question embeddings
(no OpenAI request), egress guard with no approved host (0 external, 104 internal
requests) and a sandbox that allowed TCP only to the local Postgres and Qdrant. Before:
eval_runs 13-16 (dae7fbf, the same index of 6,773 points); after: eval_runs 17-20.

| Configuration   | Recall@8      | MRR           | nDCG@10       |
| --------------- | ------------- | ------------- | ------------- |
| dense           | 0.794 → 0.824 | 0.667 → 0.667 | 0.685 → 0.703 |
| BM25            | 0.853 → 0.912 | 0.784 → 0.794 | 0.787 → 0.808 |
| hybrid (RRF)    | 0.912 → 0.912 | 0.775 → 0.775 | 0.794 → 0.794 |
| hybrid + rerank | 0.794 → 0.794 | 0.667 → 0.667 | 0.685 → 0.685 |

Where the changes come from (per question, read from eval_results):
- **The cap (decision 2).** The two cases ADR 0008 recorded, where analysis chunks filled
  a paper's three slots and pushed a reference section out, are back: dense cmp-01 (Recall
  0.5 → 1.0) and BM25 cmp-02 (the reference returns at rank 6). In most lists the slots an
  analysis chunk no longer takes go to further sections at ranks 4-10. Where an analysis
  chunk of the right paper ranks above the reference passage (the other 11 cases of ADR
  0008), nothing changes, by design: the ranking is not touched.
- **The ties (decision 1).** In hybrid two exact RRF ties are now settled by chunk id:
  attn-02 (the reference and another section both 0.032522473; reference rank 3 → 2) and
  cmp-02 (both 0.032051284; reference rank 2 → 3). The MRR effects cancel (+1/6, -1/6 over
  17 questions). In w2v-01 the reference now ties with an analysis chunk (0.032795697) and
  stays second, as in run 15. Which side of a tie a reference lands on is arbitrary with
  respect to the golden set, as intended.
- **Repeatable.** Two further passes of the same retrieval (read-only, nothing stored)
  returned all 160 lists (20 questions x 4 configurations, twice) identical to the stored
  runs 17-20. Before, identical runs differed (eval_runs 3 and 7).

The golden references contain no analysis chunk, so these numbers cannot tell whether one
analysis chunk per paper helps an answer. That is measured with the answer metrics on
questions outside the golden 20, for the settings 0, 1 and -1.

## Consequences
- Identical runs give identical lists, including at the cut. One residual case: a tie at a
  hybrid PREFETCH cut (the 50th and 51st of the dense or BM25 list having exactly the same
  score) is still settled inside Qdrant, because the prefetches run inside one Query API
  call. That needs two chunks with identical raw scores exactly at position 50.
- Search sends one more point per query (`limit + 1`) and, on a tie at the cut, a second
  Qdrant request. No extra OpenAI request.
- A paper can now give up to 4 chunks to a result list (3 of its own text and 1 analysis
  chunk) instead of 3.
- The chat's own merge of the question and rephrasing lists already broke its ties by first
  appearance; its inputs are now in a fixed order, so the merge is repeatable too.

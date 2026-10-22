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

Both rules were decided before any evaluation run of this change, on the principles
below, not by trying settings on the golden set (the golden set is the held-out check).
To be clear about what that does and does not mean: the analysis-chunk problem itself was
found on the golden set (ADR 0008), and the run after the change recovers two of exactly
those golden cases. Those gains are therefore no independent confirmation of the rule;
the setting that would have helped the golden set most (leaving analysis chunks out) was
not chosen, and whether the rule helps answers is measured on generated questions.

The answer metrics of the plan (context precision, context recall, faithfulness, response
relevancy, citation accuracy, refusal accuracy) need a judge model, a way to run the chat
pipeline without saving anything, and more questions than the 20 golden ones: 40 more,
written by a model from 30 full-text papers and reviewed one by one by the owner. Only 3
papers have full text so far, so the generator and the review tool are built and tested
here, and generating the questions waits for more full texts.

Each decision is written as decision / alternative / how it is tested.

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

### 3. The RAGAS metrics, computed by our own judge calls (`eval/judge.py`, `eval/metrics.py`)
- **Decision.** Context precision, context recall, faithfulness and response relevancy
  follow the RAGAS definitions (docs.ragas.io, "available metrics"), but the verdicts come
  from our own Structured Outputs calls and the scores from our own functions:
  - faithfulness = supported statements / statements (one call writes the statements,
    one judges them against the sources the answer was given);
  - response relevancy = mean cosine similarity between the question's embedding and
    those of `eval_relevancy_questions` (3) questions written back from the answer, 0 for a
    noncommittal answer;
  - context precision = average precision of the sources judged useful for reaching the
    reference answer (precision@k at every useful rank, averaged; 0 without any);
  - context recall = reference-answer sentences the sources state / sentences.
  The embeddings of response relevancy come from `eval_embedding_model`, fixed apart from
  the retrieval's `embedding_model`, so an ablation of the retrieval embeddings does not
  change the yardstick it is measured with. A noncommittal answer scores 0 without any
  embedding request, also when the judge wrote no question for it.
  The library `ragas` is not a dependency: adding it needs the owner's approval and
  network access, and it is listed as an open item.
- **Why own calls.** Every model call must be a versioned prompt file (rule 7), show
  third-party text only inside delimited blocks the model is told never to obey (rule 8),
  and be logged to `llm_usage` with its cost (rule 9). The RAGAS prompts are inline strings
  in the library and its calls go through its own LLM wrappers. The judge also needs a
  fixed model and temperature 0, recorded on each run.
- **Differences from RAGAS, on purpose.** One call per metric judges all items of an
  answer at once (RAGAS: one per item for some metrics), which is cheaper and keeps the
  items side by side. The reference answer is split into sentences in code (the chat's own
  sentence splitter), not by the model. Response relevancy asks for the 3 questions in one
  call at temperature 0 (RAGAS samples them).
- **Alternative.** Add `ragas` (pinned, in a dependency group that is never installed in
  the api image) and run its metrics on the same samples, as a cross-check of these
  numbers. Possible later; it does not replace the rules above.
- **Tested by.** The arithmetic against hand-computed values (`test_eval_answer_metrics`:
  average precision of [yes, no, yes] = (1 + 2/3) / 2, a useful source moving down lowers
  it, a noncommittal answer scores 0, an empty list is undefined rather than 0). Each judge
  step against a scripted fake OpenAI (`test_eval_judge`): what it sends (system prompt
  file, escaped blocks, model, temperature 0, output cap), that verdicts come back in item
  order whatever order the model uses, and that a missing, extra or repeated verdict is a
  `JudgeError`, never a guess. One answer scored end to end with scripted verdicts gives the
  hand-computed scores, the uncited sentence share included
  (`test_every_answer_metric_from_scripted_verdicts`).

### 4. Citation accuracy and refusal accuracy (our own metrics)
- **Citation accuracy** = citations whose source supports the sentence / citations, per
  `[n]`, not per sentence. A sentence citing one source must be fully supported by it; a
  sentence citing several needs each source to support at least one of its claims and
  contradict none. A `[n]` that points at no source counts as unsupported without a call.
  An item names only the sentence's valid source numbers, so "X [1][7]" with no source 7
  is judged by the one-source rule for [1], and [7] counts as unsupported.
  Citation accuracy cannot see an answer that cites nothing (it has no `[n]`, so the
  metric is undefined for it), so the **uncited sentence share** (sentences without any
  `[n]` / sentences) is reported next to it, per question and per run: the chat prompt
  asks for a citation after every factual sentence, and a setting that stops citing must
  show up somewhere.
  Alternative: reuse the chat's own verify stage, which judges each sentence against all
  its sources together. Not taken: it cannot say which `[n]` is wrong, and a system that
  grades itself with its own prompt is not an independent measurement.
- **Refusal accuracy** = unanswerable questions that got the fallback (the fixed refusal,
  `no_sources` or `insufficient_sources`) / unanswerable questions. Reported with the
  **false refusal rate** (answerable questions that got it), because refusing everything
  would score 1.0 on refusal accuracy alone. A turn that failed (an error) is neither.
- **Tested by.** `test_aggregate_per_split_with_refusal_accuracy_and_false_refusals`
  (hand-built rows: 1 of 2 unanswerable refused, 1 of 2 answerable refused, an error left
  out of both) and the citation item tests (valid and invalid numbers, one and several
  citations per sentence, a missing source never makes the rule for the others lenient).

### 5. How an answer is produced and kept for evaluation (`eval/answers.py`)
- **Decision.** Each question runs through `run_chat_turn` exactly as `POST /chat` runs it,
  with the configured stage switches, as the first message of a new conversation. The chat
  store is replaced by one that keeps nothing and the tracer by the no-op tracer, so an
  evaluation leaves no conversations behind and sends nothing to Langfuse. The search
  function is wrapped only to remember the returned chunk texts, so the judge sees exactly
  the sources generate saw. Latency and cost come from the turn itself (the `done` event and
  a clock); the judge's own cost is kept apart, including calls that were billed but gave
  no usable answer.
- **Stopping and keeping.** A judge answer that does not fit what was asked (a wrong
  verdict list, a refusal, output cut off) loses only that metric for that question, with
  the reason stored. Any other failure (an API or network error, a wrong model name) stops
  the run, so a broken judge never pays for a whole run of empty scores. The eval_runs row
  is written before the first paid call (status "running"), each question's result is
  committed as soon as it is scored, and the end writes the metrics with status
  "complete" or "stopped: <reason>" and the run's real spend, read back from the
  llm_usage rows written since it started (which also covers turns that failed).
- **Quick runs.** `--limit N` needs `--split heldout` or `--split generated`: the held-out
  questions come first in the set, so a limit over both would quietly be a held-out-only
  run.
- **Alternative.** Call the stages one by one in the evaluation. Not taken: what is
  measured must be what the API serves.
- **Tested by.** A full turn on scripted fakes through the real pipeline: answer, numbered
  sources with their texts, cost = the three chat calls plus three query embeddings, no
  condense call for a first message; both refusals and a failed stage are told apart. A
  refused judge answer loses one metric and is still counted in the judge's cost; an
  unexpected judge error stops scoring; `run` hands every row on as soon as it is scored;
  on a scratch database, a stopped run keeps its result rows and its spend counts only
  the llm_usage rows written after it started.

### 6. The question set: held-out and generated questions apart (`eval/golden.py`)
- **Decision.** The 20 golden questions stay in `golden.jsonl` (owner only) and are tagged
  "heldout"; accepted generated questions go to their own file, `generated.jsonl`, tagged
  "generated". Ids and question texts must be unique across both. Every choice between
  settings must use `for_tuning()`, which returns the generated questions only; the
  held-out ones are reported next to them, never used to choose. Runs report every metric
  per split and for both together.
- **Alternative.** One 60-line file with a flag per line. Not taken: the review tool would
  then write into the owner's file, and a slip could turn a held-out question into a
  tuning question.
- **Tested by.** Loader tests: the split tags, a missing or empty generated file, a
  generated question that repeats a golden id or text is refused, and `for_tuning` returns
  no golden question.

### 7. The test set: our own generator, seeded from full-text chunks (`eval/make_testset.py`)
- **Decision.** One Structured Outputs call per seed writes a question and a reference
  answer from the seed text (`testset_<type>_v1`, the evaluation model, temperature 0).
  Four types, 60 candidates for the 40 the plan keeps after review: 30 `fact` (one
  section chunk), 12 `compare` (two section chunks of two different papers), 8 `filter`
  (one abstract with its date and categories: "which paper in <period> did X") and 10
  `unanswerable` (one section chunk: a question about that paper it does not answer).
  The seed chunks become the question's reference chunk ids, so retrieval and answer
  metrics both apply; unanswerable questions have none. Seeds are drawn with a fixed
  random seed, one paper at a time in turns, from section chunks of at least 120 tokens,
  never from reference lists or acknowledgements. The pool is every full-text paper that
  no golden question refers to; generation refuses to start with fewer than 30 (the
  plan's number) and lists papers to add. (Compare and unanswerable seeds changed after
  the first real round: decisions 21-24.)
- **Golden papers are left out whole.** Every paper a golden question refers to (found
  through the golden reference chunks; one missing from the database stops the tool) is
  left out of the pool: its abstract, all its sections, not only the reference chunks.
  Leaving out only the reference chunks would leave their neighbours, which repeat 64
  tokens of the same passage, and the paper's abstract, which a filter question would be
  built on. It also keeps the held-out set an independent check, on other papers than the
  ones tuning looks at. Cost: the 3 papers that have full text today are all golden
  papers, so all 30 seed papers still have to be added, and the papers with generated
  analyses (only those 3 so far) cannot seed questions, so comparing the analysis
  settings 0, 1 and -1 on generated questions needs analyses of new papers.
- **Why not RAGAS test-set generation.** It is not a dependency (see decision 3), it
  builds a knowledge graph with many model calls per document whose prompts and spend sit
  outside our rules, and its contexts are texts, not our chunk ids. Seeding from our own
  chunks gives reference chunk ids for free and bounds the cost at one call per seed.
- **Which papers to add.** The rule is fixed in advance: abstract-only arXiv papers with
  the most Hugging Face upvotes first (papers the community picked out), then the
  newest; never a golden paper. The owner may prefer papers from their own research;
  that is their choice.
- **Alternative.** Hand-written questions only (best quality, slowest), or one prompt
  that writes several questions per paper (cheaper, but questions then cluster on a
  paper's abstract-level claims).
- **Tested by.** Seed tests on an invented pool (same seed same seeds, another seed
  different; short chunks, acknowledgements and reference lists never seeds; no chunk of
  a golden paper seeds any type over 20 random seeds, its abstract included; compare
  pairs always span two papers; facts visit every paper before any repeats; a small pool
  gives fewer seeds instead of repeats). The model call against a
  scripted fake (escaped data blocks, dates only for filter questions, model, temperature
  and output cap, one llm_usage feature per type, a re-run sends nothing for seeds already
  done). The estimate equals the locally counted input tokens and the output cap per call.
  On a scratch database: the pool, the golden papers found through their reference
  chunks, the upvote order of the suggestions (never a golden paper), golden papers not
  counted toward the 30, and a golden set that disagrees with the database stopping the
  tool.

### 8. Review: the owner decides every question (`eval/make_testset.py review`)
- **Decision.** Candidates the model marked unusable are kept (they were paid for) but
  never shown. Each pending one is shown with the full text of its seed chunks (read from
  Postgres, never stored in the candidates file, so that file holds no paper text and can
  be committed); an unanswerable one also with the 3 closest passages of the same paper
  by a BM25 search on this machine (free), because the seed chunk alone cannot show
  whether another part of the paper answers the question. A warning appears when the
  question's word overlap with a golden, a generated or an accepted question is at least
  0.6. Accept gives it the next `gen-NNN` id after the highest in `generated.jsonl` and in
  the candidates file; reject asks for an optional reason; skip leaves it pending; quit
  stops. Every decision is saved before the next question (write to a temporary file,
  then rename). `generated.jsonl` is rewritten as its questions that the candidates file
  does not hold (for example from an earlier candidates file) plus the candidates file's
  accepted ones; if the two files disagree about an accepted question (an id with another
  text, after a hand edit), review stops before changing anything. A question whose text
  equals a golden, a generated or an accepted one cannot be accepted. A half-written last
  line in the candidates file (an interrupted run) is reported with how to fix it.
- **Alternative.** Accept everything the model marks usable and let the owner delete
  lines later. Not taken: the plan asks for a review, and the generated questions decide
  which settings win.
- **Tested by.** A scripted review session: accept, reject with a reason, an exact
  repeat of a golden question refused and then skipped; the generated file loads as the
  "generated" split next to the golden file, whose bytes are unchanged; a second session
  resumes at the first pending candidate and numbers on from the last id; a second
  candidates file keeps the first file's questions and numbers on after them; a question
  already generated elsewhere cannot be accepted; a hand-edited `generated.jsonl` stops
  the review with both files unchanged; the whole seed text and the same-paper passages
  are shown; unknown answers are asked again; a half-written line is a clear error.

### 9. The evaluation's own block tags are escaped only in its own prompts
- **Decision.** The judge and the test-set prompts use two more data blocks, `<answer>`
  and `<reference>`. Text that goes into those prompts is escaped for them too
  (`escape_block(text, EVAL_TAGS)`); the chat's, comparison's, review's and analysis's
  escaping keeps exactly its old tag list, so their prompts are byte for byte what they
  were (no generation change, so no answer evaluation is owed for it).
- **Alternative.** Add the two tags to the shared list. Simpler, but it changes the input
  of every chat prompt whose source text contains such a tag (common in papers about
  reasoning models), which is a generation change.
- **Tested by.** The shared escaping leaves `<answer>` alone and the evaluation escaping
  neutralises it (and still the shared tags); a judge prompt neutralises `</answer>` and
  `<reference>` inside a source, while the chat's sources show them unchanged.

### 10. The ablation runner: variants in steps, each in two parts (`eval/variants.py`, `eval/runner.py`, `eval/ablations.py`)
- **Decision.** Each configuration is a variant: settings overrides plus, for ablations 7
  and 8, the index it searches. Variants are grouped in steps that follow the plan's
  order: retrieval (ablations 1-5: dense, BM25, hybrid, hybrid + `bge-reranker-base`,
  hybrid + a second reranker), rewriting (6: the planner's two rephrasings on and off),
  chunking (7: 256, 512, 1024 tokens), embedding (8: `text-embedding-3-small` at 1536,
  `text-embedding-3-large` shortened to 1536 with the `dimensions` parameter, and at
  3072), and the reranker comparison moved from M4 (threads 4/2/1, the ONNX memory arena
  off, 30/20/10 candidates, the small model), on ablation 4's configuration. A step after
  the first runs on the winner of the step before (decision 13), stored as a
  `selection:<step>` row; `--base VARIANT` names one by hand, recorded as such. Each step
  has one incumbent, what the app runs today (hybrid without rerank, rephrasings on, 512
  tokens, the small model; for the reranker comparison the reference is a4 as it is).
  Every variant runs in up to two parts, each its own eval_runs row
  (`ablation:<variant>:retrieval|answers`): retrieval (`retrieve()` on each question, top
  10, free with cached question vectors) and answers (the chat pipeline and the judge,
  paid). Each question's result is stored at once, with the retrieved chunks' texts
  (an ablation index's chunk ids mean other texts than the chunks table, and the
  per-question view must show what the judge judged).
- **Choices inside.** "Query rewriting" is the planner's rephrasings: condense, the other
  rewriting stage, never runs for the first message of a conversation, and every
  evaluation question is one. The chunk-size variants keep the configured overlap share
  of 1/8 (32 and 128 tokens), so they differ in scale only. The second reranker is
  `Xenova/ms-marco-MiniLM-L-12-v2` (`eval_second_reranker_model`): of the candidates
  docs/adr/0005 lists, a `bge-reranker-v2-m3` export is not a FastEmbed model and the
  jina v2 reranker is non-commercial (CC BY-NC); the same model is the "MiniLM" of the
  reranker comparison. The threads and arena variants change speed and memory, not the
  ranking, so they run the free retrieval part only and must return exactly the
  reference's lists; fewer candidates and the small model change the ranking, so their
  answers are measured too. Model names come from the settings, never from the variant
  table (rule 2).
- **Alternative.** One process per variant for everything (no settings scope needed);
  the reranker comparison does this anyway for its memory numbers (`--isolate`). Kept
  in-process by default because it is simpler to drive and test, with decision 11
  making it safe.
- **Tested by.** `test_eval_variants` (all 8 ablations present, one incumbent per step,
  model names read from the settings, collection names, a later step built on a winner,
  a stored record round trips), and the runner tests below.

### 11. A configuration's settings apply to the whole process, and what ran is checked (`eval/scope.py`)
- **Decision.** `run_chat_turn` builds its retrieval config from `get_settings()`, not
  from its `settings` argument; `search` takes the collection, `DenseEmbedder` the model,
  `embedding_cost` the price, `get_cross_encoder` the reranker from the same cached
  object. So each part runs inside `applied_settings(overrides)`: the overrides become
  environment variables (they win over `.env`), the settings cache is cleared, every
  override is read back and compared (a value that did not arrive stops the run before
  anything is spent), and on the way out, also after an error, the environment is
  restored exactly and the caches cleared again. Scopes do not nest. Keys, store
  addresses and every `eval_*` setting (the judge, its prices, the relevancy embeddings)
  can never be overridden: an ablation must not move its own yardstick. On top of that,
  what the run did is observed (the model and size each query encoder was built with,
  read from the encoder itself; the reranker that was actually loaded; the mode each
  search is asked for; the collection the chat's search reads from the settings at call
  time, which catches a scope lost in the middle of a run, not a stale object) and
  compared with what the settings allow; a difference stops the run as "stopped:
  configuration mismatch". The configuration stored on a run is read back inside the
  scope. The BM25 model (`sparse_model`, loaded once per process) cannot be overridden.
  Each question also records the stages that failed and fell back
  (`stage_errors`, read from the turn's own record): a plan that failed runs the default
  plan without rephrasings, which is not the configuration under test.
  `get_cross_encoder` now keeps the model together with the settings it was loaded with
  and releases it before loading another (decision 17).
- **Alternative.** Thread the settings through every stage as an argument. That changes
  the chat pipeline's signature and every caller in `app/rag` (a retrieval-affecting
  change) for the evaluation's sake; and any reader missed would leak silently, which the
  observation would still be needed to catch.
- **Tested by.** `test_configurations_run_in_one_process_never_leak_into_each_other`:
  BM25, then a large-embedding index with the bge reranker, then the second reranker,
  then BM25 again, in one process: for each, the mode and collection the search and
  retrieve stages got, the embedding model and size of every embeddings request, the
  stored effective configuration, the reranker loaded (each model loaded when its
  configuration asked, never reused), and afterwards the original settings and an
  environment without the overrides. A stage handed a stale reranker stops the run
  (`test_a_reranker_other_than_the_settings_ask_for_stops_the_run`). `test_eval_scope`:
  values reach `get_settings()` and come back (an unset variable is unset again, a set
  one keeps its value), also after an error; the shared dense embedder is rebuilt;
  unknown, forbidden, `eval_*` and `None` overrides and a value pydantic refuses change
  nothing; scopes do not nest.

### 12. Context precision and recall also for a refused answer
- **Decision.** The two context metrics judge the sources against the reference answer,
  not the answer (as in RAGAS), so they are computed whenever the turn retrieved
  something, also when the generator then refused for insufficient sources. A refusal
  because nothing was retrieved has context recall 0 (nothing can state the reference)
  without a call, and no context precision. A failed turn has none. Part 1 computed them
  for answers only.
- **Why.** The selection rule compares configurations question by question on context
  recall. With answers only, a configuration that makes the generator refuse more often
  would drop exactly its bad questions from its mean.
- **Alternative.** Keep part 1's rule and count refusals as recall 0. Not taken: a
  refusal with good sources (a cautious generator) would then look like bad retrieval.
- **Tested by.** `test_a_refusal_with_sources_gets_the_two_context_metrics_only`,
  `test_a_refusal_without_sources_has_context_recall_zero_without_a_call`,
  `test_a_failed_turn_gets_no_metric_and_no_judge_call`.

### 13. The rule that picks "the best of the above", fixed before any result (`eval/selection.py`)
- **Decision.** Written down here and in the code before any ablation ran:
  1. only generated questions (`for_tuning()`); the runs' held-out rows are dropped when
     they are read, and the rule raises if one reaches it anyway;
  2. paired: the generated questions both runs scored without an error and without a
     stage falling back; fewer than 80 % of the questions where either run has the
     metric paired is "undecided";
  3. primary metric: context recall (it does not depend on chunk ids, so chunk sizes can
     be compared, and it is defined for refusals, decision 12);
  4. a challenger is better only if the paired bootstrap interval of its mean
     difference to the incumbent lies above 0 (10,000 resamples, fixed seed, percentile
     interval), at 1 - 0.05/m for m challengers (Bonferroni; 98.75 % for the four of the
     retrieval step), so noise crowns any challenger with at most about 5 % chance;
  5. guard: a better challenger significantly worse on faithfulness (its interval below
     0) is disqualified;
  6. the winner is the better, not disqualified challenger with the highest mean; with
     none the incumbent stays.
  The runs compared come from one suite (one `make eval-ablations` invocation, in which
  every variant of the step completed; the newest such suite unless one is named) and
  must share the question file, the judge, the commit, the base and the corpus size.
  The reranker comparison runs on generated questions only (`--split generated`, its
  default), because its speed and memory are measured over every question a run took. The decision,
  every comparison (pairs, means, intervals, outcome) and the run ids are stored as a
  `selection:<step>` row. The reranker comparison asks whether a cheaper runtime setting
  is good enough, so it has its own rule: the owner's M4 bar (at least 20 candidates,
  median rerank time per question at most 5 s, peak memory at most 2,048 MiB, measured
  in a process that ran that variant alone), and quality kept (identical retrieval lists
  for threads and arena; not significantly worse context recall for fewer candidates and
  the small model). It recommends the fastest qualifying variant; changing the app's
  defaults stays the owner's decision.
- **Why this rule.** With about 30 scored generated questions, one question moves a mean
  by about 0.03; a rule that takes the highest mean would pick noise. Requiring an
  interval above 0 makes "no significant difference, keep what runs" the usual outcome,
  which is the honest result of a small test set. A percentile bootstrap on about 30
  values tends to be somewhat too narrow; the correction leans against that, and every
  interval is stored for the reader to judge.
- **Known bias.** Larger chunks put more text in front of the generator, which raises
  context recall by itself; the chunking step therefore also reports the source tokens
  and coverage, and the owner should read its winner with that in mind.
- **Alternative.** The highest mean of a composite score. Not taken: weights would be
  another unvalidated constant, and noise would pick the winner.
- **Tested by.** `test_eval_selection` (a repeatable bootstrap, the incumbent staying on
  noise, a clear winner and the best of several, the faithfulness guard, "undecided"
  with few pairs, held-out questions refused on either side, the reranker bar cases) and
  on a scratch database `test_a_step_is_run_stored_selected_on_generated_questions_and_built_on`
  (five variants stored, the rule reads 30 pairs from 31 questions, the winner stored and
  the next step built on it) and `test_selection_refuses_runs_of_different_question_sets`.

### 14. Reference coverage: a retrieval metric without chunk ids (`eval/metrics.py`)
- **Decision.** Recall@8, MRR and nDCG@10 count reference chunk ids, which only mean the
  same text under the configured chunking. Every retrieval and answers run also reports
  reference coverage: the share of the reference passages' word 8-grams (lower case)
  found in the top 8 retrieved texts (`coverage@8`) or in the generator's sources
  (`source_coverage`). It is 1.0 for a passage retrieved whole or in pieces under any
  chunking, minus the few 8-grams that straddle a cut. Id metrics are left out (not
  zero) for an index with other chunking.
- **Alternative.** Map each reference chunk to the variant's chunks by overlap and keep
  id metrics. Not taken: the mapping needs its own threshold, and a reference split into
  three small chunks would need all three in the top 8 for full recall.
- **Tested by.** `test_eval_coverage` (whole, in two pieces, inside a bigger chunk, only
  the top k, case and punctuation, short and missing references) and the runner test
  that a 256-token index gets coverage but no id metrics.

### 15. A hard spend cap: the egress guard in the evaluation process (`eval/budget.py`)
- **Decision.** Every paid run needs `--max-usd`: the ablation runner, the suites, the
  index builder and now also part 1's `python -m eval.answers`. The runner installs the egress guard of
  the proof runs with one approved host (api.openai.com), the local stores as internal
  hosts and that cap: before any request is sent, its worst case (each byte of the body
  as an input token, its `max_tokens` as output, at the highest price the run can be
  billed at) is added to a running bound, and a request that would take the bound past
  the cap is refused. The real spend is below the bound, so below the cap. Because the
  chat pipeline falls back when its plan or verify call fails, a refused request would
  quietly turn a question into a worse answer, so the runner checks the guard after
  every question: a tripped guard stops the run and that question is not stored. To stop
  cleanly in the usual case it also stops before a question when the cap left is less
  than the largest bound one question has added. `estimate` prints the requests and the
  expected cost (from measured turns) and a suggested cap; the suggestion is
  approximate, the cap is not. Because the bound takes every request at its output limit
  and every body byte as a token, a cap is about ten times the real spend: on the 20
  golden questions `make eval ESTIMATE=1` gives about $0.078 expected (measured bound
  $0.27) and suggests a cap of $0.95, about $0.05 a question. The defaults are set from
  that bound (smoke $1.00 for 15 questions, a full run $3.50). The index builder's
  estimate prints its bound too (`text-embedding-3-large` at 3072 for the corpus today:
  about $0.25 expected, a bound of $1.35).
- **Alternative.** Stop when the llm_usage spend passes the cap. Not a cap: the spend is
  only known after the request.
- **Tested by.** `test_a_tripped_guard_stops_the_run_and_drops_the_unfinished_question`,
  `test_the_run_stops_cleanly_before_a_question_the_cap_cannot_cover`,
  `test_a_paid_suite_run_needs_a_cap`, and the guard's own tests.

### 16. `make eval-smoke` and `make eval` (`eval/suite.py`)
- **Decision.** Both run the configured pipeline (no setting changed) in both parts.
  `make eval` takes every question, reported per split. The smoke set is 15 GENERATED
  questions, never held-out ones: a pull request is judged on it, and a held-out question
  that decides whether a change goes in is a held-out question tuned on. It is chosen by
  a fixed rule (round robin over the question types, each type in the order of the
  SHA-256 of the question id), so it stays the same 15 while generated.jsonl does not
  change; the ids are stored with each run. With fewer than 15 generated questions the
  smoke run refuses to start. `--report-dir` writes the markdown tables a pull request
  comment shows and a JSON export of the runs without chunk texts (an artifact of a
  public repository must not re-publish paper text), in every case, also after an
  error the command line does not know.
- **Alternative.** A smoke set with held-out questions (available today). Not taken, for
  the reason above.
- **Tested by.** `test_eval_suite` (15 generated questions of every type, independent of
  file order and of held-out questions, the error with 14, the report table, the export
  without texts, the cap) and the export on a scratch database.

### 17. Reranker runtime settings in the app (`app/rag/rerank.py`)
- **Decision.** Two settings, `rerank_threads` (default None: ONNX Runtime's own count)
  and `rerank_cpu_arena` (default on), passed to FastEmbed only when they differ from its
  defaults (its `threads` argument and its exposed `enable_cpu_mem_arena` session
  option), so the default load is exactly the call measured in docs/adr/0005.
  `get_cross_encoder` keeps the loaded model with its key (model, cache, threads, arena)
  and loads another only when the settings ask for another, dropping the old one first.
- **Alternative.** Build the variants' cross-encoders inside the evaluation only. Not
  taken: what is measured would then not be what the app can be set to run.
- **Tested by.** `test_the_default_load_passes_no_runtime_option` (the exact FastEmbed
  call as before), `test_a_thread_limit_and_the_arena_switch_reach_onnx_runtime`,
  `test_other_reranker_settings_release_the_old_model_and_load_the_new_one`, and a
  read-only replay of the stored rule-10 runs: the 20 golden questions retrieved again
  with this code in the four configurations of `make eval-retrieval` (cached question
  vectors, `bge-reranker-base` from the local cache, a read-only database transaction,
  the egress guard with no approved host: 0 external requests) gave 80 of 80 lists
  identical to eval_runs 17-20; nothing was stored (eval_runs, eval_results and
  llm_usage unchanged). A stored `make eval-retrieval` run writes eval_runs, which this
  change was not cleared to do, so the replay stands in for it. The 20 reranked
  questions took 99.8 s on this machine (about 5 s per question, 50 candidates).

### 18. A collection per index variant, built from Postgres (`eval/indexes.py`)
- **Decision.** A chunking or embedding variant searches its own Qdrant collection, named
  after what it holds (`<collection>__eval__c<tokens>o<overlap>__<model>-<dim>[-short]`),
  with the app's layout (`ensure_collection`). It is built from Postgres, which is only
  read, like the app's collection: abstract and analysis chunks are the same under any
  chunking and come from the chunks table; section chunks under another chunking are made
  again from the paper's cleaned text (`paper_texts.markdown`) with the variant's sizes,
  with the ids the app would give them. With the configured model and size, dense
  vectors are copied from the app's collection (no request); otherwise everything the
  variant holds is embedded with its model (with `dimensions` when shortened), logged as
  `eval-embed-index` at that model's price. BM25 vectors are computed locally. The build
  is resumable (points already there with the same text are skipped), checks the final
  point count against its plan (and the runner refuses to search a variant collection
  that does not hold exactly what Postgres says now: an interrupted build or papers
  added since), refuses to write to any collection without the
  `__eval__` mark, and stops when a paper with section chunks has no stored text. An
  embedding variant builds on the chunking winner (stored) or `--base a7-...`.
- **Alternative.** Re-index into the app's collection with other settings and restore it
  afterwards. Not taken: the app would search a half-built index meanwhile, and the
  owner's corpus would be at risk.
- **Tested by.** On a scratch database and scratch collections
  (`test_eval_indexes`): 256-token chunks re-made from the stored text (more chunks than
  at 512, only those embedded, abstract and analysis vectors copied unchanged), a second
  build embeds nothing, the shortened large model embeds every point with `dimensions`
  and logs it, a paper without text stops the build, the app's collection is never
  written, the command line estimates for free and needs a cap to embed; the real
  collection is checked untouched.

### 19. `eval.yml`: a capped evaluation in CI on a restored corpus (`.github/workflows/eval.yml`, `eval/corpus.py`)
- **Decision.** The workflow runs the smoke suite on pull requests that carry the
  `retrieval` label (when that label is added and on every later push; never for a
  pull request from a fork, which gets no secrets anyway) and the full or the smoke suite
  by hand (`workflow_dispatch`, with the cap as an input). Every run passes `--max-usd`
  (1.00 for the smoke run; 3.50 by default by hand), posts the metrics tables as a pull
  request comment and uploads the runs as an artifact (JSON without chunk texts), also
  after a stop; a run that did not complete fails the job. A new push cancels the older
  run of the same pull request. The OpenAI key reaches the paid step only.
- **The corpus.** A GitHub runner starts with empty stores, and the corpus cannot be in
  the repository (it holds paper text; the repository will be public). So the owner
  exports a bundle (`python -m eval.corpus export`: the `papers` and `chunks` tables as a
  data-only `pg_dump`, a Qdrant snapshot of the collection, a manifest with counts, the
  alembic revision, the embedding model and SHA-256s), keeps it at a private location, and
  gives the workflow its URL and SHA-256 as secrets. The workflow checks the hash, applies
  the migrations, restores both into service containers of the same versions as
  docker-compose.yml (postgres 16, Qdrant v1.19.1) and runs `eval.corpus check` (counts,
  revision, model, file hashes) before the paid step. The bundle's alembic revision
  must equal the code's: after a migration the owner exports the bundle again (a
  data-only dump of an older schema may not load into a newer one). The concurrency
  group sits on the job, so only a run that passed the label condition can cancel an
  older one; a pull request event that skips the job (another label) cannot. Nothing personal is in a bundle
  (no conversations, library, interests or spend).
- **Alternatives.** (a) A self-hosted runner on the owner's Mac, next to the real stores:
  no bundle, but a public repository's workflows would run on that machine, which GitHub
  advises against, and every CI run would write to the owner's stores. (b) Ingest a small
  corpus in CI from arXiv: slow, rate-limited, paid each run, and not the corpus the
  numbers are about. Not taken.
- **Tested by.** The workflow cannot run here; `test_eval_workflow` parses it and pins
  the triggers and the label and fork conditions, the permissions, smoke on pull requests
  and the cap on every run, the key in one step only, the order verify, restore, check,
  run, the artifact and the comment after a stop, pinned actions, and that every module
  it calls imports. `test_eval_corpus` exports a scratch collection and database (an
  invented dump, a real snapshot that is deleted from the server again), reads the bundle
  back, and checks that lost points, a changed file and another schema are reported.

### 20. The /evals page (`frontend/app/evals`, `components/evals`, `lib/evals`; `app/api/evals.py`)
- **Decision.** Four read-only routes (`GET /evals/runs`, `/evals/runs/{id}`,
  `/evals/runs/{id}/results`, `/evals/runs/{id}/results/{item}`) read Postgres only and
  normalise every run shape stored so far (the flat metrics of `make eval-retrieval`,
  part 1's `answers:<name>`, the runner's `<suite>:<variant>:<part>`, selections) into a
  kind, a series and headline numbers (the "all" split). The page shows: metric trends
  across runs (one line per series, a series being one variant in one suite kind, or the
  name of an older run; a metric's runs only; selections never drawn), two runs side by
  side (headline metrics with B - A coloured by whether lower or higher is better, a
  split filter that says held-out questions are reported and never used to choose, and
  every question joined on its id), and one question in a side panel (the reference
  answer, the answer or refusal, every score, the judge's verdicts drawn by their shape,
  judge errors, and the retrieved chunks with their texts, source numbers, a reference
  badge and an arXiv abstract link, never a PDF). A run that stopped or is still running
  is marked as such everywhere and drawn as a hollow point that no line joins.
  Chunk texts come from what the run stored; only a run on the app's own collection that
  stored none is looked up in the chunks table. A run on another chunking marks no chunk
  as the reference (its chunk ids name other texts than the reference chunks).
- **Charts in plain SVG (a deviation from the plan's Recharts).** Adding Recharts is a
  dependency and needs network access, which this change was not cleared for; the
  timeline page (docs/adr/0010) already draws its chart as plain SVG with the numbers
  computed in tested TypeScript, and the trend chart follows it. Recharts stays an open
  item for the owner.
- **Alternative.** A comparison endpoint that joins two runs on the server. Not taken:
  the join is a few lines over two lists the page loads anyway, and it is tested in
  `lib/evals/compare.test.ts`.
- **Tested by.** `test_evals_api` (every run shape, headline metrics, per-question rows,
  stored texts versus the chunks table, a variant index never borrowing texts, 404s) and
  `test_evals_store` on a scratch database; Node tests of the series, scales, states,
  join, split filter, sort and verdict shapes; and a Playwright flow on a mock API
  (`e2e/evals-flow.spec.ts`: trends with a stopped run, two runs compared, a question
  opened with its chunk text, reference badge, answer, verdicts and arXiv link; the
  empty, error and selection states), run by `make e2e`.

### 21. What the first generation round showed (the reason for decisions 22-24)
- **Facts.** The first real round (60 calls, the evaluation model at temperature 0, on 30
  full-text papers) gave 33 candidates to review (26 fact, 7 filter) and none for compare
  (0 of 12) or unanswerable (0 of 10): each of those 22 calls returned `usable: false` with
  an empty question, 12 output tokens each. The output shape has no field for a reason,
  so the model's reasons are not known.
- **Evidence about the seeds** (stored metadata and stored vectors only): 10 of the 12
  compare pairs joined papers of two primary categories (for example a financial
  time-series model and a music generator), and in all 12 the two excerpts were different
  kinds of passage (an introduction and an appendix, related work and a qualitative
  example), with a cosine similarity of 0.19-0.49 between their vectors. The v1 compare
  prompt allows `usable = false` exactly when the excerpts share no aspect. The 10
  unanswerable seeds were ordinary sections of 120-511 tokens.
- **Guesses (not tested).** The v1 unanswerable prompt asks for a detail "the paper is
  unlikely to report at all" while the model sees one excerpt, so it cannot judge the
  paper and takes the way out it is offered. The v1 shape asks for the verdict before the
  question, so the model decides before it has tried to write one (fact still gave 26 of
  30, so this alone does not explain the result).
- Decisions 22-24 were written after that round and before the next one, without any
  retrieval or evaluation result and without the golden set. Fact and filter seeds and
  prompts are unchanged.

### 22. Compare seeds: two papers on one topic and their two most similar sections (`topic_pairs`, `closest_excerpts`)
- **Decision.** Papers are paired by topic, from what is stored. Every pair of pool papers
  is ranked by (1) the same `primary_category` first, (2) the cosine similarity of their
  abstract vectors (the app's stored vectors, read from the local index; no request),
  highest first, (3) the two arXiv ids. Pairs are taken in that order unless a paper is
  already in an earlier pair, so each paper is in one comparison at most (30 papers: up to
  15 pairs; 12 are asked). The two excerpts are the eligible sections (the rules of every
  seed), one of each paper, whose stored vectors are the most similar (ties: the smaller
  chunk ids), never a chunk that already seeds a fact question. Nothing is random. The
  first rule's random draws are still made and not used, so the filter seeds drawn after
  them, already generated, stay exactly the same, and so do their candidate ids. The
  prompt `testset_compare_v2` says the excerpts are the two most similar passages of the
  papers, lists the aspects a comparison may be about (also how each paper evaluates, or
  what each names as a limitation, when the papers solve different problems), and allows
  `usable = false` only when the excerpts share no aspect at all.
- **Alternatives.** (a) Same category, random pairs and excerpts: the excerpts would still
  be unrelated passages. (b) The nearest abstracts without the category: categories are
  coarse, but stored and independent of the embedding model, so a near neighbour from
  another field does not win. (c) Pairs chosen by hand: the owner decides in review; the
  rule only proposes. **Known bias:** the most similar sections of two papers are often
  their introductions or related-work sections, so comparisons lean towards motivation
  and positioning rather than results; the review decides whether they are useful.
- **Tested by.** `test_compare_pairs_papers_of_one_category_first_then_the_closest_abstracts`
  (a closer pair across categories loses to pairs within one; without the category the
  closest abstracts pair first; each paper once; no more pairs than the papers allow),
  `test_equal_similarities_pair_the_smaller_ids_first`,
  `test_the_excerpts_are_the_two_most_similar_sections`,
  `test_a_compare_excerpt_is_never_a_fact_seed` (two papers; the fact seed made the closest
  section is skipped), `test_compare_seeds_need_the_vectors_of_the_local_index`,
  `test_pool_vectors_reads_the_dense_vector_in_batches`, and
  `test_fact_and_filter_seeds_are_still_those_of_the_first_rule` (seeds pinned from the
  first rule's code). With the kept draws, the category key or the fact exclusion removed,
  its test fails. On the real pool (read-only, no request), the 30 fact and 8 filter seeds
  have exactly the ids of the stored first-round candidates, so the next round sends 22.

### 23. Unanswerable seeds: the paper's abstract and outline (`testset_unanswerable_v2`)
- **Decision.** The papers stay the ones the first rule chose (the next papers in the turns
  after the fact seeds). Instead of one section, the model sees the paper's title, its
  abstract and its outline (the distinct headings of its sections in reading order,
  without the reference list and acknowledgements), and is asked for a question that a
  researcher interested in the topic would ask about this paper, about one specific detail
  that neither the abstract nor any heading suggests the paper reports (its cost in money
  or energy, a benchmark, a language or domain, a method it does not name, how it was
  deployed), not answerable from general knowledge, not yes/no. Whether the paper's text
  really leaves it unanswered is checked in review, which shows the three closest passages
  of the same paper (a local BM25 search, decision 8). The seed chunk is the abstract
  (shown in review); the question has no reference chunks. The headings are paper text:
  escaped, inside the paper's source block.
- **Alternatives.** One section with looser wording: the model still sees a fraction of
  the paper. The whole paper: 10-90k tokens a call, and a model reading all of it can still
  miss a passage, so the review check is needed anyway.
- **Tested by.** `test_unanswerable_seeds_are_the_next_papers_abstracts_with_their_outlines`
  (the first rule's papers, now with their abstracts),
  `test_the_outline_is_every_heading_once_in_reading_order`,
  `test_an_unanswerable_seed_shows_the_abstract_and_the_escaped_outline`.

### 24. The v2 prompts write before they judge; a second round sends only new seeds
- **Decision.** The v2 prompts return the question and the reference answer first and
  `usable` last (`DraftedQuestion`); the v1 prompts keep their shape
  (`GeneratedQuestion`), so a prompt version also fixes its shape (`OUTPUTS`). A candidate
  id includes its prompt version, so v2 seeds get new ids: the first round's candidates,
  its 22 unusable ones included, stay in the candidates file unchanged, and `generate`
  sends only the seeds the file does not hold. `generate --estimate` now counts only those
  seeds (it counted every seed, so a second round's estimate was the whole set's).
- **Alternative.** Delete the unusable candidates and generate again under the old ids:
  that loses the record of what was paid for and lets two prompts share one id.
- **Tested by.** `test_the_v2_prompts_ask_for_the_question_before_the_verdict`,
  `test_a_second_round_sends_only_the_new_seeds_with_their_own_shape` (the first round's
  lines unchanged byte for byte; only unanswerable and compare seeds sent, with the v2
  shape), and on a scratch database
  `test_the_estimate_reads_the_local_stores_and_counts_only_new_seeds`.

### 25. The candidates are reviewed by an AI reviewer, not by the owner (a deviation from the plan)
- **Decision.** The plan has the owner review every generated question; the owner decided
  not to, and that the review is done by an AI reviewer instead. Written down before the
  first accept or reject, without any retrieval or evaluation result:
  1. One fresh subagent (a general-purpose Claude Code agent; reviewer name
     `claude-subagent`) reviews every pending candidate. It did not configure the
     generator, so the agent that wrote the seed rules and prompts does not grade their
     output itself. It gets only these rules and, per candidate, its id, type, question,
     reference answer and the seed text exactly as `make_testset review` shows it (an
     unanswerable candidate with the paper's abstract and the three closest passages of
     the same paper, decision 8). It never gets retrieval or evaluation results or golden
     questions, and it is told to treat the paper text as data.
  2. Rules: **fact**: the reference answer is stated in the seed text, and the question is
     understandable alone (no "this paper", "the above"); **filter**: the filter is real
     (a date or a category of the paper) and the question needs it; **compare**: both
     papers are needed, and the question cannot be answered from one of them; **unanswerable**:
     the answer is absent from the paper's text, including the same-paper passages shown
     (a passage that contains the answer means reject). In every type: the reference
     answer must not contradict the seed text; a near-duplicate of a candidate accepted
     earlier in the list is rejected; if unsure, reject. The aim is about 30 accepted with
     a mix of types and at least 15 (the smoke set), but no question is accepted to reach a
     count.
  3. A candidate the review tool warns about for word overlap of 0.6 or more with a
     golden question is rejected by rule, without showing the golden question to the
     reviewer. No generated question may sit on a golden paper (the pool excludes them,
     decision 7; checked again after the review).
  4. The decisions are recorded through the review tool's own code (`review` with
     scripted answers in the reviewer's order), so ids are numbered and every decision is
     saved as for a person, and no id is deleted or reused. A rejection keeps the
     reviewer's reason, prefixed `claude-subagent:`. The candidates file and
     `generated.jsonl` have no reviewer field, so the reviewer of every accepted
     question is recorded here and in the README.
  5. These questions are **AI-generated and AI-reviewed**; they must never be described as
     reviewed by a person.
- **Alternatives.** (a) Accept every candidate the generator marked usable: no review at
  all. (b) The agent that wrote the seed rules and prompts reviews: it would grade the
  output of its own configuration. (c) Wait for the owner: the owner decided against it.
- **Consequences.** The generated split, which decides every "best of the above" choice,
  rests on two models (the generator `gpt-4o-mini` and the reviewer); a wrong question that
  both let through can move a selection. The held-out golden numbers are reported next to
  every choice and never used to make it.
- **Tested by.** The review tool's tests (decision 8: numbering, saving after every
  decision, no deleted or reused ids, a hand-edited file refused). The decisions and the
  reasons are in the candidates file; the counts per type are in the part 3 report.

### Measured: the ablations on real data (part 3)
Run at commit 39cd4c5 (the code of decisions 1-25) on the corpus of that day: 8,912 points in
the app's collection, 33 full-text papers (30 of them added for the test set; the 3 golden
papers before). Questions: the 20 held-out golden questions (17 scored, 3 unanswerable) and the
27 generated ones (21 fact, 2 filter, 2 compare, 2 unanswerable; AI-generated and AI-reviewed,
decision 25). Judge and generator `gpt-4o-mini`, temperature 0; every run is stored in
eval_runs with its configuration, commit and per-question rows. Steps in order, each on the
stored winner of the step before (suites 8956c062dd89, 4a883a841610, 9bd752f18578,
97e7370ca00c, ee2fcb8199e8). Retrieval metrics on 10 results; R@8, MRR and nDCG@10 count
reference chunk ids and are left out for another chunking (decision 14); `runs` = the
eval_runs ids of the retrieval and the answers part.

**Generated questions (27; the rule chose on these, 25 paired per comparison):**

| Configuration | runs | R@8 | MRR | nDCG@10 | cov@8 | ctx recall | faithfulness | relevancy | ctx precision | citation acc | refusal acc | false refusal | p50 ms | $/answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 dense only | 23/24 | 0.540 | 0.513 | 0.510 | 0.550 | 0.667 | 0.835 | 0.673 | 0.683 | 0.822 | 0.500 | 0.160 | 5317 | 0.000793 |
| 2 BM25 only | 25/26 | 0.620 | 0.613 | 0.612 | 0.640 | 0.767 | 0.872 | 0.745 | 0.779 | 0.862 | 0.500 | 0.120 | 4694 | 0.000804 |
| 3 hybrid (RRF) | 27/28 | 0.660 | 0.627 | 0.625 | 0.682 | 0.807 | 0.846 | 0.694 | 0.767 | 0.912 | 1.000 | 0.040 | 5527 | 0.000843 |
| 4 hybrid + bge-reranker-base | 29/30 | 0.900 | 0.793 | 0.816 | 0.914 | 0.987 | 0.848 | 0.718 | 0.922 | 0.821 | 0.500 | 0.040 | 9719 | 0.000842 |
| 5 hybrid + MiniLM-L-12 reranker | 31/32 | 0.880 | 0.760 | 0.790 | 0.889 | 0.973 | 0.859 | 0.763 | 0.863 | 0.868 | 1.000 | 0.080 | 7469 | 0.000815 |
| 6 on 4, with query rewriting | 34/35 | 0.900 | 0.793 | 0.816 | 0.914 | 0.987 | 0.882 | 0.724 | 0.919 | 0.829 | 0.500 | 0.040 | 10442 | 0.000851 |
| 6 on 4, without query rewriting | 36/37 | 0.900 | 0.793 | 0.816 | 0.914 | 0.987 | 0.865 | 0.740 | 0.919 | 0.900 | 0.500 | 0.040 | 9724 | 0.000792 |
| 7 on 6, chunks of 256 | 39/40 | n/a | n/a | n/a | 0.682 | 0.987 | 0.893 | 0.731 | 0.865 | 0.808 | 0.500 | 0.040 | 9818 | 0.000673 |
| 7 on 6, chunks of 512 | 41/42 | 0.900 | 0.793 | 0.816 | 0.914 | 0.987 | 0.846 | 0.721 | 0.922 | 0.801 | 0.500 | 0.040 | 10594 | 0.000848 |
| 7 on 6, chunks of 1024 | 43/44 | n/a | n/a | n/a | 0.893 | 0.960 | 0.923 | 0.729 | 0.870 | 0.809 | 0.500 | 0.040 | 10846 | 0.000979 |
| 8 on 7, 3-small at 1536 | 46/47 | 0.900 | 0.793 | 0.816 | 0.914 | 0.987 | 0.863 | 0.726 | 0.919 | 0.811 | 0.500 | 0.040 | 10647 | 0.000838 |
| 8 on 7, 3-large shortened to 1536 | 48/49 | 0.860 | 0.773 | 0.791 | 0.883 | 0.947 | 0.825 | 0.731 | 0.878 | 0.874 | 0.500 | 0.080 | 11305 | 0.000835 |
| 8 on 7, 3-large at 3072 | 50/51 | 0.860 | 0.773 | 0.791 | 0.883 | 0.947 | 0.884 | 0.722 | 0.879 | 0.844 | 0.500 | 0.080 | 10745 | 0.000845 |

**Held-out golden questions (20; reported, never used to choose):**

| Configuration | R@8 | MRR | nDCG@10 | cov@8 | ctx recall | faithfulness | citation acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 dense only | 0.824 | 0.667 | 0.703 | 0.824 | 0.882 | 0.911 | 0.846 |
| 2 BM25 only | 0.912 | 0.779 | 0.796 | 0.919 | 0.882 | 0.833 | 0.784 |
| 3 hybrid (RRF) | 0.912 | 0.775 | 0.794 | 0.919 | 0.882 | 0.844 | 0.784 |
| 4 hybrid + bge-reranker-base | 0.794 | 0.667 | 0.685 | 0.801 | 0.882 | 0.832 | 0.748 |
| 5 hybrid + MiniLM-L-12 reranker | 0.912 | 0.716 | 0.759 | 0.919 | 0.941 | 0.849 | 0.632 |
| 6 on 4, with query rewriting | 0.794 | 0.667 | 0.685 | 0.801 | 0.882 | 0.819 | 0.711 |
| 6 on 4, without query rewriting | 0.794 | 0.667 | 0.685 | 0.801 | 0.882 | 0.785 | 0.727 |
| 7 on 6, chunks of 256 | n/a | n/a | n/a | 0.788 | 0.882 | 0.761 | 0.690 |
| 7 on 6, chunks of 512 | 0.794 | 0.667 | 0.685 | 0.801 | 0.882 | 0.831 | 0.725 |
| 7 on 6, chunks of 1024 | n/a | n/a | n/a | 0.801 | 0.882 | 0.856 | 0.739 |
| 8 on 7, 3-small at 1536 | 0.794 | 0.667 | 0.685 | 0.801 | 0.824 | 0.879 | 0.762 |
| 8 on 7, 3-large shortened to 1536 | 0.794 | 0.667 | 0.685 | 0.801 | 0.882 | 0.827 | 0.731 |
| 8 on 7, 3-large at 3072 | 0.794 | 0.667 | 0.685 | 0.801 | 0.882 | 0.826 | 0.725 |

**What the rule chose** (decision 13: context recall, paired bootstrap at 1 - 0.05/m for m challengers:
98.75 % for four, 97.5 % for two, 95 % for one; faithfulness as the guard; selection rows
33, 38, 45, 52):
- Ablations 1-5: **hybrid + `bge-reranker-base` (4)** replaces hybrid (3): context recall
  +0.18 [0.02, 0.38], faithfulness -0.011 [-0.148, 0.122] (not worse). The MiniLM reranker
  (5) was also better (+0.167 [2.2e-18, 0.373]: its lower bound is 0 up to rounding, so
  "better" rests on float noise) but by a smaller mean; dense (1) and BM25 (2) were not better.
- Ablation 6: query rewriting stays: without it the paired difference was 0.0 [0, 0].
- Ablation 7: 512 tokens stays: 256 gave 0.0 [0, 0], 1024 gave -0.027 [-0.133, 0.040].
  (Context recall favours larger chunks by construction; it did not here.)
- Ablation 8: `text-embedding-3-small` at 1536 stays: the large model gave -0.040
  [-0.160, 0.000] shortened to 1536 and at 3072 alike.
So the configuration chosen on generated questions is hybrid + `bge-reranker-base`, query
rewriting on, 512-token chunks, `text-embedding-3-small` at 1536: the app's defaults plus the
reranker, which the app has off (`rerank_enabled`). Turning it on is the owner's decision:
on this machine its median rerank time was 5.2 s per question and its process peaked at
about 2.1 GB (below).

**The reranker comparison moved from M4** (27 generated questions, each variant in its own
process, on configuration 4; selection row 67; bar: at least 20 candidates, median rerank
time per question at most 5 s, peak memory at most 2,048 MiB, quality kept):

| Variant | runs | R@8 | cov@8 | ctx recall | faithfulness | median rerank s | peak MiB | meets the bar |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference (bge, default threads, arena on, 50 candidates) | 53/54 | 0.900 | 0.914 | 0.987 | 0.835 | 5.24 | 2129.6 | no (time, memory) |
| 4 threads | 55 | 0.900 | 0.914 | - | - | 5.30 | 2439.2 | no (time, memory) |
| 2 threads | 56 | 0.900 | 0.914 | - | - | 7.35 | 2576.0 | no (time, memory) |
| 1 thread | 57 | 0.900 | 0.914 | - | - | 7.60 | 2465.7 | no (time, memory) |
| arena off | 58 | 0.900 | 0.914 | - | - | 6.41 | 2861.7 | no (time, memory) |
| 30 candidates | 59/60 | 0.860 | 0.874 | 0.947 | 0.868 | 3.90 | 2822.0 | no (memory) |
| 20 candidates | 61/62 | 0.820 | 0.834 | 0.867 | 0.891 | 1.92 | 2822.0 | no (memory) |
| 10 candidates | 63/64 | 0.780 | 0.800 | 0.907 | 0.892 | 0.93 | 2724.2 | no (10 < 20, memory) |
| MiniLM-L-12 (`Xenova/ms-marco-MiniLM-L-12-v2`) | 65/66 | 0.880 | 0.889 | 0.933 | 0.889 | 2.33 | 1500.5 | **yes** |

Only the small model meets the owner's bar; it is the recommendation. Thread and arena
settings change speed and memory only (identical retrieval lists), and on this machine none
made `bge-reranker-base` fit. Changing the app's reranker or `rerank_enabled` stays the
owner's decision.

**How far to trust these numbers.**
- The same configuration ran three times in three suites (eval_runs 35, 42 and 47: 4 with
  rewriting, 512 tokens, the small model): context recall 0.987 / 0.987 / 0.987 on generated
  questions, faithfulness 0.882 / 0.846 / 0.863. Retrieval parts are identical run to run
  (cached question vectors); answer metrics move by about 0.02-0.04 between identical runs
  (the model, the live query embeddings). Differences of that size in the tables are noise.
- 27 generated questions, 25 paired per comparison: one question moves a mean by 0.04. Only
  the reranker's gain cleared the corrected interval.
- The held-out split disagrees on the reranker: on the golden questions configuration 4 has
  R@8 0.794 against hybrid's 0.912 (as in the first measurement, M3), while on generated
  questions it has 0.900 against 0.660. A possible reason (not tested): the golden questions
  come from three papers whose reference passages the hybrid search already ranks high, the
  generated ones from 30 other, longer papers. Nothing was chosen on the held-out numbers.
- The generated questions were written and reviewed by models (decision 25); refusal
  accuracy on them rests on 2 unanswerable questions.

**Spend.** The smoke run and all of D cost $2.682941 (llm_usage), against $3.16 estimated and
a $5.00 ceiling: smoke $0.035, ablations 1-5 $0.611, 6 $0.247, 7 $0.373 (+ indexes $0.032),
8 $0.375 (+ indexes $0.701), reranker comparison $0.308. The first `text-embedding-3-large`
index build stopped at its own guard cap after 8,512 of 8,912 points (the guard counted 143
attempts for 133 billed requests, likely rate-limit retries) and was resumed (decision 18).

### Review before merge (part 2)
A fresh reviewer, given only the plan's evaluation text and the diff, found: a variant
index that was only partly built (a capped build) or older than the corpus would still be
searched and the run stored as complete; a plan stage that failed and fell back to the
default plan was stored as a complete run of the configuration under test; workflow-level
concurrency let any pull request event (another label) cancel a paid run; the selection
could compare runs of different suites (other bases, corpora); the reranker
recommendation could rest on held-out timings with no generated question; a chunk on
another chunking was badged as the reference; a crash outside the known errors left no
report; the stale-encoder observation could not fail; four challengers at 95 % each
without correction; estimates that priced large-model vectors at the small model's price
and did not show the guard's bound; an undocumented corpus revision rule. Each is fixed
above, with a test that fails without the fix. It also noted that the rule-10 replay
(decision 17) read the real stores; it was read-only (no write, no network) and is
reported to the owner as such. Not changed: `make eval-retrieval EMBED=1` (an M3 command)
embeds without a cap; it embeds only missing golden-question vectors, a fraction of a
cent, and stays behind its own ask-first rule.

### Review before merge (part 1)
An independent review of this change (a fresh reviewer with only the plan's evaluation text
and the diff) found, among others: filter seeds that ignored the golden exclusion, a review
run with a second candidates file that could delete accepted questions and reuse their ids,
a noncommittal answer scored "undefined" instead of 0 when no question was written back,
answers without any citation invisible to citation accuracy, billed judge failures missing
from the judge's cost, a paid run that kept nothing until its end and paid on through a
broken judge, the relevancy yardstick moving with the retrieval embedding model, and
claims here and in the README that were stronger than the code. Each is fixed above, with
a test that fails without the fix.

## Consequences
- Equal fused (or single-list) scores no longer depend on the order Qdrant returns them
  in, at the cut included; the retrieval evaluation, which reads cached question
  vectors, now gives identical lists on identical runs. Two cases remain outside this
  rule. (a) Inside one prefetch list, chunks with exactly equal raw scores (the same
  text twice, identical vectors) get their positions from Qdrant, and RRF turns
  positions into different fused scores, so their order can still change; the same holds
  at a prefetch's cut. (b) A live query is embedded again on every call, and the
  embeddings API can return slightly different vectors for the same text.
- Search sends one more point per query (`limit + 1`) and, on a tie at the cut, a second
  Qdrant request. No extra OpenAI request.
- A paper can now give up to 4 chunks to a result list (3 of its own text and 1 analysis
  chunk) instead of 3.
- The chat's own merge of the question and rephrasing lists already broke its ties by first
  appearance; its inputs are now in a fixed order, so the merge is repeatable too.
- An answer run costs money: about $0.004 per question with gpt-4o-mini as chat and judge
  model (one chat turn of about $0.001, measured from earlier turns, plus up to six judge
  calls and one embeddings request). `--estimate` prints requests and cost first; its
  bound uses the largest measured turn, so it is not a hard limit.
- The judge is a model: its verdicts can be wrong, and a model alias such as `gpt-4o-mini`
  can change behind the same name. The model, temperature and prompt versions are stored
  on every run, so runs judged differently are never compared silently. A dated model
  snapshot can be set in `EVAL_MODEL`.
- The judge and the generator use the same model family as the chat, which can favour
  answers written in its own style. A cross-check with the `ragas` library or another
  judge model is possible later.
- The generated questions all come from full-text papers (filter questions use those
  papers' abstracts), so they measure answers about full-text papers; questions about
  abstract-only papers come only from the golden set. A filter question's only reference
  chunk is the paper's abstract, so retrieval metrics count the paper's own sections as
  misses for it, as for the golden filter and find-papers questions.
- The retrieval runner (`make eval-retrieval`) still reads the golden file only; the full
  question set is for the ablation runner of a later step.

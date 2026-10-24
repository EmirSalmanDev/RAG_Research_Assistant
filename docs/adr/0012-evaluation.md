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
  plan's number) and lists papers to add.
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

### Review before merge
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

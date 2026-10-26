# 0016: Paper analyses go through the OpenAI Batch API

## Context
Each full-text paper gets a structured analysis (TL;DR, contributions, method, results,
limitations) that its page shows and that is indexed as extra chunks. Nobody waits for an
analysis: it is written once per paper, in the background. The OpenAI Batch API takes a file of
requests and returns the results within 24 hours at half the price of the same requests made
one at a time. Interactive features (chat, compare, review) need an answer in seconds and use
ordinary requests. The details of the implementation are in 0008.

## Decision
- **Analyses use the Batch API; nothing interactive does.** `app/llm/batch.py` selects
  full-text papers without an analysis of the current prompt version (at most
  `analysis_max_papers`, 20, newest first), writes one JSONL file with one request per paper,
  uploads it and creates one batch. Each request asks for a strict JSON schema, with the paper
  inside a delimited data block that the prompt tells the model to treat as data, never as
  instructions.
- **Creating the batch is sent once and never retried.** It is the request that spends money.
  Our own batch row and one item per paper are committed first. If the outcome is unknown (a
  timeout, a dropped connection, a 5xx) the row stays `created`, its papers stay queued, the
  command prints the row id and stops; a person checks OpenAI's batch list. A clear refusal (a
  4xx) frees the papers.
- **Collecting is separate and cheap.** `analysis.collect` runs every 30 minutes; with no open
  batch it makes no request and writes nothing. For a finished batch it validates each answer
  against the schema, and per paper in one transaction stores the analysis, its chunks and a
  `llm_usage` row at the Batch price, then indexes the chunks. A refused or cut-off answer is
  a failed item, and its usage is still logged because OpenAI billed it.
- **The daily submission is off** (`analysis_auto_submit = false`). A started Beat must not
  spend money unasked. `make analyze-submit DRY_RUN=1` prints the papers, the tokens and a
  cost bound without any request; `make analyze-submit IDS="..."` submits.
- The budget guard (0013) stops both the submission and the collection while the monthly cap
  is reached.

## Alternatives considered
- **Ordinary requests at ingestion.** Twice the price, and the single worker would wait on up to 20
  long requests, one after the other, inside the nightly run.
- **No analyses.** The paper page and the compare table would show less; the abstract and the
  full text are still searchable.
- **Retry the batch creation on a timeout.** Safe for most calls, not for this one: a second
  batch would bill the same papers twice. A rare manual check is cheaper than a double charge.

## Consequences
- Results arrive up to 24 hours after a submission; a paper has no analysis until then.
- Cost: measured on a real batch of three analyses, $0.002462 against a bound of $0.003187
  computed before submitting (about $0.0008 a paper). At the caps a submission costs at most
  about $0.03 and about $0.80 a month. Batch prices are configured (`batch_price_in_per_1m`,
  `batch_price_out_per_1m`) at half the chat prices; if OpenAI changes them, the logged cost
  is wrong until the configuration is.
- Spend is written to `llm_usage` when results are collected, so the budget guard sees a
  batch's cost only then.
- Analyses are model output, not the paper's words. Pages and chat label them as generated
  analysis, and they are capped in the search results so they do not displace the paper's own
  text (0012, decision 2).
- The schema conversion uses an SDK-internal helper; a test pins its output shape, so an SDK
  upgrade that changes it fails the tests.

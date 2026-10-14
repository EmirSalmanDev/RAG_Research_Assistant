# 0008: Enrichment and paper analysis

## Context
Each paper page should show the paper's code, the Hugging Face models, datasets and
Spaces built on it, the papers it cites and that cite it, and a structured analysis
(TL;DR, contributions, method, results, limitations). The enrichment data comes from
three free APIs: GitHub, the Hugging Face Hub and Semantic Scholar (S2). One-paper
enrichment already existed inside `make ingest-one`; the daily chain had no enrichment
step. Analyses are not urgent, so they go through the OpenAI Batch API (half price,
results within 24 hours).

## Decision

**One module, two entry points** (`app/ingest/enrich.py`). `enrich_one()` runs every
source for one paper; `make ingest-one` and the new `make enrich-one` use it.
`enrich_daily()` is the daily chain's step, after the abstracts are indexed:
`fetch_arxiv -> fetch_hf_daily -> index_abstracts -> enrich_new -> select_fulltext`.

**Calls per source.** GitHub: only the repo already linked to the paper
(`papers.github_url`, from Hugging Face), one `GET /repos/{owner}/{repo}` per repo.
There is no search by title and no "official" flag. READMEs are not fetched, so
`repos.readme_summary` stays empty (a value written before this change is kept, not
shown). Hugging Face Hub: one call per kind (model, dataset, space), at most 100 each.
S2: the batch endpoint, one request per 500 papers.

**Which papers the daily step looks at.** No column marks a paper as enriched. The step
keeps a cursor in `ingest_runs` (source `enrich-daily`), like the arXiv step. Papers
added since the last run get S2 data, their linked repo and, if Hugging Face lists them
(`hf_upvotes` set), their Hub artifacts. Two cheap rechecks cover data that arrives
later: papers added within `enrich_recheck_days` (7) whose S2 data is missing
(`citation_count` NULL; S2 often lists a paper days after arXiv) go into the S2 batch
again, and papers with a GitHub link but no repo row get their repo fetched again.
Hub artifacts are not rechecked (three calls per paper). Each list is capped at
`enrich_max_papers` (500).

**Failures.** Every source is best-effort. A source that answers replaces that paper's
rows for it (stale rows are deleted); a source that fails leaves them as they are. A
rate-limit refusal (429, or GitHub's 403) stops that source for the rest of the run;
the next run picks the papers up through the rechecks above (S2, GitHub) or not at all
(Hub artifacts, until `make enrich-one`). Per-paper errors are counted in the run's
`counts`; they do not fail the run, so the cursor moves on.

**Deviation (accepted by the owner): "cited by" from held papers only.** The
data-source plan lists S2's citing papers. `citations.citing_id` references `papers`, so
a citing paper we do not hold cannot be stored there. The "cited by" list is read from
the references of held papers, and S2's citing-paper list is not fetched. The rejected
alternative was one extra S2 request per paper (`fields=citations.externalIds`) stored
in a table without a foreign key to `papers`.

**Related papers** (`GET /papers/{id}/related`, `app/features/related.py`). "Similar"
queries Qdrant with the id of the paper's abstract point, so Qdrant searches with that
stored vector (no embedding request), over other papers' abstract points only. "Cites"
and "cited by" read `citations`; a referenced paper we do not hold is listed by its
arXiv id without a title. If Qdrant fails, "similar" reports `unavailable` and the
citation lists are still returned.

**Paper analyses through the Batch API** (`app/llm/batch.py`). `submit()` picks
full-text papers (indexed, text stored) without an analysis from the current prompt
version, at most `analysis_max_papers` (20), newest first. It commits our own batch row
and one item per paper first, then uploads one JSONL file (one chat request per paper)
and creates the batch; a crash in between leaves a 'created' row that the next
`submit()` finishes rather than repeats. Each request sends the `analyze_v1` system
prompt, the paper inside one `<source>` block (cut to `analysis_max_input_tokens`,
12,000, and marked when cut), a strict JSON schema (the SDK's own conversion of a
pydantic model, as `chat.completions.parse` sends it) and `analysis_max_output_tokens`
(1,200). `collect()` checks open batches; for an ended one it validates each answer
against the schema and, per paper in one transaction, stores the analysis, its chunk
rows and one `llm_usage` row, then indexes the chunks and marks the batch 'collected'.
A refused or cut-off answer is a failed item but its usage is still logged (it was
billed). A paper whose submission failed is tried again, at most
`analysis_max_attempts` (2) times per prompt version.

**Schema additions (migration 0004; deviations from the plan's data model, accepted by
the owner).** `analysis_batches` (one row per Batch job: OpenAI batch and file ids, our
or OpenAI's status, model, prompt version, request count, timestamps, errors);
`analysis_batch_items` (one row per paper and batch; a paper's queued batch is its
'pending' item, and a partial unique index allows one pending item per paper);
`paper_analyses.prompt_version` and `paper_analyses.batch_id`. The index
`ix_paper_analyses_batch_id` is a further non-plan addition: the repo indexes every
foreign key (a rule its migration test checks). **One analysis per paper**:
`paper_analyses.arxiv_id` is unique, and an analysis is replaced when a newer
`prompt_version` analyses the paper again.

**Analysis chunks.** Each analysis becomes up to four `analysis` chunks with fixed ords
from 10000: Summary (TL;DR and contributions; the TL;DR alone is too short to be a
useful search hit), Method, Results and Limitations. A new analysis overwrites them;
an empty section's old chunk is deleted. They are indexed by `upsert_analysis_chunks`,
which, like `upsert_paper_chunks`, embeds only points whose text changed and deletes
stale points of its own chunk type only.

**Cost and schedule.** `llm_usage.cost_usd` uses `batch_price_in_per_1m` and
`batch_price_out_per_1m` (0.075 and 0.30 USD, half the chat prices) with the token
counts each answer reports. At the caps a submission costs at most about $0.03 (20
papers, up to 12,000 input and 1,200 output tokens each), about $0.80 a month.
Beat submits daily at 09:00 only when `analysis_auto_submit` is on (off by default: a
started Beat must not spend money unasked) and collects every 30 minutes (no API call
and no run record while no batch is open). `make analyze-submit DRY_RUN=1` prints the
selected papers and the cost bound without any API call.

## Consequences
- The daily step makes about one S2 request, one GitHub call per linked new paper and
  three Hub calls per new paper that Hugging Face lists; all free.
- Without `GITHUB_TOKEN`, GitHub allows 60 calls an hour; a day with more linked papers
  stops GitHub early and the rest follow on the next runs (7-day recheck).
- The 6,653 papers indexed before this step are not enriched by it (their `created_at`
  is older than the first cursor). Enriching them is a separate backfill decision.
- "Cited by" stays short for a recent corpus: most citing papers are newer than the
  papers we hold, or not held at all.
- An analysis describes the paper text it was given: a text cut at 12,000 tokens can
  miss late results, and the prompt asks the model not to guess them.
- The schema conversion uses an SDK-internal helper (`type_to_response_format_param`);
  a test pins its output shape, so an SDK upgrade that changes it fails the tests.

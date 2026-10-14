# 0008: Enrichment and paper analysis

## Context
Each paper page should show the paper's code, the Hugging Face models, datasets and
Spaces built on it, and the papers it cites and that cite it. The data comes from three
free APIs: GitHub, the Hugging Face Hub and Semantic Scholar (S2). One-paper enrichment
already existed inside `make ingest-one`; the daily chain had no enrichment step.

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

**DEVIATION PROPOSAL (pending the owner's decision): "cited by" from held papers only.**
The data-source plan lists S2's citing papers. `citations.citing_id` references
`papers`, so a citing paper we do not hold cannot be stored there. As implemented, the
"cited by" list is read from the references of held papers, and S2's citing-paper list
is not fetched. The alternative keeps the plan: one extra S2 request per paper
(`fields=citations.externalIds`, up to 1000 citing ids) stored in a table without a
foreign key to `papers`. Until the owner decides, the code stays as implemented.

## Consequences
- The daily step makes about one S2 request, one GitHub call per linked new paper and
  three Hub calls per new paper that Hugging Face lists; all free.
- Without `GITHUB_TOKEN`, GitHub allows 60 calls an hour; a day with more linked papers
  stops GitHub early and the rest follow on the next runs (7-day recheck).
- The 6,653 papers indexed before this step are not enriched by it (their `created_at`
  is older than the first cursor). Enriching them is a separate backfill decision.
- With the held-papers-only proposal, "cited by" stays short for a recent corpus: most
  citing papers are newer than the papers we hold, or not held at all.

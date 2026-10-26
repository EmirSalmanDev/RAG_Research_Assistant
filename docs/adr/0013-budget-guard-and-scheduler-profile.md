# 0013: The monthly budget guard, and Beat in a compose profile

## Context
The settings page has stored a monthly budget cap since the digest milestone (0009) and
shows it next to this month's spend, but nothing used it. Meanwhile two daily runs started
without anyone asking for them (2026-10-02 and 2026-10-04): each time the stack was
restarted, Beat found the 07:00 run overdue and sent it at once, and the worker then called
arXiv, Hugging Face and, on the first occasion, OpenAI (11 embedding requests, $0.0058).
Two separate protections are needed: a limit on what unattended jobs may spend in a month,
and a way to start the stack without starting the scheduler.

## Decision

**What the guard pauses (`app/features/budget.py`).** The background tasks that spend money
check the budget before they do any paid work and skip when it is reached:

| Task | When paused |
| --- | --- |
| `ingest.index_abstracts` | no embeddings; the run is recorded as skipped |
| `ingest.embed_paper` | no embeddings; the paper keeps status `fetched` |
| `ingest.select_fulltext` | starts no full-text chain |
| `digest.score_relevance` | an edited interest is not embedded; scoring with the stored vectors still runs (it costs nothing) |
| `analysis.submit` | no Batch job is created, whatever `analysis_auto_submit` says |
| `analysis.collect` | open batches are left open (indexing the results embeds); no run row is written, because the task runs every 30 minutes |

A skipped run is recorded in `ingest_runs` with `skipped: "budget cap reached"`, is not
retried, and is not an error. Everything that costs nothing keeps running: fetching from
arXiv, Hugging Face, GitHub and Semantic Scholar, cleaning and chunking, building the digest
from stored vectors.

**When it pauses.** `paused` is true when a cap is stored and this month's spend is at or above
it. The spend is the sum of `llm_usage.cost_usd` since midnight of the first day of the month in
the digest time zone (`app_settings.month_to_date_spend`), so every model call is counted
the same way as on the settings page. No cap stored means no guard; a cap of 0 pauses
everything above. The check reads the database each time a task starts, so raising the cap or
the arrival of a new month resumes the jobs without a restart.

**Nothing is lost by pausing.** Abstracts that were not embedded are found again by the next
`index_abstracts`. A paper whose text is stored but not indexed (`fetched`) was invisible to the
selection, which skips papers that have text; `select_fulltext` now also restarts the clean,
chunk and embed steps for such papers (all three are idempotent), which also repairs a worker
that died between the steps. Open batches are collected once the guard allows it.

**What it does not pause.** Chat, search, compare, timeline and the literature review are
started by a person, who sees what they cost, and they work on stored data too. A cap that
locked the owner out of reading papers would be worse than the overspend it prevents. The
OpenAI account's own monthly limit (the README tells you to set one) is the hard backstop.
While the guard is paused, a banner on every page says so and links to the settings; `/health`
shows the state.

**Beat is a compose profile (`scheduler`).** `make up` starts the stores, the API and the
worker. `make up-scheduler` adds Beat. `make down` uses `--profile '*'`, which also stops the
services that are in a profile (Beat and the frontend); a plain `docker compose down` leaves
them running.

## Alternatives considered
- **Block interactive requests at the cap too.** Rejected for the reason above; it would also
  need an error path in every feature for a state the owner can change in one field.
- **Check the cap inside every OpenAI call.** It would stop a task midway and leave half-done
  work, and the egress guard already is the per-run hard cap for evaluations (0012). The
  tasks here are small and their steps idempotent, so deciding at the start is enough.
- **Stop the catch-up in code** (for example Beat's schedule file on a temporary filesystem,
  so a new container has no record of a missed run). It would fix a recreated container but
  not a restarted one, and it was not tested; the profile is visible in `docker compose ps`
  and in the command that starts it.

## Consequences
- **The guard can overshoot by one run and reacts late to Batch jobs.** The check is made
  before a task starts, not during it, so one run can cross the cap by its own cost (a day's
  abstracts, at most 20 papers' embeddings, one batch of at most about $0.03). A Batch job's
  cost is written to `llm_usage` when its results are collected, not when it is submitted, so
  a job submitted just under the cap is counted a day later.
- A started Beat no longer comes with `make up`. Someone who wants the nightly pipeline runs
  `make up-scheduler`, and should know it sends an overdue run on start.
- Tests: the state (no cap, exactly at the cap, a cap of 0, a new month, a raised cap), each
  gated task with every paid path replaced by a function that fails the test, the resume of
  stuck papers, and the compose, Makefile and workflow files by parsing them. Removing the
  guard fails 12 of the 14 budget tests.

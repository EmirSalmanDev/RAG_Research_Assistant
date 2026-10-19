# 0009: Daily digest and interests

## Context
The app should have a morning list of new papers, ranked against what the owner cares
about, with Save and Dismiss buttons, and the buttons should teach the ranking. The
inputs already exist: every new paper has an abstract vector in Qdrant, `hf_upvotes`,
categories and a nightly chain. This milestone adds interests, a relevance score, the
choice of papers that get full text, the digest itself, a feedback rule, and the
`/` and `/settings` pages.

Rules that shaped it: the pages and every CRUD endpoint make no paid call; the only paid
call in this feature is one embeddings request made by a background task (or by
`make digest`) when an interest is new or edited. Retrieval code (`app/rag/`, chunking,
embedding) is not touched, so no retrieval evaluation is owed. The scoring is not tuned
against the golden set.

## Decision

**Data (migration 0005, additive).** `interests` gets `weight` (float, default 1.0),
`embedding` (`real[]`, NULL until embedded), `embedding_model`, `embedding_hash` (hash of
the model and the embedded text), `is_active` and the two timestamps. `app_settings` is a
key to JSON table for the settings page. `digest_entries` has one row per paper per
digest: `digest_date` (the date in Europe/Istanbul), `arxiv_id`, `rank`, `score`,
`best_interest_id` (foreign key, SET NULL when the interest is deleted), `status` (`new`,
`saved`, `dismissed`, a CHECK), `weight_delta`, `actioned_at`; unique per
(`digest_date`, `arxiv_id`), both foreign keys indexed. Upgrade, downgrade and upgrade
again is tested on a scratch database; no existing row is dropped or rewritten.

**Interests and embeddings.** An interest is a name, a description, keywords, optional
categories and an active flag. Its embedding is made from `name`, `description` and
`keywords` by the scoring task, in one embeddings request for all interests that have no
vector for their current text (the hash stored beside the vector is compared with the
hash of the current text, so editing the text, and only the text, triggers a new
embedding). Creating or editing an interest through the API never embeds. A new interest
therefore ranks by keywords and categories alone until the next nightly run (the settings
page says "waiting for the next nightly embedding"). If the embedding request fails, the
scoring step records the error in its run and scores with the vectors it has.

**Scoring** (`app/features/relevance.py`, a pure function; constants in `config.py`).
For each active interest `i`: `s_i = w_i * (0.8 * cos + 0.2 * kw_i) + 0.05 * category_i`.
`cos` is the cosine between the paper's abstract vector (read from Qdrant by the abstract
point's deterministic id, no embedding request) and the interest vector; a missing vector
counts as 0. `kw_i` is the share of the interest's distinct keywords that occur in title
and abstract as whole words or phrases, case-insensitively. `category_i` is 1 when the
interest lists categories and the paper shares one, else 0 (no penalty). The paper's
`relevance_score` is the best `s_i` plus `0.05 * min(1, ln(1 + upvotes) / ln(101))`. With
no active interest the score is the upvote boost alone. The best interest (named on the
card, and the one Save and Dismiss move) is the one with the largest unweighted part
`0.8 * cos + 0.2 * kw`, ties to the smaller interest id; if that part is 0 and no category
matches, there is none. Papers are ordered by score, then newer first, then arXiv id, so a
tie is never decided by the order the database returns rows in. Tests cover ties, zero
interests, a missing vector, keyword edge cases (`rag` is not in `storage`, `c++`,
phrases), weights, the upvote cap, ordering stability under every input order, and
idempotence.

**The nightly chain and the digest.** The 07:00 chain is now `fetch_arxiv -> fetch_hf_daily
-> index_abstracts -> enrich_new -> score_relevance -> select_fulltext`. `score_relevance`
(`digest.score_relevance`, in `app/features/tasks.py`) embeds stale interests and stores
`relevance_score` for every paper published within `digest_window_days` (7).
`select_fulltext` keeps its library and requested papers and adds the top
`fulltext_top_n` (20) papers by `relevance_score` among those of the window that were not
dismissed. The top N are taken first and papers that already hold text are removed
afterwards, so each night only papers that newly entered the top N cost a download; a
paper marked `failed` or `unavailable` is not retried through this route. At 08:00 Beat
runs `digest.build`. It takes the papers of the window that were not dismissed and are not
in an earlier digest, scores them again (reading vectors from Qdrant is free, and a digest
never depends on a stale stored score), keeps the top `digest_size` (20) and writes the
entries. A second run for the same date changes nothing; `--rebuild` replaces only the
entries nobody acted on, and ranks are renumbered over saved, dismissed and new entries
together. `make digest` does the same by hand (`--dry-run` writes nothing and makes no
request, `--date`, `--rebuild`, `--no-embed`) and prints which papers the full-text
selection would pick. A trending badge shows at `hf_upvotes >= 10`.

**Feedback.** Save adds 0.10 to the weight of the entry's best interest, Dismiss subtracts
0.05, within 0.5 to 2.0, and the change takes effect at the next scoring run. The entry
records the change really made (`weight_delta`; 0 at a limit), weights are rounded to six
decimals so that a round trip returns exactly the starting value, and Undo gives back
exactly `weight_delta`. The state machine is `new -> saved | dismissed`; repeating an
action changes nothing; going from saved to dismissed (or back) undoes the first nudge
before applying the second; the entry row is locked while the weight moves. Save also adds
the paper to the library collection "Saved" (status `to_read`), which makes the next
nightly selection fetch its full text; Undo removes that row only if it is still
untouched. A dismissed paper never appears in a later digest.

**API.** `GET/POST /interests`, `PUT/DELETE /interests/{id}`, `GET/PUT /settings` (the
four settings plus this month's spend, summed from `llm_usage`), `GET /digest?date=` (the
requested day's digest, else the latest at or before it, else the newest; read only) and
`POST /digest/{arxiv_id}/save|dismiss|undo?date=`. Inputs are validated: categories must
look like an arXiv category, lengths and counts are capped. Categories saved on the
settings page replace the configured list in the arXiv fetch (backfill included); adding
one does not backfill older papers, and the page says so. The arXiv cursor logic is
unchanged.

**Pages.** `/` is the digest: a date header, cards (rank, date, categories, trending
badge, best interest, title, first three authors, summary, links to the paper page and to
the arXiv abstract page), Save, Dismiss and Undo with optimistic updates that roll back on
an error, loading, error and empty states, and a note asking for interests when there are
none. `/settings` lists interests with create, edit and delete (two-step) and holds the
digest size, full-text papers per night, categories and budget cap with the month-to-date
spend. The form rules are mirrored in plain TypeScript (`lib/settings/form.ts`) so they
are unit-tested without a browser.

## Deviations from the original design
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): the card's one-line summary
  is the stored analysis TL;DR when there is one, else the abstract's first sentence cut
  to 200 characters, computed in code. Batch-generated TL;DRs for every card are not part
  of this milestone.
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): the budget cap is stored and
  shown next to the month-to-date spend, but nothing is paused when it is reached; that is
  a later milestone.
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): the data model gains the
  interest columns, `app_settings` and `digest_entries` described above.
- DEVIATION PROPOSAL (needs owner ok): "top N by relevance" is read as the N best papers
  of the window, not N new papers per day. The result is at most N downloads in the first
  night and a few per night after, but a paper that stays in the top N for a week is
  fetched once.
- DEVIATION PROPOSAL (needs owner ok): the digest build re-scores its candidates instead
  of reading `papers.relevance_score`, which the score step still writes for the
  selection. The two agree unless interest weights change between 07:00 and 08:00.
- DEVIATION PROPOSAL (needs owner ok): a failed interest embedding does not stop the
  nightly chain (the error is recorded in the run); the plan's steps are otherwise
  unchanged.
- DEVIATION PROPOSAL (needs owner ok): the digest is also built when there are no
  interests (ordered by upvotes only), so the page is never empty for lack of setup.

## Consequences
- The literal "ready tomorrow morning" depends on Beat and the worker running at 07:00
  and 08:00. A missed run is not caught up; the page then shows the latest digest or "No
  digest yet", and `make digest` builds one by hand.
- The nightly selection can start up to `fulltext_top_n` full-text chains per night:
  arXiv or Hugging Face downloads (arXiv at one request per 3 s) and embeddings of the
  chunks. At the size of the three full-text papers held when this was written (about 36
  chunks each) that is roughly 18,000 embedding tokens per paper, about $0.0004, so
  about $0.01 per night at 20 papers. Setting "Full-text papers per night" to 0 switches
  the route off.
- The only paid call added is the interest embedding: a few hundred tokens, once per new
  or edited interest.
- Dismissing hides a paper from later digests only; it is still found by search and chat.
- Feedback moves one interest per click, the best one by similarity. A paper that matches
  an interest weakly nudges it as much as a strong match does.
- The month-to-date spend counts every row in `llm_usage`, including embeddings; the Batch
  analyses are counted when they are collected.

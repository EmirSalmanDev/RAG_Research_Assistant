# 0010: Paper comparison and topic timeline

## Context
Two features build on the index without adding anything to it: a table that compares 2 to
5 papers by dimension, with a cited cell per paper, and a timeline of the papers on a
topic with the key ones marked and a line on what each changed. Both cost money (an
embeddings request and a model call), so they run only when a button is pressed and they
store nothing. The retrieval code (`app/rag/search.py`, `rerank.py`, `retrieve.py`,
chunking and embedding) is not touched, so no retrieval evaluation is owed; the new
modules call the existing `search()`.

## Decision

**Where it lives.** `app/features/compare.py` and `app/features/timeline.py` (the logic),
`app/features/explore_cli.py` (`make compare`, `make timeline`), `app/api/explore.py`
(`POST /compare`, `POST /timeline`), prompts `compare_v1` and `timeline_v1`, a block of
settings in `config.py` (dimension questions, top-k, caps, weights; nothing is inline),
shared pieces in `deps.py`, `errors.py`, `spend.py`, `text.py` and `paper_facts.py`.
Nothing is persisted: no tables, no migration, no cache.

**Comparison.** The request is 2 to 5 distinct arXiv ids and a non-empty subset of six
dimensions (`task`, `method`, `data`, `metrics`, `compute`, `limitations`; default all). Count,
duplicates, unknown dimensions and unknown papers are refused (422, 404) before any paid
call.

1. *Retrieval.* Each dimension has one fixed question (`compare_dimension_queries`), the same
   for every paper, so all of them are embedded in ONE embeddings request. Then one hybrid
   query per (paper, dimension) runs against Qdrant with the filter `arxiv_id = paper`. The
   search function has no `chunk_type` filter, so 12 hits are fetched and only `abstract` and
   `section` chunks are kept (top 3): generated `analysis` chunks are model output, and a
   citation must point at the paper's own words. The embedded vectors reach `search()`
   through a small encoder that returns them, so the search code is unchanged. Rerank stays
   off; there is no planner (the filters are explicit inputs).
2. *Sources.* The chunks are de-duplicated into one list `[1]..[N]`, numbered paper by paper,
   each source showing paper id, title, section path and evidence level. If the input would
   exceed 30,000 tokens (cl100k_base, counted before the call) the per-cell top-k drops
   3 -> 2 -> 1; a single source longer than 800 tokens is cut and marked, which bounds the
   input at top-k 1.
3. *Fill.* ONE Structured Outputs call (`compare_v1`, temperature 0, the configured chat model):
   per paper and dimension a text of at most 40 words and the source numbers. Sources are only
   inside delimited blocks, escaped, and the prompt tells the model to ignore instructions in
   them. The model writes "Not stated in the sources." when nothing supports a cell.
4. *Validation in code.* This is the guarantee behind "every cell has a working citation": a
   cited number must exist AND belong to a source of the cell's own paper; invalid numbers
   are dropped; a cell with text but no valid citation becomes `not_stated` and shows no claim;
   a cell the model marked not stated stays so; a missing cell is `not_stated`; stray `[n]`
   markers in the text are removed; a text over 40 words is cut with an ellipsis. The counts
   of everything dropped or converted are in the response (`notes`).
5. *Response.* One row per paper (id, title, whether it has full text, link) with a cell per
   dimension (text, status `cited` or `not_stated`, citations with chunk id, evidence level
   and section), the numbered sources (with chunk id, title, date, abstract-page link),
   token and cost totals and the prompt version. 503 without an OpenAI key, 502 when OpenAI or
   Qdrant fails after the retries. Every request writes its own `llm_usage` row
   (`embed-compare`, `compare-fill`), so a failure after a billed call still shows the spend.

*Honest evidence.* A paper that has only its abstract indexed will often have `not_stated`
cells for data, compute and metrics. The table marks its column "abstract only", and so are
its citation chips; the sources panel says the claim rests on the abstract.

**Timeline.** `topic` (3 to 200 characters), optional dates and categories, `max_papers`
(default 30, at most 50), `key_papers` (default 8, at most 15).

1. *Search.* One embeddings request for the topic and one hybrid query (50 candidates per
   list, filters applied inside Qdrant), asked for the best 100 chunks and reduced to
   `abstract` chunks in code. Papers are taken in order of their best fused position;
   equal RRF scores are ordered by arXiv id first, because Qdrant returns ties in no fixed
   order. The list is cut to `max_papers`. A paper whose abstract chunk is not among the
   hits (only a section or analysis chunk matched) is not listed.
2. *Key papers.* A pure function: `topic_relevance = 1 - (rank - 1) / (n - 1)` over that order
   (1.0 when there is one paper); `citation_norm = ln(1 + c) / ln(1 + max c)` with an unknown
   count as 0 (0 when the maximum is 0); `key_score = 0.7 * topic_relevance + 0.3 *
   citation_norm`. Ties go to the higher relevance, then the earlier date, then the smaller
   arXiv id. "Relevance" is relevance to the topic; the interest-based `relevance_score` of
   the digest is not used.
3. *Lines.* ONE structured call (`timeline_v1`) gets the key papers in date order with their
   abstracts in delimited blocks and returns one line per paper: what it changed, at most
   30 words, only from its abstract, referring only to other listed papers. Code accepts a
   line only if there is exactly one non-empty line for the paper, within 30 words, without
   an arXiv id of a paper that is not a key paper. Anything else, and a failed call, falls
   back to the abstract's first sentence cut to 200 characters, with no second paid call;
   `line_source` says which it was and the page labels every line "from the abstract".
4. *Response.* Points (date, title, citation count, `topic_relevance`, `key_score`, `is_key`,
   line, abstract chunk id, link), the date range, notes (papers, key papers, how many
   citation counts are known, whether the range is narrow, fallback lines, whether the
   lines call failed), prompt version and cost.

**A free lookup.** `GET /papers?q=&limit=&has_fulltext=` (title substring, case-insensitive,
or an exact arXiv id, newest first, 1 to 50) feeds the picker. It reads Postgres only. It is
the collection root and does not shadow `GET /papers/{arxiv_id}`. `has_fulltext` means the
paper has `section` chunks, not that `fulltext_status` says so.

**Pages.** `/compare`: a picker (the lookup, a quick list of full-text papers, chips for the
picked ones, a limit of five), dimension toggles, a Compare button that is off outside 2 to 5
papers, loading, error and empty states, and the table (papers as columns, an "abstract only"
badge, citation chips that open the chat's sources panel with the chunk text from
`GET /chunks/{id}` and a link to the arXiv abstract page). Markdown export is a pure function
in the browser (download and copy): a table with `[n]` markers and a Sources list that links
abstract pages, never a PDF. `/timeline`: a topic form with an optional date range, category
and counts, a Run button, a plain SVG time axis whose unit follows the range (days, months or
years) with every paper as a dot and key papers larger, a list below with the one-line
summaries, links to the paper page and arXiv, and the notes line. Typing never makes a paid
call; the two buttons are the only way to one.

**Tracing.** Both features use the existing `Tracer`: a trace `compare` with spans
`compare-retrieve` and `compare-fill`, and a trace `timeline` with `timeline-search` and
`timeline-lines`. The Langfuse adapter now names the trace with the name it is given (chat
still passes `chat-turn`), so the chat turn's six spans are unchanged.

## Deviations from the original design
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): nothing is persisted. The original
  design's feature rows do not say where results live; there is no result table and no cache, so
  pressing the button twice costs twice.
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): the timeline is plain SVG, not
  Recharts. Recharts is not installed and a new dependency needs a network approval. A later
  milestone may add it.
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): "relevance" in "ranked by
  citations and relevance" means relevance to the topic, not the digest's interest score.
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): `GET /papers` is not in the original design;
  it is a free read endpoint for the picker, like the read endpoints of the paper page.
- DEVIATION PROPOSAL (needs owner ok): the original design's comparison sentence says "one retrieval per
  paper and dimension". The embeddings of the six dimension questions are one batched
  request, but the Qdrant queries are one per paper and dimension, as written.
- DEVIATION PROPOSAL (needs owner ok): the timeline's key-paper line falls back to the
  abstract's first sentence when the model's line fails a check. The original design says only that
  an LLM pass writes the line.
- The Langfuse adapter change (the trace name) touches existing code, though not the chat
  pipeline.

## Consequences
- A comparison costs about two requests and a fraction of a cent (the upper bound for the
  three full-text papers is printed by `make compare ... ESTIMATE=1` from the stored chunk
  sizes). A timeline costs less. Both are shown on the page after the call.
- What the table can say is limited by what is indexed: only three papers have full text, so
  most comparisons are mostly abstract-level, and the page says so instead of hiding it.
- The model may still cite the wrong source of the right paper. Code guarantees that a shown
  claim has a citation to a passage of its own paper, not that the passage supports the claim;
  the reader opens the chip to check, which is why every chip opens the passage.
- Citation counts are known for very few papers, so key papers are chosen mostly by their
  match to the topic until Semantic Scholar enrichment covers more of the corpus.
- Rule-10 retrieval evaluation is not owed: no file under `app/rag/search.py`, `rerank.py`,
  `retrieve.py`, `app/ingest/chunk.py` or `embed.py` changed. Nothing was tuned against the
  golden set.

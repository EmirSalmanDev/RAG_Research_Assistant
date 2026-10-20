# 0011: Library, BibTeX export, literature review and PDF upload

## Context
The library turns the index into a working tool for a literature review: named collections
of papers with a reading status and notes, a `.bib` file built from metadata, a related-work
draft with `[n]` citations, and a way to add a journal paper that is not on arXiv by uploading
its PDF. Two of these cost money (the draft: chat calls; the upload: embeddings), so they run
only when a button is pressed. Pages never call OpenAI on load, typing never starts a paid
call, and nothing from a review is stored.

The retrieval code (`app/rag/search.py`, `rerank.py`, `retrieve.py`, `app/ingest/chunk.py`,
`app/ingest/embed.py`) is not touched, so no retrieval evaluation is owed. The review uses no
retrieval and no embeddings: it reads Postgres only. (`app/rag/citations.py` changed only in
how it builds the arXiv link, see "Uploaded documents".)

## Decision

### Data (migration 0006, additive)
- New table `library_collections` (unique name, description, created_at). A collection exists
  when it has a row there OR when some `library_items` row names it, so the "Saved" collection
  that the digest's Save button fills needs no seeding and an empty collection can be created.
  Renaming updates both tables in one transaction; deleting removes the collection's items,
  never the papers.
- `library_items` keeps its primary key `(arxiv_id, collection)`: one item per paper and
  collection. Only an index on `collection` is added. Statuses `to_read`, `reading`, `read`,
  `archived` are validated by the API (a pydantic enum) with no database CHECK, so one can be
  added without a migration.
- DEVIATION PROPOSAL (needs owner ok): items are addressed by `arxiv_id` plus `?collection=`
  (`PATCH/DELETE /library/items/{arxiv_id}?collection=`), not by an `{id}`: the table has no
  surrogate id and none was added. The path accepts old-style ids with a slash.
- The digest's Save and Undo keep working unchanged (regression test).

### Library API (`app/api/library.py`, `app/features/library.py`)
`GET/POST /library/collections`, `PUT/DELETE /library/collections/{name}`, `GET
/library/items?collection=`, `POST /library/items`, `PATCH/DELETE /library/items/{arxiv_id}`,
`GET /library/bibtex?collection=`. All free, Postgres only, 404 / 409 / 422 with a readable
sentence. Adding a paper takes an arXiv id (with or without a version) or an arxiv.org link
and works only for a paper already in the database; an unknown one is a 404 whose message says
it has to be ingested first. DEVIATION PROPOSAL (pre-accepted by owner in this prompt): "Add by
arXiv id" for papers not in the database goes through the worker and is a later step.

### BibTeX (`app/features/bibtex.py`: pure functions, no model, no network)
- Key: the first author's last name, the year and the first significant title word, folded to
  lower-case ASCII (`vaswani2017attention`); a key already taken gets `a`, `b`, `c`, ... (the
  first paper keeps the plain key). Entry type `@misc`. The title gets a second pair of braces
  (capitals kept) and LaTeX special characters are escaped (math in a title becomes text, because
  a stray `$` breaks a document). Unicode letters stay in the entry; only the key is folded.
- arXiv papers: `eprint`, `archivePrefix = {arXiv}`, `primaryClass`, `url` (the abstract page).
  Uploads: `note = {Uploaded document}` and no `eprint` or `url`. No authors: `Anonymous`.
- One order everywhere (oldest first, ties by id), and the keys are computed over the WHOLE
  collection, so a review that uses only some papers still shows the keys of the full `.bib`.
  Tests check determinism, unicode, escaping, collisions, old-style ids, uploads and run a small
  validator (balanced braces, unique keys, required fields) over the output.

### Literature review (`POST /library/review`, `app/features/review.py`)
- Each paper is ONE numbered source `[1]..[N]`. Its evidence is its stored analysis
  (`paper_analyses`; the citation points at the paper's first `analysis` chunk, the summary) or,
  when it has none (or the analysis chunk is not indexed), its abstract chunk, flagged
  `abstract`. A paper with no chunk cannot be cited and is counted. Over `max_papers` (default
  30, cap 60) papers with an analysis are kept first, then the stable order.
- Up to 8 papers the writer reads them directly. Above that, groups of 8 (stable order) each get
  ONE structured map call (`review_map_v1`, short theme notes that keep the source numbers), then
  ONE writing call (`review_write_v1`) reads the notes. At most 9 chat calls for 60 papers.
  Nothing is persisted. Both calls are logged to `llm_usage` (`review-map`, `review-write`) and
  traced as spans `review-map` / `review-write` of a trace `review` (the existing `Tracer`
  interface expresses this; the chat pipeline's six spans are unchanged).
- Papers, analyses and notes appear only inside escaped, delimited blocks, and both prompts say
  to ignore instructions inside them (rule 8). The prompts restrict statements about limitations
  to the `limitations` field and about results to the `results` field, and say that a source with
  evidence `abstract` supports only what its abstract says.
- Code, not the model, decides what is shown (`validate_draft`): a source number must exist (and
  after the map step appear in a note); invalid numbers are dropped; at most 4 per sentence; a
  `claim` with no valid number is REMOVED and counted in `dropped_uncited`; a `transition` may be
  uncited but is treated as an uncited claim (and removed) when it names an author, says "et al."
  or holds a digit, and transitions are limited to 25 % of the kept sentences; a transition that
  carries a valid citation is a claim. An uncited claim is never shown.
- Everything that can be refused is checked before the first paid call: unknown collection (404),
  fewer than two citable papers, a focus over 200 characters or `max_papers` outside 1..60 (422),
  a missing key (503, before any request). 502 when the model fails after the retries or returns
  something unusable; what was billed is in `llm_usage`.
- Response: sections, paragraphs and sentences (kind, source numbers), references (n, arXiv id,
  BibTeX key, title, year, chunk id, section path, evidence `analysis`|`abstract`, link or null,
  how often cited), counts (papers, papers without analysis, omitted, uncitable, claims,
  transitions, dropped uncited, dropped transitions, invalid numbers, map calls, dropped notes),
  tokens, cost and prompt versions.
- Honest evidence. The draft rests on model-generated analyses and abstracts, not on full text,
  and the page and the markdown export say so. The guarantee is that every shown claim has at
  least one citation that exists. It is NOT that the cited text supports the claim: the M8 finding
  (a real citation for a claim the text does not make, for example a motivation shown as a
  limitation) applies here too and code cannot check it. The prompt rules reduce it; the reader
  has to check. A chip opens the paper's analysis summary chunk, which holds the TL;DR and the
  contributions: a sentence about method, results or limitations rests on a field of the
  analysis that the chunk does not show, so the paper page (linked) is where to check it.
- `make review COLLECTION=... ESTIMATE=1` prints an upper bound of the input tokens and the cost
  from the stored text (no network, no key; for the direct case it equals the call, for the map
  case it bounds the notes by the output cap). `BIB=1` prints the `.bib`.

### Uploaded documents and PDF upload (`app/ingest/upload.py`, `app/api/upload.py`)
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): an uploaded document is a `papers`
  row with the id `local-<first 12 hex of sha256 of the PDF bytes>` in the `arxiv_id` column
  (version 1, `fulltext_source` `upload`, status `indexed`, no categories, no GitHub link), not
  new columns. Every code path that assumes an arXiv id skips it: `enrich_one` makes no call,
  `plan_daily` and the full-text selection leave it out, `request_fulltext` does nothing, the
  digest window leaves it out. The arXiv link (`app/ids.py: abs_url`) is null for it, so
  `abs_url` is `string | null` in every response that has it (papers, search, digest, compare,
  timeline, the chat sources); the UI shows "uploaded document" and no link. Compare and the
  timeline may include an upload.
- `POST /library/upload?title=&collection=&year=&abstract=` takes the raw `application/pdf` body
  (no multipart, no new dependency). It checks the content type (415), title and year (422), the
  size (413, also from Content-Length before a byte is read; `upload_max_bytes`, default 25 MB),
  an empty body (422), the `%PDF-` magic bytes (415), the key (503) and that no other upload is
  running (429), all before parsing. The PDF is parsed in memory by the existing PDF-to-markdown,
  then the existing clean, chunk, embed and upsert functions run (chunk ids `uuid5(id, 1, ord)`).
  The PDF is never stored or served; only the parsed text is kept. A PDF with no text layer (a
  scan) is refused: there is no OCR. The abstract chunk is the abstract the uploader gave, else
  the first 1,500 characters of the body. Embeddings are logged as `embed-upload`.
- Idempotent: the same bytes give the same id; a second upload finds the paper, parses nothing,
  embeds nothing and changes nothing (the title sent the second time is ignored). An upload that
  stopped at the embeddings (Postgres already holds the text) is finished by uploading again.
  `DELETE /library/uploads/{id}` accepts only `local-` ids and removes the Qdrant points first,
  then the paper row (chunks, text, library items cascade).
- DEVIATION PROPOSAL (pre-accepted by owner in this prompt): parsing runs in the API process, not
  in the worker (the worker is not started in this project yet). Risk: PyMuPDF4LLM peaks at about
  430 MiB for a long paper, inside the API container that also holds the reranker when search
  runs with it; one upload at a time (a non-blocking lock, 429 otherwise) bounds it. Moving it to
  the worker is a later step.
- `make ingest-pdf FILE=... TITLE=... [ESTIMATE=1]` runs the same service function natively;
  `ESTIMATE=1` parses and chunks in memory and prints the tokens and a cost bound.
- DEVIATION PROPOSAL (needs owner ok): `frontend/next.config.ts` sets
  `experimental.proxyClientMaxBodySize: "30mb"`. Next's `/api` rewrite buffers a request body up
  to 10 MB and silently cuts the rest; the browser test (a 12 MB body through the rewrite) showed
  it, and a cut PDF would parse as garbage. The limit is kept above the API's 25 MB so the API is
  what refuses a big file.

### Frontend (`/library`)
Collections (create, rename, delete with a confirmation), a table of the papers (title, evidence
badge `full text` / `abstract only` / `uploaded`, status select, inline notes, move, remove), an
add-by-id box, the upload card (file chooser, title prefilled from the file name, size, the cost
note "about $0.0002 per 10 pages of embeddings", limit checked before a byte is sent), `.bib`
download and copy, and the review panel: collection select, optional focus, Generate review
(off below two papers, with its number of model calls), the draft with `[n]` chips labelled
"analysis" or "abstract" that open the existing sources panel, the references with their BibTeX
keys, "N claims removed because they had no valid citation", and a markdown export (a pure
function with a node test). The panel now says "Uploaded document" for a `local-` id and
explains a generated analysis chunk.

## Consequences
- The Done-when is read as: one collection yields a cited related-work draft and a `.bib` file
  in which every cited number resolves to a real chunk of the right paper and evidence level.
  It does not claim the draft is correct. The owner's graduation collection does not exist yet,
  so the evidence is a proof collection (see the report of the run).
- Only 3 papers have a generated analysis today, so most reviews rest on abstracts and will be
  thin. The prompts say what an abstract can support, but a draft over abstracts is a draft.
- Each click of Generate review repays the model calls: nothing is cached or stored.
- Only papers already in the database can be added by id. Adding any other arXiv paper needs the
  ingest path through the worker (a later step).
- An upload is parsed in the API process (memory risk above) and only text PDFs work.
- `abs_url` became nullable in the OpenAPI schema and the generated client; the places that
  show it already went through `arxivAbsUrl`, which accepts null.

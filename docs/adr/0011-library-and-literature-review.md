# 0011: Library, BibTeX export, literature review and PDF upload

## Context
The library turns the index into a working tool for a literature review: named collections of
papers with a reading status and notes, a `.bib` file built from metadata, a related-work
draft with `[n]` citations, and a way to add a journal paper that is not on arXiv by uploading
its PDF. Two of these cost money (the draft: chat calls; the upload: embeddings), so they run
only when a button is pressed. Pages never call OpenAI on load.

The retrieval code (`app/rag/search.py`, `rerank.py`, `retrieve.py`, `app/ingest/chunk.py`,
`app/ingest/embed.py`) is not touched, so no retrieval evaluation is owed. The review uses no
retrieval and no embeddings: it reads Postgres only.

## Decision

(Filled in as the stages land; see the sections below.)

## Consequences

(Filled in at the end.)

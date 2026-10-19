# 0009: Daily digest and interests

## Context
The app should have a morning list of new papers, ranked against what the owner cares
about, with Save and Dismiss buttons, and the buttons should teach the ranking. The
inputs already exist: every new paper has an abstract vector in Qdrant, `hf_upvotes`,
categories and a nightly chain. This milestone adds interests, a relevance score, the
choice of papers that get full text, the digest itself and a feedback rule.

Rules that shaped it: the page and every CRUD endpoint make no paid call (CLAUDE.md
rule 5 about spending; the only paid call in this feature is an embedding request made
by a background task), retrieval code is not touched (`app/rag/`, chunking and
embedding code stay as they are), and the scoring is never tuned against the golden set.

## Decision

(Filled in as the stages land; this file is the record of the choices below.)

- Interests, settings and digest tables (migration 0005, additive).
- Scoring function and constants.
- Candidate window, digest build and the nightly chain.
- Feedback rule.
- API and pages.
- Deviations from the plan text.

## Consequences

(Filled in at the end.)

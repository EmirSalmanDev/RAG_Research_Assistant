"""Paper ids: arXiv ids, and the ids of uploaded documents.

A paper that is not on arXiv (a journal paper uploaded as a PDF) gets the id
`local-<first 12 hex digits of the sha256 of the PDF bytes>` (docs/adr/0011). It lives in
the same `papers.arxiv_id` column as every other id, so each code path that assumes an arXiv
id (the arXiv link, the enrichment sources, the digest) has to recognise it and skip it.
This module is the one place that knows the prefix.
"""

ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
LOCAL_PREFIX = "local-"
LOCAL_HASH_CHARS = 12  # hex digits of the PDF's sha256 kept in the id


def is_local_id(arxiv_id: str) -> bool:
    """True for the id of an uploaded document (never an arXiv id)."""
    return arxiv_id.startswith(LOCAL_PREFIX)


def abs_url(arxiv_id: str) -> str | None:
    """The arXiv abstract page of a paper, or None for an uploaded document. Links always
    go to the abstract page, never to a PDF (we do not re-serve papers)."""
    return None if is_local_id(arxiv_id) else ARXIV_ABS_URL.format(arxiv_id=arxiv_id)

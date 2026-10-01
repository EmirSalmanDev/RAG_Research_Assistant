# 0004: The project is licensed AGPL-3.0

## Context
The PDF fallback of the full-text chain uses PyMuPDF4LLM, which depends on PyMuPDF and
PyMuPDF Layout. All three are dual-licensed: GNU AGPL-3.0 or a commercial Artifex
licence. The repository will be public and container images will be published, which
counts as distribution. Alternatives with permissive licences (pdfminer-based parsers)
produce plain text without reliable headings or tables, which weakens section-aware
chunking for papers that are only available as PDF.

## Decision
- Keep PyMuPDF4LLM for the PDF fallback and license this project under AGPL-3.0 when it
  becomes public (version 1.0). The LICENSE file is added together with the release work.
- Import PyMuPDF4LLM only inside the PDF conversion function, so only the process that
  parses a PDF loads it.
- Third-party data committed in the repo keeps its own licence, stated in the file
  (e.g. the BM25 stopword list, Apache-2.0).

## Consequences
- Anyone who distributes the project, or offers it as a network service with changes,
  must make the source available under AGPL-3.0.
- Replacing the PDF parser with a permissively licensed one later would allow a
  permissive licence again; the converter sits behind one function, `pdf_to_markdown()`.

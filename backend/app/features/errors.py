"""Errors of the compare and timeline features, and how the API reports them."""


class InvalidRequest(ValueError):
    """The request breaks a rule (count, duplicates, unknown dimension): HTTP 422.
    Always raised before any paid call."""


class UnknownPapers(LookupError):
    """Some requested arXiv ids are not in the database: HTTP 404, before any paid call."""

    def __init__(self, arxiv_ids: list[str]) -> None:
        super().__init__("unknown paper(s): " + ", ".join(arxiv_ids))
        self.arxiv_ids = arxiv_ids


class UpstreamError(RuntimeError):
    """OpenAI or Qdrant failed after the retries (or answered with something unusable):
    HTTP 502. Requests that were billed before the failure are in llm_usage anyway."""


class UnknownCollection(LookupError):
    """The library collection does not exist: HTTP 404, before any paid call."""

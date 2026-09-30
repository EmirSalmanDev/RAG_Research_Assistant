"""arXiv metadata client (abstracts/metadata only; we never fetch or re-serve PDFs).

How we stay inside arXiv's "1 request per 3 seconds, one connection at a time" rule
(CLAUDE.md rule 4) while using the `arxiv` package:

- `arxiv.Client(num_retries=0, delay_seconds=0)`: the package neither retries nor
  sleeps by itself. Retries are done by *our* tenacity layer (`call_with_retry`) and
  spacing by *our* Redis limiter, which also works across processes.
- The package normally walks all result pages inside one generator, firing
  back-to-back requests. We avoid that: we ask for exactly ONE page per call
  (`Search(max_results=offset+N)` + `client.results(search, offset=offset)` with
  `page_size=N` makes the package issue one HTTP request and stop), and do the paging
  loop ourselves, so every page goes through `limiter.slot()`.
- The package's `requests` session has no timeout, so we install one that does
  (a hung request would otherwise hold the limiter lock until its TTL).
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any

import arxiv
import requests
from pydantic import BaseModel

from app.sources.http import Limiter, RetryableError, call_with_retry
from app.sources.ratelimit import RedisRateLimiter, get_redis

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT_S = 20.0
MIN_INTERVAL_S = 3.0

_VERSION_RE = re.compile(r"^(?P<id>.+?)(?:v(?P<version>\d+))?$")


class ArxivPaper(BaseModel):
    arxiv_id: str  # without version, e.g. "1706.03762"
    version: int  # latest version number at fetch time
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published_at: datetime  # date of v1
    updated_at: datetime


class _TimeoutSession(requests.Session):
    """requests.Session that always sets a timeout (the arxiv package sets none)."""

    def request(  # type: ignore[override]
        self, method: str, url: str | bytes, **kwargs: Any
    ) -> requests.Response:
        kwargs.setdefault("timeout", REQUEST_TIMEOUT_S)
        return super().request(method, url, **kwargs)


def split_version(arxiv_id: str) -> tuple[str, int | None]:
    """'1706.03762v5' -> ('1706.03762', 5); '1706.03762' -> ('1706.03762', None)."""
    m = _VERSION_RE.match(arxiv_id.strip())
    assert m is not None  # the pattern matches any string
    version = m.group("version")
    return m.group("id"), int(version) if version else None


def _to_paper(result: arxiv.Result) -> ArxivPaper:
    arxiv_id, version = split_version(result.get_short_id())
    return ArxivPaper(
        arxiv_id=arxiv_id,
        version=version or 1,
        title=" ".join(result.title.split()),  # the feed hard-wraps long titles
        abstract=" ".join(result.summary.split()),
        authors=[a.name for a in result.authors],
        categories=list(result.categories),
        primary_category=result.primary_category,
        published_at=result.published,
        updated_at=result.updated,
    )


class ArxivClient:
    def __init__(
        self,
        limiter: Limiter | None = None,
        session: requests.Session | None = None,
        page_size: int = 50,
        sleep: Any = None,
    ) -> None:
        self._limiter: Limiter = limiter or RedisRateLimiter(
            get_redis(), "arxiv", min_interval_s=MIN_INTERVAL_S
        )
        self._page_size = page_size
        self._sleep = sleep
        self._client = arxiv.Client(page_size=page_size, delay_seconds=0.0, num_retries=0)
        # Private attribute of the package, replaced on purpose (see module docstring).
        self._client._session = session or _TimeoutSession()

    def _fetch_page(
        self, search_kwargs: dict[str, Any], offset: int, size: int
    ) -> list[arxiv.Result]:
        """Exactly one HTTP request (plus retries), each attempt inside a limiter slot."""

        def once() -> list[arxiv.Result]:
            self._client.page_size = size
            search = arxiv.Search(max_results=offset + size, **search_kwargs)
            try:
                with self._limiter.slot():
                    return list(self._client.results(search, offset=offset))
            except arxiv.HTTPError as err:
                if err.status in RETRYABLE_STATUS:
                    raise RetryableError(str(err), status=err.status) from err
                raise
            except (requests.ConnectionError, requests.Timeout) as err:
                raise RetryableError(f"network error: {err}") from err

        kwargs: dict[str, Any] = {} if self._sleep is None else {"sleep": self._sleep}
        return call_with_retry(once, **kwargs)

    def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        """Metadata for one paper id (with or without 'vN'). None if arXiv has no such id."""
        results = self._fetch_page({"id_list": [arxiv_id]}, offset=0, size=1)
        return _to_paper(results[0]) if results else None

    def new_papers(
        self, category: str, since: datetime, max_results: int = 100, until: datetime | None = None
    ) -> list[ArxivPaper]:
        """Papers first submitted in `category` between `since` and `until` (default now),
        newest first. Pages are fetched one at a time, each through the limiter."""
        until = until or datetime.now(UTC)
        fmt = "%Y%m%d%H%M"
        query = (
            f"cat:{category} AND submittedDate:"
            f"[{since.astimezone(UTC).strftime(fmt)} TO {until.astimezone(UTC).strftime(fmt)}]"
        )
        search_kwargs: dict[str, Any] = {
            "query": query,
            "sort_by": arxiv.SortCriterion.SubmittedDate,
            "sort_order": arxiv.SortOrder.Descending,
        }
        papers: list[ArxivPaper] = []
        while len(papers) < max_results:
            size = min(self._page_size, max_results - len(papers))
            page = self._fetch_page(search_kwargs, offset=len(papers), size=size)
            papers.extend(_to_paper(r) for r in page)
            if len(page) < size:  # short page = no more results
                break
        return papers

"""arXiv client: metadata (the API at export.arxiv.org) and, for full text, a paper's
HTML or PDF from arxiv.org. A PDF is only parsed in memory and never stored or
re-served (CLAUDE.md rule 12); users are linked to the arXiv abstract page.

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
- HTML and PDF downloads use the same limiter. Redirects are followed by hand, each
  hop in its own limiter slot, and only within arxiv.org.
"""

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlparse

import arxiv
import httpx
import requests
from pydantic import BaseModel

from app.sources.http import Limiter, RetryableError, call_with_retry, make_client, send
from app.sources.ratelimit import RedisRateLimiter, get_redis

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REQUEST_TIMEOUT_S = 20.0
MIN_INTERVAL_S = 3.0
ID_LIST_BATCH = 50  # ids per metadata request (get_papers)
WEB_URL = "https://arxiv.org"
MAX_REDIRECTS = 3
MAX_PDF_BYTES = 50_000_000  # refuse anything larger: not a normal paper PDF

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


_ID_RE = re.compile(r"^(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$")


def validate_arxiv_id(raw: str) -> str:
    """Return the id without version ('1706.03762v5' -> '1706.03762'), or raise ValueError.

    Ids end up inside URLs and queries, so anything unexpected is rejected up front.
    """
    raw = raw.strip()
    if not _ID_RE.match(raw):
        raise ValueError(f"not a valid arXiv id: {raw!r}")
    return split_version(raw)[0]


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
        http: httpx.Client | None = None,
    ) -> None:
        self._limiter: Limiter = limiter or RedisRateLimiter(
            get_redis(), "arxiv", min_interval_s=MIN_INTERVAL_S
        )
        self._page_size = page_size
        self._sleep = sleep
        # For HTML/PDF downloads. No automatic redirects: see `_get_web`.
        self._http = http or make_client(WEB_URL, follow_redirects=False)
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

    def get_papers(self, arxiv_ids: Sequence[str]) -> list[ArxivPaper]:
        """Metadata for many ids, one request per 50 ids (each through the limiter).
        Ids arXiv does not know are simply missing from the result."""
        papers: list[ArxivPaper] = []
        for start in range(0, len(arxiv_ids), ID_LIST_BATCH):
            batch = list(arxiv_ids[start : start + ID_LIST_BATCH])
            results = self._fetch_page({"id_list": batch}, offset=0, size=len(batch))
            papers.extend(_to_paper(r) for r in results)
        return papers

    def updated_papers(
        self,
        categories: Sequence[str],
        since: datetime,
        until: datetime | None = None,
        page_size: int = 100,
        max_papers: int = 20_000,
    ) -> list[ArxivPaper]:
        """Papers in any of `categories` whose latest version is dated in [since, until],
        newest first: new submissions and new versions (v2, v3, ...).

        arXiv's `lastUpdatedDate:[..]` query filter does not reliably restrict results to
        the window (checked live on 2026-10-01: a one-day window returned entries updated
        on other days), so the window is applied here instead: one combined category
        query sorted by lastUpdatedDate descending, paged until an entry is older than
        `since` (or `max_papers` is reached). Each page is one request through the limiter.
        """
        until = until or datetime.now(UTC)
        query = " OR ".join(f"cat:{c}" for c in categories)
        search_kwargs: dict[str, Any] = {
            "query": f"({query})" if len(categories) > 1 else query,
            "sort_by": arxiv.SortCriterion.LastUpdatedDate,
            "sort_order": arxiv.SortOrder.Descending,
        }
        papers: list[ArxivPaper] = []
        offset = 0
        while len(papers) < max_papers:
            page = self._fetch_page(search_kwargs, offset=offset, size=page_size)
            offset += len(page)
            for result in page:
                paper = _to_paper(result)
                if paper.updated_at < since:
                    return papers  # sorted newest first: everything after is older
                if paper.updated_at <= until and len(papers) < max_papers:
                    papers.append(paper)
            if len(page) < page_size:  # short page = no more results
                break
        return papers

    def _get_web(self, path: str) -> httpx.Response:
        """GET from arxiv.org with retries; every attempt and every redirect hop takes its
        own limiter slot (rule 4: one request per 3 s, counted per request)."""
        url = urljoin(WEB_URL, path)
        kwargs: dict[str, Any] = {} if self._sleep is None else {"sleep": self._sleep}
        for _ in range(MAX_REDIRECTS + 1):
            resp = send(self._http, "GET", url, limiter=self._limiter, **kwargs)
            if not resp.is_redirect:
                return resp
            url = urljoin(url, resp.headers["location"])
            host = urlparse(url).hostname or ""
            if host != "arxiv.org" and not host.endswith(".arxiv.org"):
                raise ValueError(f"refusing to follow a redirect off arxiv.org: {host}")
        raise RuntimeError(f"more than {MAX_REDIRECTS} redirects for {path}")

    def html(self, arxiv_id: str, version: int) -> str | None:
        """arXiv's HTML rendering (LaTeXML) of one version, or None if it has none."""
        resp = self._get_web(f"/html/{validate_arxiv_id(arxiv_id)}v{int(version)}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        if "ltx_document" not in resp.text:  # e.g. a "no HTML for this paper" page
            return None
        return resp.text

    def pdf_bytes(self, arxiv_id: str, version: int) -> bytes | None:
        """The PDF of one version, in memory only (never written to disk or re-served).
        None if arXiv has no PDF or the response is not a PDF."""
        resp = self._get_web(f"/pdf/{validate_arxiv_id(arxiv_id)}v{int(version)}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF") or len(resp.content) > MAX_PDF_BYTES:
            return None
        return resp.content

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

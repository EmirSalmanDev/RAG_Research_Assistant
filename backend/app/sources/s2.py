"""Semantic Scholar Graph API client (citation count + references).

The API key is OPTIONAL and normally absent: without it we share S2's public pool, so
429s are common. The client therefore never raises for "S2 is unavailable": after
retries (exponential backoff + jitter, `Retry-After` honoured) it returns a typed
`S2Unavailable`, so ingestion can mark citation data as missing and carry on.

Rate limit: max 1 request/second across all processes (Redis limiter), one at a time,
never parallel (CLAUDE.md rule 4). Prefer `batch()`: one request for many papers.
"""

import logging
from typing import Any

import httpx
from pydantic import BaseModel

from app.config import get_settings
from app.sources.http import Limiter, RetryableError, make_client, send
from app.sources.ratelimit import RedisRateLimiter, get_redis

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org"
MIN_INTERVAL_S = 1.0
BATCH_LIMIT = 500  # S2 accepts at most 500 ids per batch request
FIELDS = "citationCount,references.externalIds"


class S2Paper(BaseModel):
    arxiv_id: str
    citation_count: int | None
    references: list[str]  # arXiv ids of cited papers (references without an arXiv id are skipped)
    citations: list[str] = []  # arXiv ids of citing papers, only when requested


class S2Unavailable(BaseModel):
    """S2 could not give an answer; `reason` says why (not found, rate limited, ...)."""

    arxiv_id: str
    reason: str


S2Result = S2Paper | S2Unavailable


def _arxiv_ids(items: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    for item in items or []:
        ext = (item or {}).get("externalIds") or {}
        if ext.get("ArXiv"):
            out.append(ext["ArXiv"])
    return out


def _parse(arxiv_id: str, data: dict[str, Any]) -> S2Paper:
    return S2Paper(
        arxiv_id=arxiv_id,
        citation_count=data.get("citationCount"),
        references=_arxiv_ids(data.get("references")),
        citations=_arxiv_ids(data.get("citations")),
    )


class S2Client:
    def __init__(
        self,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        limiter: Limiter | None = None,
        sleep: object = None,
        max_attempts: int = 5,
    ) -> None:
        if api_key is None:
            secret = get_settings().s2_api_key
            api_key = secret.get_secret_value() if secret else None
        # The header is sent only when a key exists (normally it does not).
        self._headers = {"x-api-key": api_key} if api_key else {}
        self._http = client or make_client(BASE_URL)
        self._limiter: Limiter = limiter or RedisRateLimiter(
            get_redis(), "s2", min_interval_s=MIN_INTERVAL_S
        )
        self._max_attempts = max_attempts
        self._send_kwargs: dict[str, object] = {} if sleep is None else {"sleep": sleep}

    def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        return send(
            self._http, method, url, limiter=self._limiter, max_attempts=self._max_attempts,
            headers=self._headers, **self._send_kwargs, **kwargs,  # type: ignore[arg-type]
        )  # fmt: skip

    def paper(self, arxiv_id: str, include_citations: bool = False) -> S2Result:
        """One paper. `include_citations` also fetches who cites it (up to 1000 ids)."""
        fields = FIELDS + (",citations.externalIds" if include_citations else "")
        try:
            resp = self._request(
                "GET", f"/graph/v1/paper/arXiv:{arxiv_id}", params={"fields": fields}
            )
        except (RetryableError, httpx.TransportError) as err:
            return self._unavailable(arxiv_id, err)
        if resp.status_code == 404:
            return S2Unavailable(arxiv_id=arxiv_id, reason="not found on Semantic Scholar")
        if resp.status_code != 200:
            return S2Unavailable(arxiv_id=arxiv_id, reason=f"HTTP {resp.status_code}")
        return _parse(arxiv_id, resp.json())

    def batch(self, arxiv_ids: list[str]) -> dict[str, S2Result]:
        """Many papers per request (chunks of 500, sent one after another)."""
        results: dict[str, S2Result] = {}
        for start in range(0, len(arxiv_ids), BATCH_LIMIT):
            chunk = arxiv_ids[start : start + BATCH_LIMIT]
            results.update(self._batch_chunk(chunk))
        return results

    def _batch_chunk(self, ids: list[str]) -> dict[str, S2Result]:
        try:
            resp = self._request(
                "POST", "/graph/v1/paper/batch", params={"fields": FIELDS},
                json={"ids": [f"ARXIV:{i}" for i in ids]},
            )  # fmt: skip
        except (RetryableError, httpx.TransportError) as err:
            return {i: self._unavailable(i, err) for i in ids}
        if resp.status_code != 200:
            return {i: S2Unavailable(arxiv_id=i, reason=f"HTTP {resp.status_code}") for i in ids}
        # The response is a list in request order; unknown papers come back as null.
        return {
            arxiv_id: _parse(arxiv_id, data)
            if data
            else S2Unavailable(arxiv_id=arxiv_id, reason="not found on Semantic Scholar")
            for arxiv_id, data in zip(ids, resp.json(), strict=True)
        }

    @staticmethod
    def _unavailable(arxiv_id: str, err: Exception) -> S2Unavailable:
        logger.warning("Semantic Scholar unavailable for %s: %s", arxiv_id, err)
        return S2Unavailable(arxiv_id=arxiv_id, reason=f"unavailable after retries: {err}")

"""GitHub REST client: repo stats and README. The token is optional.

Unauthenticated calls are limited to 60/hour, which is plenty for ingest-one; with
`GITHUB_TOKEN` set we send it (and only then).
"""

import re
from datetime import datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.sources.http import Limiter, make_client, send

BASE_URL = "https://api.github.com"

_URL_RE = re.compile(r"github\.com[/:](?P<owner>[\w.-]+)/(?P<name>[\w.-]+?)(?:\.git)?(?:[/#?].*)?$")


def parse_github_url(url: str | None) -> str | None:
    """'https://github.com/owner/name/tree/main' -> 'owner/name' (None if not a repo URL)."""
    if not url:
        return None
    m = _URL_RE.search(url.strip())
    return f"{m.group('owner')}/{m.group('name')}" if m else None


class GitHubRepo(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    full_name: str
    stars: int = Field(alias="stargazers_count", default=0)
    last_push: datetime | None = Field(alias="pushed_at", default=None)
    license: str | None = None  # SPDX id, e.g. "MIT"; None when GitHub can't tell


class GitHubClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        token: str | None = None,
        limiter: Limiter | None = None,
        sleep: object = None,
    ) -> None:
        if token is None:
            secret = get_settings().github_token
            token = secret.get_secret_value() if secret else None
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:  # only send credentials when we actually have them
            self._headers["Authorization"] = f"Bearer {token}"
        self._http = client or make_client(BASE_URL)
        self._limiter = limiter
        self._send_kwargs: dict[str, object] = {} if sleep is None else {"sleep": sleep}

    def _get(self, url: str, accept: str | None = None) -> httpx.Response:
        # Headers are set per request (not on the httpx client) so they are applied
        # whichever client is used.
        headers = {**self._headers, **({"Accept": accept} if accept else {})}
        return send(
            self._http, "GET", url, limiter=self._limiter, headers=headers,
            **self._send_kwargs,  # type: ignore[arg-type]
        )  # fmt: skip

    def repo(self, full_name: str) -> GitHubRepo | None:
        resp = self._get(f"/repos/{full_name}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        spdx = (data.get("license") or {}).get("spdx_id")
        return GitHubRepo.model_validate(
            {**data, "license": None if spdx in (None, "NOASSERTION") else spdx}
        )

    def readme(self, full_name: str) -> str | None:
        """README as raw text (the `raw` media type avoids base64 decoding)."""
        resp = self._get(f"/repos/{full_name}/readme", accept="application/vnd.github.raw+json")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text

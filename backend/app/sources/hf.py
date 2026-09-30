"""Hugging Face client (no auth needed): daily papers, paper pages, markdown full text,
and the models/datasets/spaces linked to a paper.

Everything returned here is untrusted third-party content (CLAUDE.md rule 8): it is
stored and shown as data and must only ever enter prompts inside delimited data blocks.
"""

import logging
from datetime import date, datetime
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.sources.http import Limiter, make_client, send

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co"
ArtifactKind = Literal["model", "dataset", "space"]
_KIND_PATH: dict[ArtifactKind, str] = {"model": "models", "dataset": "datasets", "space": "spaces"}


class HFPaper(BaseModel):
    # extra="ignore": HF adds fields often; we only keep what we use.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str  # arXiv id
    title: str | None = None
    summary: str | None = None
    upvotes: int = 0
    github_repo: str | None = Field(default=None, alias="githubRepo")
    github_stars: int | None = Field(default=None, alias="githubStars")
    published_at: datetime | None = Field(default=None, alias="publishedAt")


class HFDailyEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paper: HFPaper


class HFArtifact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: ArtifactKind
    repo_id: str
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None


class HFClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        limiter: Limiter | None = None,
        sleep: object = None,
    ) -> None:
        self._http = client or make_client(BASE_URL)
        self._limiter = limiter  # HF publishes no strict limit; calls are sequential anyway
        self._send_kwargs: dict[str, object] = {} if sleep is None else {"sleep": sleep}

    def _get(self, url: str, **params: object) -> httpx.Response:
        return send(
            self._http, "GET", url, limiter=self._limiter, params=params or None,
            **self._send_kwargs,  # type: ignore[arg-type]
        )  # fmt: skip

    def daily_papers(self, day: date) -> list[HFPaper]:
        """Papers HF featured on `day`, most upvoted ('trending') first."""
        resp = self._get("/api/daily_papers", date=day.isoformat(), sort="trending")
        resp.raise_for_status()
        return [HFDailyEntry.model_validate(e).paper for e in resp.json()]

    def paper(self, arxiv_id: str) -> HFPaper | None:
        """HF's page data for a paper (upvotes, linked GitHub repo). None if HF lacks it."""
        resp = self._get(f"/api/papers/{arxiv_id}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return HFPaper.model_validate(resp.json())

    def paper_markdown(self, arxiv_id: str) -> str | None:
        """Full text as markdown (HF's rendering). None if HF has no markdown for it."""
        resp = self._get(f"/papers/{arxiv_id}.md")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.text

    def artifacts(self, arxiv_id: str, kind: ArtifactKind) -> list[HFArtifact]:
        """Models, datasets or spaces that cite this paper (`filter=arxiv:<id>`)."""
        resp = self._get(f"/api/{_KIND_PATH[kind]}", filter=f"arxiv:{arxiv_id}", limit=100)
        resp.raise_for_status()
        return [
            HFArtifact(
                kind=kind,
                repo_id=item["id"],
                downloads=item.get("downloads"),
                likes=item.get("likes"),
                pipeline_tag=item.get("pipeline_tag"),
            )
            for item in resp.json()
        ]

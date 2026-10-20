"""Builders and fakes for the review tests (no database, no OpenAI; all text is invented)."""

import uuid
from datetime import UTC, datetime
from typing import Any

from app.features.bibtex import BibPaper
from app.features.review import (
    AnalysisText,
    DraftOut,
    DraftParagraph,
    DraftSection,
    DraftSentence,
    ReviewDeps,
    ReviewPaper,
)
from tests.chat_fakes import fake_chat
from tests.tracing_fakes import RecordingTracer


def make_paper(
    n: int,
    *,
    analysis: bool = True,
    analysis_chunk: bool = True,
    abstract_chunk: bool = True,
    title: str | None = None,
    authors: tuple[str, ...] | None = None,
    year: int = 2020,
) -> ReviewPaper:
    arxiv_id = f"2601.{n:05d}"
    return ReviewPaper(
        bib=BibPaper(
            arxiv_id,
            title or f"Invented Study {n}",
            authors if authors is not None else (f"Ada Author{chr(96 + n)}",),
            datetime(year, 1, n, tzinfo=UTC),
            "cs.CL",
        ),
        abstract=f"Invented abstract {n}.",
        abstract_chunk_id=uuid.UUID(int=1000 + n) if abstract_chunk else None,
        analysis=AnalysisText(
            f"Invented tldr {n}.",
            f"Invented contributions {n}.",
            f"Invented method {n}.",
            f"Invented results {n}.",
            f"Invented limitations {n}.",
        )
        if analysis
        else None,  # fmt: skip
        analysis_chunk_id=uuid.UUID(int=2000 + n) if analysis_chunk else None,
    )


class FakeStore:
    def __init__(self, collections: dict[str, list[ReviewPaper]]) -> None:
        self.collections = collections
        self.loads: list[str] = []

    def load(self, collection: str) -> list[ReviewPaper] | None:
        self.loads.append(collection)
        return self.collections.get(collection)


def sentence(text: str, kind: str = "claim", *numbers: int) -> DraftSentence:
    return DraftSentence(text=text, kind=kind, source_numbers=list(numbers))


def draft(*sections: tuple[str, list[list[DraftSentence]]]) -> DraftOut:
    return DraftOut(
        sections=[
            DraftSection(heading=h, paragraphs=[DraftParagraph(sentences=p) for p in paragraphs])
            for h, paragraphs in sections
        ]
    )


def deps_for(
    papers: list[ReviewPaper], replies: list[Any]
) -> tuple[ReviewDeps, Any, RecordingTracer, list[tuple[str, str, int, int]]]:
    chat, completions, rows = fake_chat(replies)
    tracer = RecordingTracer()
    return ReviewDeps(chat, FakeStore({"Thesis": papers}), tracer), completions, tracer, rows


def collection(count: int, **kwargs: Any) -> list[ReviewPaper]:
    return [make_paper(i, **kwargs) for i in range(1, count + 1)]

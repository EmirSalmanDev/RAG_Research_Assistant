"""fetch_fulltext (fallback chain, statuses, force_source) and the selection rules, on a
throwaway database with fake sources (no network)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.db.models import LibraryItem, Paper, PaperText
from app.ingest.fulltext import (
    MIN_FULLTEXT_CHARS,
    FulltextError,
    fetch_fulltext,
    request_fulltext,
    select_for_fulltext,
)
from tests.unit.test_fulltext_convert import LATEXML, _invented_pdf

pytestmark = pytest.mark.integration

ID = "2601.00001"
# The invented HF-style fixture, padded with invented text to the length of a real paper.
HF_MD = (Path(__file__).parents[1] / "fixtures" / "markdown" / "hf_style.md").read_text() + (
    "\n\nInvented filler sentence for length. " * 60
)


@dataclass
class FakeSources:
    """Each source returns a value, None ('missing') or raises an error."""

    hf: str | None | Exception = None
    html: str | None | Exception = None
    pdf: bytes | None | Exception = None
    calls: list[str] = field(default_factory=list)

    def _give(self, name: str, value: object) -> object:
        self.calls.append(name)
        if isinstance(value, Exception):
            raise value
        return value

    def hf_markdown(self, arxiv_id: str) -> str | None:
        return self._give("hf", self.hf)  # type: ignore[return-value]

    def arxiv_html(self, arxiv_id: str, version: int) -> str | None:
        return self._give("html", self.html)  # type: ignore[return-value]

    def arxiv_pdf(self, arxiv_id: str, version: int) -> bytes | None:
        return self._give("pdf", self.pdf)  # type: ignore[return-value]


# Long enough to count as a full text (MIN_FULLTEXT_CHARS).
LONG_HTML = LATEXML.replace("Invented proof.", "Invented proof. " * 200)


def _paper(session: Session, arxiv_id: str = ID, version: int = 1, status: str = "none") -> Paper:
    paper = Paper(
        arxiv_id=arxiv_id, version=version, title="Invented", abstract="Invented abstract.",
        authors=[], categories=["cs.CL"], primary_category="cs.CL",
        published_at=datetime(2026, 1, 2, tzinfo=UTC), fulltext_status=status,
    )  # fmt: skip
    session.add(paper)
    session.commit()
    return paper


# --- the fallback chain -------------------------------------------------------------------------


def test_hf_markdown_is_tried_first(migrated_session: Session) -> None:
    _paper(migrated_session)
    sources = FakeSources(hf=HF_MD, html=LONG_HTML)
    result = fetch_fulltext(migrated_session, ID, sources=sources)
    migrated_session.commit()
    assert (result.source, result.status, sources.calls) == ("hf", "fetched", ["hf"])
    text = migrated_session.get(PaperText, ID)
    assert text is not None and text.raw_markdown == HF_MD and text.version == 1
    assert text.markdown == "" and text.token_count == 0  # cleaned by the next step
    paper = migrated_session.get(Paper, ID)
    assert paper is not None and (paper.fulltext_source, paper.fulltext_status) == ("hf", "fetched")


def test_missing_hf_falls_back_to_html_then_pdf(migrated_session: Session) -> None:
    _paper(migrated_session)
    sources = FakeSources(hf=None, html=None, pdf=_invented_pdf())
    # The invented one-page PDF is shorter than a real paper: it counts as 'too short'.
    result = fetch_fulltext(migrated_session, ID, sources=sources)
    assert sources.calls == ["hf", "html", "pdf"]
    assert result.tried["hf"] == "missing" and result.tried["html"] == "missing"
    assert result.tried["pdf"].startswith("too short")
    assert result.status == "unavailable"

    sources = FakeSources(hf=None, html=LONG_HTML)
    result = fetch_fulltext(migrated_session, ID, sources=sources)
    assert (result.source, sources.calls) == ("html", ["hf", "html"])
    text = migrated_session.get(PaperText, ID)
    assert text is not None and text.raw_markdown.startswith("# Invented Paper")  # converted


def test_errors_move_on_to_the_next_source(migrated_session: Session) -> None:
    _paper(migrated_session)
    sources = FakeSources(hf=RuntimeError("HF is down"), html=LONG_HTML)
    result = fetch_fulltext(migrated_session, ID, sources=sources)
    assert result.source == "html"
    assert result.tried["hf"].startswith("error: RuntimeError: HF is down")


def test_errors_and_no_text_raise_so_the_caller_can_retry(migrated_session: Session) -> None:
    paper = _paper(migrated_session)
    sources = FakeSources(hf=RuntimeError("HF is down"), html=None, pdf=None)
    with pytest.raises(FulltextError, match="HF is down"):
        fetch_fulltext(migrated_session, ID, sources=sources)
    assert paper.fulltext_status == "none"  # not 'unavailable': a retry may succeed


def test_all_sources_missing_means_unavailable(migrated_session: Session) -> None:
    paper = _paper(migrated_session)
    result = fetch_fulltext(migrated_session, ID, sources=FakeSources())
    assert (result.status, paper.fulltext_status, paper.fulltext_source) == (
        "unavailable", "unavailable", None,
    )  # fmt: skip
    assert migrated_session.get(PaperText, ID) is None


def test_force_source_uses_only_that_source(migrated_session: Session) -> None:
    _paper(migrated_session)
    sources = FakeSources(hf=HF_MD, html=LONG_HTML)
    result = fetch_fulltext(migrated_session, ID, sources=sources, force_source="html")
    assert (result.source, sources.calls) == ("html", ["html"])

    sources = FakeSources(hf=HF_MD, pdf=None)
    result = fetch_fulltext(migrated_session, ID, sources=sources, force_source="pdf")
    assert (result.status, sources.calls) == ("unavailable", ["pdf"])


def test_a_new_fetch_overwrites_the_stored_text(migrated_session: Session) -> None:
    paper = _paper(migrated_session)
    fetch_fulltext(migrated_session, ID, sources=FakeSources(hf=HF_MD))
    paper.version = 2
    fetch_fulltext(migrated_session, ID, sources=FakeSources(html=LONG_HTML), force_source="html")
    migrated_session.commit()
    text = migrated_session.get(PaperText, ID)
    assert text is not None and text.version == 2 and text.raw_markdown.startswith("# Invented")
    assert len(text.raw_markdown) >= MIN_FULLTEXT_CHARS


# --- selection ---------------------------------------------------------------------------------


def _store_text(session: Session, arxiv_id: str, version: int) -> None:
    session.add(PaperText(arxiv_id=arxiv_id, version=version, raw_markdown="x", markdown="x",
                          token_count=1))  # fmt: skip
    session.commit()


def test_selection_rules(migrated_session: Session) -> None:
    s = migrated_session
    _paper(s, "2601.00001")  # in the library, no text          -> selected
    _paper(s, "2601.00002", status="requested")  # requested, no text -> selected
    _paper(s, "2601.00003")  # neither                             -> not selected
    _paper(s, "2601.00004", version=2)  # library, text of v1 only -> selected (version bump)
    _paper(s, "2601.00005")  # library, text of current version    -> not selected
    _paper(s, "2601.00006", status="failed")  # library but failed  -> not selected
    for arxiv_id in ("2601.00001", "2601.00004", "2601.00005", "2601.00006"):
        s.add(LibraryItem(arxiv_id=arxiv_id, collection="invented"))
    s.commit()
    _store_text(s, "2601.00004", 1)
    _store_text(s, "2601.00005", 1)
    assert list(select_for_fulltext(s)) == ["2601.00001", "2601.00002", "2601.00004"]

    # A manual request resets 'failed', so the paper is picked up again.
    request_fulltext(s, "2601.00006")
    s.commit()
    assert "2601.00006" in select_for_fulltext(s)


def test_request_for_an_unknown_paper_is_an_error(migrated_session: Session) -> None:
    with pytest.raises(ValueError, match="no paper"):
        request_fulltext(migrated_session, "2601.99999")

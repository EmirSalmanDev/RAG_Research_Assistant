"""Get a paper's full text: Hugging Face .md, then arXiv HTML, then the PDF.

- `html_to_markdown()` turns arXiv's LaTeXML HTML into markdown: section titles become
  headings, math becomes its LaTeX `alttext`, figure and table captions are kept, tables
  become pipe tables, and the bibliography becomes a `## References` block (which the
  cleaner then cuts out into `paper_texts.references`).
- `pdf_to_markdown()` uses PyMuPDF4LLM (AGPL-3.0), imported only inside the function so
  only the process that parses a PDF loads it. The PDF bytes stay in memory: they are
  never written to disk, stored or re-served (rule 12).
- `fetch_fulltext()` walks the fallback chain and stores the raw text in
  `paper_texts.raw_markdown` (cleaning happens in the next step).

Everything fetched here is untrusted third-party text (rule 8): it is stored and shown
as data and may only enter prompts inside delimited data blocks.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, Tag
from sqlalchemy import exists, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import DigestEntry, LibraryItem, Paper, PaperText
from app.ids import LOCAL_PREFIX, is_local_id

logger = logging.getLogger(__name__)

FulltextSource = Literal["hf", "html", "pdf"]
SOURCE_ORDER: tuple[FulltextSource, ...] = ("hf", "html", "pdf")
# Shorter results are not a real full text (e.g. an abstract-only or error page).
MIN_FULLTEXT_CHARS = 2000


# --- HTML (LaTeXML) -> markdown ---------------------------------------------------------------

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_SKIP_CLASSES = {"ltx_note_mark", "ltx_page_footer", "ltx_page_header", "ltx_dates"}


def _classes(tag: Tag) -> set[str]:
    value = tag.get("class")
    return set(value) if isinstance(value, list) else set()


def _math(tag: Tag, display: bool) -> str:
    alt = str(tag.get("alttext") or tag.get_text(" ", strip=True))
    alt = " ".join(alt.split())
    return f"$$\n{alt}\n$$" if display else f"${alt}$"


def _inline(node: Tag | NavigableString) -> str:
    """The text of an inline element, with math as `$alttext$`."""
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in ("script", "style") or _classes(node) & _SKIP_CLASSES:
        return ""
    if node.name == "math":
        return _math(node, display=False)
    if node.name == "br":
        return " "
    if "ltx_note" in _classes(node):  # a footnote: kept inline, in parentheses
        content = node.find(class_="ltx_note_content")
        text = _inline(content) if isinstance(content, Tag) else ""
        return f" ({' '.join(text.split())})" if text.strip() else ""
    return "".join(_inline(child) for child in node.children)  # type: ignore[arg-type]


def _text(node: Tag) -> str:
    return " ".join(_inline(node).split())


def _table(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_text(cell).replace("|", "\\|") for cell in tr.find_all(["td", "th"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _blocks(node: Tag, out: list[str]) -> None:
    """Append the markdown blocks of `node`'s children to `out`, in document order."""
    for child in node.children:
        if isinstance(child, Comment) or not isinstance(child, (Tag, NavigableString)):
            continue
        if isinstance(child, NavigableString):
            if child.strip():
                out.append(" ".join(child.split()))
            continue
        classes = _classes(child)
        if child.name in ("script", "style", "nav", "header", "footer") or classes & _SKIP_CLASSES:
            continue
        if child.name in _HEADING_TAGS:
            level = 2 if "ltx_title_abstract" in classes else _HEADING_TAGS[child.name]
            title = _text(child)
            if "ltx_title_bibliography" in classes:
                title = "References"
            if title:
                out.append("#" * level + " " + title)
        elif child.name == "math":
            out.append(_math(child, display=child.get("display") == "block"))
        elif "ltx_bibliography" in classes:
            out.append("## References")
            for item in child.find_all(class_="ltx_bibitem"):
                out.append(_text(item))
        elif child.name == "figure" or "ltx_figure" in classes or "ltx_table" in classes:
            caption = child.find("figcaption")
            if isinstance(caption, Tag) and _text(caption):
                out.append(_text(caption))
            for table in child.find_all("table", class_="ltx_tabular"):
                if text := _table(table):
                    out.append(text)
        elif child.name == "table" and classes & {
            "ltx_equation",
            "ltx_equationgroup",
            "ltx_eqn_table",
        }:
            for math in child.find_all("math"):
                out.append(_math(math, display=True))
        elif child.name == "table":
            if text := _table(child):
                out.append(text)
        elif child.name in ("ul", "ol"):
            for item in child.find_all("li", recursive=False):
                out.append("- " + _text(item))
        elif child.name == "pre" or "ltx_listing" in classes:
            out.append("```\n" + child.get_text("\n").strip("\n") + "\n```")
        elif child.name == "p" or "ltx_p" in classes:
            if text := _text(child):
                out.append(text)
        else:  # section, div.ltx_para, ...: containers, walk into them
            _blocks(child, out)


def html_to_markdown(html: str) -> str:
    """arXiv's LaTeXML HTML as markdown. Raises ValueError if it is not a LaTeXML page."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.find(class_="ltx_document")
    if not isinstance(article, Tag):
        raise ValueError("not an arXiv LaTeXML document (no .ltx_document)")
    out: list[str] = []
    _blocks(article, out)
    return "\n\n".join(block for block in out if block.strip()) + "\n"


# --- PDF -> markdown --------------------------------------------------------------------------


def pdf_to_markdown(pdf: bytes) -> str:
    """A PDF (bytes, in memory) as markdown, via PyMuPDF4LLM. No OCR: arXiv PDFs carry
    text; page headers and footers are left out."""
    import pymupdf  # AGPL-3.0, with pymupdf4llm: imported here so only PDF parsing loads it
    import pymupdf4llm

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        text = pymupdf4llm.to_markdown(doc, use_ocr=False, header=False, footer=False)
    return str(text)


# --- sources and the fallback chain -----------------------------------------------------------


class FulltextSources(Protocol):
    """Where full text comes from (tests pass fakes)."""

    def hf_markdown(self, arxiv_id: str) -> str | None: ...
    def arxiv_html(self, arxiv_id: str, version: int) -> str | None: ...
    def arxiv_pdf(self, arxiv_id: str, version: int) -> bytes | None: ...


class _HFMarkdown(Protocol):
    def paper_markdown(self, arxiv_id: str) -> str | None: ...


class _ArxivWeb(Protocol):
    def html(self, arxiv_id: str, version: int) -> str | None: ...
    def pdf_bytes(self, arxiv_id: str, version: int) -> bytes | None: ...


@dataclass
class ClientSources:
    """FulltextSources backed by the real Hugging Face and arXiv clients."""

    hf: _HFMarkdown
    arxiv: _ArxivWeb

    def hf_markdown(self, arxiv_id: str) -> str | None:
        return self.hf.paper_markdown(arxiv_id)

    def arxiv_html(self, arxiv_id: str, version: int) -> str | None:
        return self.arxiv.html(arxiv_id, version)

    def arxiv_pdf(self, arxiv_id: str, version: int) -> bytes | None:
        return self.arxiv.pdf_bytes(arxiv_id, version)


def default_sources() -> ClientSources:
    from app.sources.arxiv import ArxivClient
    from app.sources.hf import HFClient

    return ClientSources(hf=HFClient(), arxiv=ArxivClient())


class FulltextError(RuntimeError):
    """No source gave a full text and at least one failed with an error (worth a retry)."""


@dataclass
class FulltextResult:
    arxiv_id: str
    source: FulltextSource | None  # None: no source had a usable text
    status: str  # papers.fulltext_status after this step: 'fetched' or 'unavailable'
    chars: int = 0
    tried: dict[str, str] = field(default_factory=dict)  # source -> outcome


def _from_source(sources: FulltextSources, source: FulltextSource, paper: Paper) -> str | None:
    if source == "hf":
        return sources.hf_markdown(paper.arxiv_id)
    if source == "html":
        html = sources.arxiv_html(paper.arxiv_id, paper.version)
        return html_to_markdown(html) if html is not None else None
    pdf = sources.arxiv_pdf(paper.arxiv_id, paper.version)
    return pdf_to_markdown(pdf) if pdf is not None else None


def fetch_fulltext(
    session: Session,
    arxiv_id: str,
    *,
    sources: FulltextSources,
    force_source: FulltextSource | None = None,
) -> FulltextResult:
    """Try the sources in order (or only `force_source`) and store the first usable text
    in `paper_texts.raw_markdown` for the paper's current version.

    - A source that has no text (404, no LaTeXML, too short) is 'missing': try the next.
    - A source that fails (network error after retries, a parse error) is logged and the
      next one is tried; if no source succeeds, FulltextError is raised so the caller
      can retry later. Only when every source is 'missing' is the paper 'unavailable'.
    The caller commits.
    """
    paper = session.get(Paper, arxiv_id)
    if paper is None:
        raise ValueError(f"no paper {arxiv_id} in Postgres")
    result = FulltextResult(arxiv_id=arxiv_id, source=None, status="unavailable")
    errors = 0
    for source in (force_source,) if force_source else SOURCE_ORDER:
        try:
            text = _from_source(sources, source, paper)
        except Exception as err:  # noqa: BLE001 - any failure moves on to the next source
            logger.warning("full text %s: source %s failed: %s", arxiv_id, source, err)
            result.tried[source] = f"error: {type(err).__name__}: {err}"[:300]
            errors += 1
            continue
        if text is None or len(text.strip()) < MIN_FULLTEXT_CHARS:
            result.tried[source] = "missing" if text is None else f"too short ({len(text)} chars)"
            continue
        _store_raw(session, paper, text)
        paper.fulltext_source, paper.fulltext_status = source, "fetched"
        result.tried[source] = "ok"
        result.source, result.status, result.chars = source, "fetched", len(text)
        return result
    if errors:
        raise FulltextError(f"{arxiv_id}: no full text; errors from sources: {result.tried}")
    paper.fulltext_source, paper.fulltext_status = None, "unavailable"
    return result


def _store_raw(session: Session, paper: Paper, raw: str) -> None:
    """Upsert the raw text. The cleaned `markdown` is emptied: the clean step fills it."""
    values = {
        "arxiv_id": paper.arxiv_id,
        "version": paper.version,
        "raw_markdown": raw,
        "markdown": "",
        "references": None,
        "token_count": 0,
    }
    stmt = pg_insert(PaperText).values(**values)
    session.execute(
        stmt.on_conflict_do_update(
            index_elements=[PaperText.arxiv_id],
            set_={k: stmt.excluded[k] for k in values if k != "arxiv_id"}
            | {"parsed_at": func.now()},
        )
    )


# --- which papers get full text ---------------------------------------------------------------


def request_fulltext(session: Session, arxiv_id: str) -> None:
    """A manual request (UI, CLI): mark the paper 'requested', which also resets 'failed'
    and 'unavailable', so the next selection picks it up. The caller commits."""
    paper = session.get(Paper, arxiv_id)
    if paper is None:
        raise ValueError(f"no paper {arxiv_id} in Postgres")
    if is_local_id(arxiv_id):  # an uploaded document keeps the text it was uploaded with
        return
    paper.fulltext_status = "requested"


def select_for_fulltext(
    session: Session, *, top_n: int = 0, now: datetime | None = None
) -> Sequence[str]:
    """Papers that need (new) full text: first those in the library or requested by hand,
    then (when `top_n` > 0) the best papers by `relevance_score`. A paper is left out when
    it has stored text of its current version. 'failed' papers are left out too (no
    automatic retry; a manual request resets them to 'requested'), and so are
    'unavailable' ones among the top N (no source had text, and retrying every night
    would only repeat the downloads).

    The top N are the N best-scored papers of the digest window (published within
    `digest_window_days`, never a dismissed one), so each night only the papers that newly
    entered the top N cost a download. Selection only: nothing is fetched here.
    """
    has_current_text = exists().where(
        PaperText.arxiv_id == Paper.arxiv_id, PaperText.version >= Paper.version
    )
    in_library = exists().where(LibraryItem.arxiv_id == Paper.arxiv_id)
    query = (
        select(Paper.arxiv_id)
        .where(in_library | (Paper.fulltext_status == "requested"))
        .where(~Paper.arxiv_id.startswith(LOCAL_PREFIX))  # an upload has no source to fetch
        .where(Paper.fulltext_status != "failed")
        .where(~has_current_text)
        .order_by(Paper.arxiv_id)
    )
    chosen = list(session.scalars(query).all())
    if top_n > 0:
        taken = set(chosen)
        chosen += [i for i in _top_by_relevance(session, top_n, now) if i not in taken]
    return chosen


def _top_by_relevance(session: Session, top_n: int, now: datetime | None) -> list[str]:
    """The top `top_n` scored papers of the window that still need text, best first
    (ties: newer first, then arXiv id)."""
    start = (now or datetime.now(UTC)) - timedelta(days=get_settings().digest_window_days)
    dismissed = exists().where(
        DigestEntry.arxiv_id == Paper.arxiv_id, DigestEntry.status == "dismissed"
    )
    top = session.execute(
        select(Paper.arxiv_id, Paper.fulltext_status)
        .where(
            Paper.relevance_score.is_not(None),
            Paper.published_at >= start,
            ~dismissed,
            ~Paper.arxiv_id.startswith(LOCAL_PREFIX),
        )
        .order_by(Paper.relevance_score.desc(), Paper.published_at.desc(), Paper.arxiv_id)
        .limit(top_n)
    ).all()
    has_current_text = exists().where(
        PaperText.arxiv_id == Paper.arxiv_id, PaperText.version >= Paper.version
    )
    holding_text = set(
        session.scalars(
            select(Paper.arxiv_id).where(Paper.arxiv_id.in_([r[0] for r in top]), has_current_text)
        )
    )
    return [
        arxiv_id
        for arxiv_id, status in top
        if arxiv_id not in holding_text and status not in ("failed", "unavailable")
    ]

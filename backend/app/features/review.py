"""The literature review builder: a cited related-work draft for one library collection.

    POST /library/review  {"collection": "...", "focus": "...", "max_papers": 30}

How it works (docs/adr/0011):
1. The papers are read from Postgres only: no retrieval, no embeddings. Each paper becomes
   ONE numbered source [1]..[N]. Its evidence is its stored analysis (`paper_analyses`, with
   its analysis chunk as the citation target) or, when it has none, its abstract chunk.
2. Up to `review_direct_max_papers` papers go straight to the writer. More papers are first
   grouped (stable order, groups of `review_map_group_size`) and ONE structured "map" call
   per group (`review_map_v1`) returns short theme notes that keep the source numbers.
3. ONE structured "write" call (`review_write_v1`) produces the draft: sections, paragraphs
   and sentences, each sentence a `claim` or a `transition` with its source numbers.
4. Code, not the model, decides what is shown (`validate_draft`): a source number must exist
   (and, after the map step, appear in a note); invalid numbers are dropped; a `claim` with no
   valid number is REMOVED and counted in `dropped_uncited`; a transition that names an
   author or holds a number is an uncited claim and is removed too; transitions are limited
   to a share of the sentences. An uncited claim is never shown.

What the guarantee is NOT: a citation that resolves does not prove the cited text supports
the sentence (the model can cite a real source for a claim it does not make). The draft also
rests on generated analyses and abstracts, not on the papers' full text. The response says
which evidence each reference has.

Nothing is stored. Every request writes its own `llm_usage` rows (`review-map`,
`review-write`).
"""

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal, Protocol

from pydantic import BaseModel
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.models import Chunk, PaperAnalysis
from app.db.session import session_scope
from app.features import library
from app.features.bibtex import BibPaper, assign_keys, fold_words, split_name
from app.features.deps import end_span, upstream
from app.features.errors import InvalidRequest, UnknownCollection, UpstreamError
from app.features.spend import Spend
from app.features.text import cut_to_tokens
from app.ids import abs_url
from app.ingest.tokens import count_tokens
from app.llm.client import MESSAGE_OVERHEAD_TOKENS, ChatClient, ChatMessage, ChatResponseError
from app.observability.tracing import Tracer, get_tracer
from app.rag.prompts import BLOCK_TAGS, load_prompt

MIN_PAPERS = 2  # the API contract: a review needs at least two papers
FOCUS_MAX_CHARS = 200
MAP_FEATURE = "review-map"  # llm_usage.feature of a map call
WRITE_FEATURE = "review-write"  # llm_usage.feature of the writing call
NOT_STATED_FIELD = "Not stated in the available text."

Evidence = Literal["analysis", "abstract"]
SentenceKind = Literal["claim", "transition"]

_CITATION_MARKER = re.compile(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]")
# The tag names used in this module's prompts, on top of those the chat prompts escape.
_TAGS = (*BLOCK_TAGS, "notes", "source_list", "focus")
_TAG = re.compile(r"<(/?)\s*(" + "|".join(_TAGS) + r")\b", re.IGNORECASE)


def escape(text: str) -> str:
    """Untrusted text made safe for a data block: our tag names inside it cannot close the
    block (the same idea as `app.rag.prompts.escape_block`, with this module's tags)."""
    return _TAG.sub(lambda m: f"&lt;{m[1]}{m[2]}", text)


# --- what the model returns (Structured Outputs: every field required) -------------------------


class ThemeNote(BaseModel):
    theme: str
    statement: str
    source_numbers: list[int]


class MapOut(BaseModel):
    notes: list[ThemeNote]


class DraftSentence(BaseModel):
    text: str
    kind: SentenceKind
    source_numbers: list[int]


class DraftParagraph(BaseModel):
    sentences: list[DraftSentence]


class DraftSection(BaseModel):
    heading: str
    paragraphs: list[DraftParagraph]


class DraftOut(BaseModel):
    sections: list[DraftSection]


# --- the API response --------------------------------------------------------------------------


class SentenceOut(BaseModel):
    text: str
    kind: SentenceKind
    source_numbers: list[int]  # empty only for a transition


class ParagraphOut(BaseModel):
    sentences: list[SentenceOut]


class SectionOut(BaseModel):
    heading: str
    paragraphs: list[ParagraphOut]


class ReferenceOut(BaseModel):
    n: int
    arxiv_id: str
    bibtex_key: str  # the key of the same paper in GET /library/bibtex
    title: str
    authors: list[str]
    year: int
    chunk_id: uuid.UUID  # what a citation chip opens (GET /chunks/{chunk_id})
    chunk_type: Literal["analysis", "abstract"]
    evidence: Evidence  # "analysis": a generated analysis; "abstract": the abstract only
    abs_url: str | None  # null for an uploaded document
    cited_count: int  # sentences of the draft that cite it


class ReviewCounts(BaseModel):
    papers: int  # papers the review used
    papers_without_analysis: int  # of those, the ones that rest on their abstract
    papers_omitted: int  # in the collection but over max_papers
    papers_uncitable: int  # in the collection but with no chunk to cite
    claims: int  # claim sentences shown
    transitions: int  # transition sentences shown
    dropped_uncited: int  # claim sentences removed for having no valid citation
    dropped_transitions: int  # transitions removed (over the share limit)
    invalid_citations_dropped: int  # source numbers that did not exist or were not allowed
    map_calls: int
    map_notes_dropped: int  # theme notes removed for having no valid source number


class ReviewOut(BaseModel):
    collection: str
    focus: str | None
    sections: list[SectionOut]
    references: list[ReferenceOut]  # numbered 1..N over all papers used, cited or not
    counts: ReviewCounts
    tokens_in: int
    tokens_out: int
    cost_usd: float
    prompt_versions: dict[str, str]  # {"map": ... or absent, "write": ...}


# --- the papers (Postgres only) -----------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisText:
    tldr: str | None
    contributions: str | None
    method: str | None
    results: str | None
    limitations: str | None


@dataclass(frozen=True)
class ReviewPaper:
    bib: BibPaper
    abstract: str
    abstract_chunk_id: uuid.UUID | None
    analysis: AnalysisText | None
    analysis_chunk_id: uuid.UUID | None  # the paper's first `analysis` chunk, if indexed

    @property
    def arxiv_id(self) -> str:
        return self.bib.arxiv_id


class ReviewStore(Protocol):
    def load(self, collection: str) -> list[ReviewPaper] | None:
        """Every paper of the collection in the stable order (oldest first, ties by id);
        None when the collection does not exist."""
        ...


class SqlReviewStore:
    def load(self, collection: str) -> list[ReviewPaper] | None:
        with session_scope() as session:
            if not library.collection_exists(session, collection):
                return None
            papers = library.collection_papers(session, collection)
            ids = [p.arxiv_id for p in papers]
            analyses = {
                a.arxiv_id: a
                for a in session.scalars(
                    select(PaperAnalysis).where(PaperAnalysis.arxiv_id.in_(ids))
                )
            }
            abstract: dict[str, tuple[uuid.UUID, str]] = {}
            analysis_chunk: dict[str, uuid.UUID] = {}
            rows = session.execute(
                select(Chunk.arxiv_id, Chunk.chunk_type, Chunk.id, Chunk.text)
                .where(Chunk.arxiv_id.in_(ids), Chunk.chunk_type.in_(("abstract", "analysis")))
                .order_by(Chunk.arxiv_id, Chunk.ord)
            ).all()
            for arxiv_id, chunk_type, chunk_id, text in rows:
                if chunk_type == "abstract":
                    abstract.setdefault(arxiv_id, (chunk_id, text))
                else:
                    analysis_chunk.setdefault(arxiv_id, chunk_id)  # the lowest ord: the summary
            out: list[ReviewPaper] = []
            for paper in papers:
                row = analyses.get(paper.arxiv_id)
                found = abstract.get(paper.arxiv_id)
                out.append(
                    ReviewPaper(
                        bib=library.bib_paper(paper),
                        abstract=found[1] if found else paper.abstract,
                        abstract_chunk_id=found[0] if found else None,
                        analysis=None
                        if row is None
                        else AnalysisText(
                            row.tldr, row.contributions, row.method, row.results, row.limitations
                        ),
                        analysis_chunk_id=analysis_chunk.get(paper.arxiv_id),
                    )
                )
            return out


@dataclass
class ReviewDeps:
    """What a review talks to, so that tests can replace every part."""

    chat: ChatClient
    store: ReviewStore
    tracer: Tracer

    @classmethod
    def default(cls) -> "ReviewDeps":
        return cls(chat=ChatClient(), store=SqlReviewStore(), tracer=get_tracer())


# --- choosing and numbering the sources ----------------------------------------------------------


@dataclass(frozen=True)
class ReviewSource:
    n: int
    paper: ReviewPaper
    evidence: Evidence
    chunk_id: uuid.UUID
    key: str  # the BibTeX key of the paper in the collection's .bib

    @property
    def arxiv_id(self) -> str:
        return self.paper.arxiv_id


@dataclass(frozen=True)
class Selection:
    sources: list[ReviewSource]  # numbered 1..N in the stable order
    omitted: int  # usable papers left out by max_papers
    uncitable: int  # papers with no chunk to cite


def _evidence_of(paper: ReviewPaper) -> tuple[Evidence, uuid.UUID] | None:
    """The paper's evidence level and the chunk a citation points at, or None when it has
    no chunk at all. An analysis counts only when its chunk is indexed: the citation must
    open the very text the sentence rests on."""
    if paper.analysis is not None and paper.analysis_chunk_id is not None:
        return "analysis", paper.analysis_chunk_id
    if paper.abstract_chunk_id is not None:
        return "abstract", paper.abstract_chunk_id
    return None


def select_sources(papers: Sequence[ReviewPaper], max_papers: int) -> Selection:
    """Number the papers of a collection. The BibTeX keys come from the WHOLE collection (the
    same call the .bib export makes), so a key never depends on `max_papers`. When the
    collection is larger than `max_papers`, papers with an analysis are preferred (their
    evidence is richer), then the stable order; the chosen ones are numbered in the stable order."""
    keys = dict(
        zip((p.arxiv_id for p in papers), assign_keys([p.bib for p in papers]), strict=True)
    )
    usable = [(p, e) for p in papers if (e := _evidence_of(p)) is not None]
    uncitable = len(papers) - len(usable)
    chosen = usable
    if len(usable) > max_papers:
        ranked = sorted(range(len(usable)), key=lambda i: (usable[i][1][0] != "analysis", i))
        chosen = [usable[i] for i in sorted(ranked[:max_papers])]
    sources = [
        ReviewSource(n, paper, evidence, chunk_id, keys[paper.arxiv_id])
        for n, (paper, (evidence, chunk_id)) in enumerate(chosen, start=1)
    ]
    return Selection(sources, omitted=len(usable) - len(chosen), uncitable=uncitable)


# --- the prompts' data blocks -------------------------------------------------------------------


def author_label(authors: Sequence[str]) -> str:
    """ "Smith", "Smith and Jones", "Smith et al.", or "unknown"."""
    lasts = [split_name(a)[0] for a in authors if a.strip()]
    if not lasts:
        return "unknown"
    if len(lasts) == 1:
        return lasts[0]
    return f"{lasts[0]} and {lasts[1]}" if len(lasts) == 2 else f"{lasts[0]} et al."


def _field(label: str, value: str | None, max_tokens: int) -> str | None:
    if value is None or not value.strip():
        return None
    text, _ = cut_to_tokens(value.strip(), max_tokens)
    return f"{label}: {escape(text)}"


def format_source(source: ReviewSource, max_tokens: int) -> str:
    """One paper as a data block. Everything in it comes from papers or generated analyses:
    escaped, inside a block, and the prompts say it is data."""
    paper = source.paper
    lines = [
        f"Title: {escape(paper.bib.title)}",
        f"Authors: {escape(author_label(paper.bib.authors))}",
        f"Year: {paper.bib.published_at.year}",
    ]
    if source.evidence == "analysis" and paper.analysis is not None:
        a = paper.analysis
        fields = (
            ("TL;DR", a.tldr), ("Contributions", a.contributions), ("Method", a.method),
            ("Results", a.results), ("Limitations", a.limitations),
        )  # fmt: skip
        lines += [f for label, value in fields if (f := _field(label, value, max_tokens))]
    else:
        lines.append(_field("Abstract", paper.abstract, max_tokens) or "Abstract: (empty)")
    body = "\n".join(lines)
    return f'<source n="{source.n}" evidence="{source.evidence}">\n{body}\n</source>'


def format_sources(sources: Sequence[ReviewSource], max_tokens: int) -> str:
    return (
        "<sources>\n" + "\n\n".join(format_source(s, max_tokens) for s in sources) + "\n</sources>"
    )


def format_source_list(sources: Sequence[ReviewSource]) -> str:
    lines = [
        f"[{s.n}] {escape(author_label(s.paper.bib.authors))} ({s.paper.bib.published_at.year}),"
        f" evidence: {s.evidence}"
        for s in sources
    ]
    return "<source_list>\n" + "\n".join(lines) + "\n</source_list>"


def format_notes(notes: Sequence[ThemeNote]) -> str:
    blocks = [
        f'<note sources="{", ".join(map(str, n.source_numbers))}">\n'
        f"Theme: {escape(n.theme)}\nStatement: {escape(n.statement)}\n</note>"
        for n in notes
    ]
    return "<notes>\n" + "\n\n".join(blocks) + "\n</notes>"


def _focus_block(focus: str | None) -> str:
    return f"<focus>\n{escape(focus)}\n</focus>\n\n" if focus else ""


def map_messages(
    group: Sequence[ReviewSource], focus: str | None, settings: Settings
) -> list[ChatMessage]:
    user = _focus_block(focus) + format_sources(group, settings.review_field_max_tokens)
    return [
        {"role": "system", "content": load_prompt(settings.review_map_prompt).text},
        {"role": "user", "content": user},
    ]


def write_messages(
    sources: Sequence[ReviewSource],
    notes: Sequence[ThemeNote] | None,
    focus: str | None,
    settings: Settings,
) -> list[ChatMessage]:
    """`notes` None: the writer reads the papers directly; otherwise it reads the notes."""
    if notes is None:
        body = format_sources(sources, settings.review_field_max_tokens)
    else:
        body = format_source_list(sources) + "\n\n" + format_notes(notes)
    return [
        {"role": "system", "content": load_prompt(settings.review_write_prompt).text},
        {"role": "user", "content": _focus_block(focus) + body},
    ]


def groups_of(sources: Sequence[ReviewSource], size: int) -> list[list[ReviewSource]]:
    return [list(sources[i : i + size]) for i in range(0, len(sources), size)]


def uses_map(sources: Sequence[object], settings: Settings) -> bool:
    return len(sources) > settings.review_direct_max_papers


def input_tokens(messages: Sequence[ChatMessage]) -> int:
    return sum(count_tokens(m["content"]) + MESSAGE_OVERHEAD_TOKENS for m in messages)


# --- what the model returned -> what is shown ---------------------------------------------------


@dataclass
class _Counts:
    claims: int = 0
    transitions: int = 0
    dropped_uncited: int = 0
    dropped_transitions: int = 0
    invalid_citations_dropped: int = 0
    map_notes_dropped: int = 0


def validate_notes(
    out: MapOut, group: Sequence[ReviewSource], settings: Settings, counts: _Counts
) -> list[ThemeNote]:
    """The theme notes of one group that can be used: numbers must belong to the group; a
    note with none left is removed."""
    allowed = {s.n for s in group}
    kept: list[ThemeNote] = []
    for note in out.notes:
        numbers = list(dict.fromkeys(n for n in note.source_numbers if n in allowed))
        counts.invalid_citations_dropped += len(set(note.source_numbers) - set(numbers))
        statement = _CITATION_MARKER.sub("", note.statement).strip()
        words = statement.split()
        if len(words) > settings.review_note_max_words:
            statement = " ".join(words[: settings.review_note_max_words]) + "…"
        if not numbers or not statement:
            counts.map_notes_dropped += 1
            continue
        kept.append(
            ThemeNote(
                theme=_CITATION_MARKER.sub("", note.theme).strip(),
                statement=statement,
                source_numbers=numbers,
            )
        )
    return kept


def _clean_text(text: str) -> str:
    return " ".join(_CITATION_MARKER.sub("", text).split())


def _names_a_paper(text: str, surnames: set[str]) -> bool:
    """Does a transition name an author or hold a number? Then it states something about a
    paper and needs a citation like any claim."""
    if re.search(r"\d", text) or re.search(r"\bet al\b", text, re.IGNORECASE):
        return True
    return bool(surnames & set(fold_words(text).split()))


def validate_draft(
    draft: DraftOut,
    sources: Sequence[ReviewSource],
    allowed: set[int],
    settings: Settings,
    counts: _Counts,
) -> list[SectionOut]:
    """Turn the model's draft into the sections to show. The guarantee: every shown `claim`
    has at least one citation that exists (and is allowed). Everything else is removed."""
    existing = {s.n for s in sources} & allowed
    surnames = {
        w
        for s in sources
        for a in s.paper.bib.authors
        for w in fold_words(split_name(a)[0]).split()
        if len(w) >= 3
    }
    # 1. sentence by sentence: citations, kinds
    sections: list[tuple[str, list[list[SentenceOut]]]] = []
    for section in draft.sections:
        paragraphs: list[list[SentenceOut]] = []
        for paragraph in section.paragraphs:
            sentences: list[SentenceOut] = []
            for sentence in paragraph.sentences:
                text = _clean_text(sentence.text)
                if not text:
                    continue
                valid = list(dict.fromkeys(n for n in sentence.source_numbers if n in existing))
                counts.invalid_citations_dropped += len(set(sentence.source_numbers) - set(valid))
                valid = valid[: settings.review_max_citations_per_sentence]
                if valid:  # a sentence with a citation is a claim, whatever it was called
                    sentences.append(SentenceOut(text=text, kind="claim", source_numbers=valid))
                elif sentence.kind == "claim" or _names_a_paper(text, surnames):
                    counts.dropped_uncited += 1
                else:
                    sentences.append(SentenceOut(text=text, kind="transition", source_numbers=[]))
            if sentences:
                paragraphs.append(sentences)
        if paragraphs:
            sections.append((_clean_text(section.heading)[:100] or "Related work", paragraphs))
    # 2. transitions are limited to a share of the kept sentences (later ones go first)
    claims = sum(s.kind == "claim" for _, ps in sections for p in ps for s in p)
    share = settings.review_max_transition_share
    allowed_transitions = int(share * claims / (1 - share)) if 0 < share < 1 else 0
    seen = 0
    shown: list[SectionOut] = []
    for heading, paragraphs in sections:
        out_paragraphs: list[ParagraphOut] = []
        for sentences in paragraphs:
            kept: list[SentenceOut] = []
            for s in sentences:
                if s.kind == "transition":
                    seen += 1
                    if seen > allowed_transitions:
                        counts.dropped_transitions += 1
                        continue
                    counts.transitions += 1
                else:
                    counts.claims += 1
                kept.append(s)
            if kept:
                out_paragraphs.append(ParagraphOut(sentences=kept))
        if out_paragraphs:
            shown.append(SectionOut(heading=heading, paragraphs=out_paragraphs))
    return shown


# --- the whole request ----------------------------------------------------------------------------


def _money(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), ROUND_HALF_UP))


def check_request(
    collection: str, focus: str | None, max_papers: int, settings: Settings
) -> str | None:
    """Raise InvalidRequest for a bad request; return the cleaned focus (None when empty)."""
    if not collection.strip():
        raise InvalidRequest("name a collection")
    cleaned = " ".join((focus or "").split()) or None
    if cleaned is not None and len(cleaned) > FOCUS_MAX_CHARS:
        raise InvalidRequest(f"the focus has at most {FOCUS_MAX_CHARS} characters")
    if not 1 <= max_papers <= settings.review_max_papers:
        raise InvalidRequest(f"max_papers is between 1 and {settings.review_max_papers}")
    return cleaned


def prepare(
    deps: ReviewDeps, collection: str, focus: str | None, max_papers: int
) -> tuple[Selection, str | None]:
    """Everything that can be refused, checked BEFORE the first paid call."""
    settings = get_settings()
    cleaned = check_request(collection, focus, max_papers, settings)
    papers = deps.store.load(collection)
    if papers is None:
        raise UnknownCollection(f"no collection named {collection!r}")
    selection = select_sources(papers, max_papers)
    if len(selection.sources) < MIN_PAPERS:
        raise InvalidRequest(
            f"a review needs at least {MIN_PAPERS} papers with something to cite; "
            f"{collection!r} has {len(selection.sources)}"
        )
    return selection, cleaned


def run_review(
    collection: str,
    focus: str | None = None,
    max_papers: int | None = None,
    *,
    deps: ReviewDeps,
) -> ReviewOut:
    """Write the related-work draft for a collection. Raises InvalidRequest /
    UnknownCollection before any paid call, OpenAIKeyMissing when there is no key,
    UpstreamError when OpenAI fails after the retries or answers with something unusable."""
    settings = get_settings()
    wanted = settings.review_default_papers if max_papers is None else max_papers
    selection, cleaned = prepare(deps, collection, focus, wanted)
    sources = selection.sources
    map_prompt = load_prompt(settings.review_map_prompt)
    write_prompt = load_prompt(settings.review_write_prompt)
    spend = Spend()
    counts = _Counts()
    trace = deps.tracer.start_trace(
        "review",
        input={"collection": collection, "focus": cleaned, "papers": len(sources)},
        metadata={},
    )

    def call[M: BaseModel](
        span_name: str,
        messages: list[ChatMessage],
        schema: type[M],
        feature: str,
        max_tokens: int,
        prompt: str,
        **info: object,
    ) -> M:
        span = trace.span(
            span_name, input={"prompt": prompt, "input_tokens": input_tokens(messages), **info}
        )
        try:
            parsed, usage = deps.chat.parse(
                messages, schema, feature=feature, max_tokens=max_tokens
            )
        except ChatResponseError as err:
            if err.usage is not None:  # billed even though the output was unusable
                spend.add_chat(err.usage)
            end_span(span, error=err)
            trace.end(output="error")
            raise UpstreamError(f"the model did not return a usable result: {err}") from err
        except Exception as err:
            end_span(span, error=err)
            trace.end(output="error")
            mapped = upstream(err)
            if mapped is err:
                raise
            raise mapped from err
        spend.add_chat(usage)
        span.end(usage={"input": usage.tokens_in, "output": usage.tokens_out})
        return parsed

    # --- map: theme notes per group of papers (only above the direct limit)
    notes: list[ThemeNote] | None = None
    map_calls = 0
    if uses_map(sources, settings):
        notes = []
        for index, group in enumerate(groups_of(sources, settings.review_map_group_size), start=1):
            messages = map_messages(group, cleaned, settings)
            out = call(
                "review-map",
                messages,
                MapOut,
                MAP_FEATURE,
                settings.review_map_max_output_tokens,
                map_prompt.id,
                group=index,
                papers=len(group),
            )
            map_calls += 1
            notes += validate_notes(out, group, settings, counts)
        if not notes:  # nothing usable to write from; every claim would be dropped anyway
            trace.end(output="error")
            raise UpstreamError("the model returned no usable theme notes")
        allowed = {n for note in notes for n in note.source_numbers}
    else:
        allowed = {s.n for s in sources}

    # --- write: the draft
    messages = write_messages(sources, notes, cleaned, settings)
    draft = call(
        "review-write",
        messages,
        DraftOut,
        WRITE_FEATURE,
        settings.review_write_max_output_tokens,
        write_prompt.id,
        mode="notes" if notes is not None else "direct",
    )
    shown = validate_draft(draft, sources, allowed, settings, counts)

    cited: dict[int, int] = {}
    for section in shown:
        for paragraph in section.paragraphs:
            for sentence in paragraph.sentences:
                for n in sentence.source_numbers:
                    cited[n] = cited.get(n, 0) + 1
    trace.end(
        output="ok",
        metadata={
            "prompts": [write_prompt.id] + ([map_prompt.id] if map_calls else []),
            "claims": counts.claims,
            "dropped_uncited": counts.dropped_uncited,
            "invalid_citations_dropped": counts.invalid_citations_dropped,
        },
    )
    prompts = {"write": write_prompt.id}
    if map_calls:
        prompts["map"] = map_prompt.id
    return ReviewOut(
        collection=collection,
        focus=cleaned,
        sections=shown,
        references=[_reference(s, cited.get(s.n, 0)) for s in sources],
        counts=ReviewCounts(
            papers=len(sources),
            papers_without_analysis=sum(s.evidence == "abstract" for s in sources),
            papers_omitted=selection.omitted,
            papers_uncitable=selection.uncitable,
            claims=counts.claims,
            transitions=counts.transitions,
            dropped_uncited=counts.dropped_uncited,
            dropped_transitions=counts.dropped_transitions,
            invalid_citations_dropped=counts.invalid_citations_dropped,
            map_calls=map_calls,
            map_notes_dropped=counts.map_notes_dropped,
        ),
        tokens_in=spend.tokens_in,
        tokens_out=spend.tokens_out,
        cost_usd=_money(spend.cost_usd),
        prompt_versions=prompts,
    )


def _reference(source: ReviewSource, cited_count: int) -> ReferenceOut:
    bib = source.paper.bib
    published: datetime = bib.published_at
    return ReferenceOut(
        n=source.n,
        arxiv_id=source.arxiv_id,
        bibtex_key=source.key,
        title=bib.title,
        authors=list(bib.authors),
        year=published.year,
        chunk_id=source.chunk_id,
        chunk_type=source.evidence,
        evidence=source.evidence,
        abs_url=abs_url(source.arxiv_id),
        cited_count=cited_count,
    )

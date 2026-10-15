"""Split a paper into chunks for embedding. Pure functions, no I/O (besides tiktoken's
cached tokenizer file).

Input is the CLEANED markdown (`paper_texts.markdown`, from clean.py). Output is a list
of `Chunk`s with deterministic ids, so re-running overwrites instead of duplicating.

How a paper is split:
1. Sections first: every heading starts a section; its path is the chain of headings
   above it ('3 Method > 3.1 Router').
2. A section whose text is shorter than `chunk_min_tokens` is merged into the section
   right after it (its next sibling, or its first subsection), keeping its heading line
   in the text. Merges do not chain (see `_merge_small_sections`).
3. A section that fits in `chunk_tokens` (512) becomes one chunk. A longer one is cut
   into windows of up to 512 tokens on paragraph boundaries, then sentence boundaries;
   each window repeats up to `chunk_overlap` (64) tokens from the end of the previous
   one, so a sentence near a cut is still found with its context.
4. Tables, code blocks and display math are never cut in the middle, unless one is
   larger than `chunk_hard_max_tokens`: then a table is split by rows with its header
   (and caption) repeated in every piece, and code or math is split by lines.
5. Each chunk's embedding text starts with 'Title > Section path', so a chunk carries
   its context even when it is retrieved alone.

Chunk ids: uuid5(NAMESPACE, '<arxiv_id>v<version>#<ord>'). ord 0 is the abstract chunk,
section chunks are numbered from 1, and analysis chunks (`analysis_chunks`) from 10000.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

from app.config import get_settings
from app.ingest.markdown import is_table_row, parse_heading, protected_lines, protected_spans
from app.ingest.tokens import count_tokens, split_by_tokens, tail_within

# Fixed for this project. Changing it changes every chunk id (= every Qdrant point id).
NAMESPACE = uuid.UUID("da2a306b-299a-49f3-9ba9-5a1724f60504")
ABSTRACT_ORD = 0
FIRST_SECTION_ORD = 1
FIRST_ANALYSIS_ORD = 10_000

ChunkType = Literal["abstract", "section", "analysis"]


def chunk_id(arxiv_id: str, version: int, ord: int) -> uuid.UUID:
    """Deterministic id: the same paper version and position always give the same id."""
    return uuid.uuid5(NAMESPACE, f"{arxiv_id}v{version}#{ord}")


@dataclass(frozen=True)
class ChunkSettings:
    tokens: int  # target size of a window inside a long section
    overlap: int  # tokens repeated from the previous window
    min_tokens: int  # smaller sections are merged into the next one
    hard_max: int  # largest table / code / math piece

    def __post_init__(self) -> None:
        if not 0 <= self.overlap < self.tokens <= self.hard_max:
            raise ValueError(f"need 0 <= overlap < tokens <= hard_max, got {self}")

    @classmethod
    def from_config(cls) -> "ChunkSettings":
        s = get_settings()
        return cls(s.chunk_tokens, s.chunk_overlap, s.chunk_min_tokens, s.chunk_hard_max_tokens)


@dataclass(frozen=True)
class Chunk:
    id: uuid.UUID
    ord: int
    chunk_type: ChunkType
    section_path: str | None  # 'Title' is not part of it; None for text before any heading
    text: str  # what is shown and cited (stored in Postgres and the Qdrant payload)
    token_count: int  # tokens in `text`
    embedding_text: str  # 'Title > Section path' + blank line + text; what gets embedded


def embedding_text(title: str, section_path: str | None, text: str) -> str:
    """What gets embedded: 'Title > Section path', a blank line, then the chunk text.
    Indexing rebuilds it from the stored row with this same function."""
    prefix = f"{title} > {section_path}" if section_path else title
    return f"{prefix}\n\n{text}"


def abstract_chunk(arxiv_id: str, version: int, title: str, abstract: str) -> Chunk:
    """The ord-0 chunk: title + abstract. It is also the paper-level vector."""
    text = abstract.strip()
    return Chunk(
        id=chunk_id(arxiv_id, version, ABSTRACT_ORD),
        ord=ABSTRACT_ORD,
        chunk_type="abstract",
        section_path="Abstract",
        text=text,
        token_count=count_tokens(text),
        embedding_text=embedding_text(title, "Abstract", text),
    )


# The analysis chunks of a paper, in ord order from FIRST_ANALYSIS_ORD: (section_path,
# fields of the analysis joined into its text). The TL;DR is too short to be a useful
# search hit alone, so it opens the summary chunk together with the contributions.
ANALYSIS_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Analysis > Summary", ("tldr", "contributions")),
    ("Analysis > Method", ("method",)),
    ("Analysis > Results", ("results",)),
    ("Analysis > Limitations", ("limitations",)),
)


def analysis_chunks(
    arxiv_id: str, version: int, title: str, fields: dict[str, str | None]
) -> list[Chunk]:
    """The 'analysis' chunks of a generated paper analysis (ord 10000, 10001, ...).
    The ord of each section is fixed, so a new analysis overwrites the old chunks.
    A section whose fields are all empty gets no chunk."""
    chunks: list[Chunk] = []
    for n, (path, names) in enumerate(ANALYSIS_SECTIONS):
        text = "\n\n".join(v.strip() for name in names if (v := fields.get(name)) and v.strip())
        if not text:
            continue
        ord = FIRST_ANALYSIS_ORD + n
        chunks.append(
            Chunk(
                id=chunk_id(arxiv_id, version, ord),
                ord=ord,
                chunk_type="analysis",
                section_path=path,
                text=text,
                token_count=count_tokens(text),
                embedding_text=embedding_text(title, path, text),
            )
        )
    return chunks


def chunk_paper(
    arxiv_id: str,
    version: int,
    title: str,
    markdown: str,
    settings: ChunkSettings | None = None,
) -> list[Chunk]:
    """Section chunks (ord 1, 2, ...) for a paper's cleaned markdown."""
    cfg = settings or ChunkSettings.from_config()
    chunks: list[Chunk] = []
    for section in _merge_small_sections(_sections(markdown), cfg.min_tokens):
        path = " > ".join(section.path) or None
        for text in _section_texts(section.lines, cfg):
            ord = FIRST_SECTION_ORD + len(chunks)
            if ord >= FIRST_ANALYSIS_ORD:
                raise ValueError(f"{arxiv_id}: more than {ord - 1} chunks, ids would collide")
            chunks.append(
                Chunk(
                    id=chunk_id(arxiv_id, version, ord),
                    ord=ord,
                    chunk_type="section",
                    section_path=path,
                    text=text,
                    token_count=count_tokens(text),
                    embedding_text=embedding_text(title, path, text),
                )
            )
    return chunks


# --- 1-2. sections ------------------------------------------------------------------------


@dataclass
class _Section:
    path: tuple[str, ...]  # heading texts from the top level down
    heading: str | None  # the heading line itself ('### 2.2 Decoder')
    lines: list[str] = field(default_factory=list)  # body, without the heading line


def _sections(markdown: str) -> list[_Section]:
    lines = markdown.split("\n")
    protected = protected_lines(lines)
    stack: list[tuple[int, str]] = []  # (level, text) of the headings above
    sections = [_Section(path=(), heading=None)]  # text before the first heading
    for line, keep in zip(lines, protected, strict=True):
        heading = None if keep else parse_heading(line)
        if heading is None:
            sections[-1].lines.append(line)
            continue
        level, text = heading
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, text))
        sections.append(_Section(path=tuple(t for _, t in stack), heading=line))
    return sections


def _merge_small_sections(sections: list[_Section], min_tokens: int) -> list[_Section]:
    """Merge each section shorter than `min_tokens` into the section right after it.

    Merges do not chain: the combined section is kept even if it is still short. So an
    empty '2 Method' plus '2.1 Router' becomes one chunk labelled '2 Method > 2.1 Router',
    and a run of tiny sections never ends up under the label of a section far away. The
    merged section's heading line stays in the text, so its title is still searchable.
    """
    # Text before the first heading is usually empty after cleaning: nothing to merge.
    sections = [s for s in sections if s.heading is not None or "\n".join(s.lines).strip()]
    merged: list[_Section] = []
    carry: list[str] = []  # the small section waiting to be prepended to the next one
    for i, section in enumerate(sections):
        small = count_tokens("\n".join(section.lines).strip()) < min_tokens
        if small and not carry and i < len(sections) - 1:
            carry = ([section.heading, ""] if section.heading is not None else []) + section.lines
            carry.append("")
            continue
        lines = carry + section.lines
        carry = []
        if "\n".join(lines).strip():
            merged.append(_Section(path=section.path, heading=section.heading, lines=lines))
    return merged


# --- 3-4. blocks, units and windows ---------------------------------------------------------

BlockKind = Literal["prose", "table", "code", "math"]
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_TABLE_CAPTION_RE = re.compile(r"^Table\s+\w+[.:]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass
class _Block:
    kind: BlockKind
    lines: list[str]
    head: int = 0  # leading lines repeated in every piece when split (table caption/header)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _blocks(lines: list[str]) -> list[_Block]:
    """Paragraphs, tables, code blocks and math blocks, in order."""
    spans = {start: (end, kind) for start, end, kind in protected_spans(lines)}
    blocks: list[_Block] = []
    para: list[str] = []

    def end_paragraph() -> None:
        if para:
            blocks.append(_Block("prose", para.copy()))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if i in spans:  # a code or math block, fences included
            end_paragraph()
            end, kind = spans[i]
            blocks.append(_Block("code" if kind == "code" else "math", lines[i:end]))
            i = end
        elif is_table_row(line):
            end_paragraph()
            j = i
            while j < len(lines) and j not in spans and is_table_row(lines[j]):
                j += 1
            rows = lines[i:j]
            head = 2 if len(rows) > 1 and _TABLE_SEPARATOR_RE.match(rows[1]) else 1
            blocks.append(_Block("table", rows, head=head))
            i = j
        else:
            if line.strip():
                para.append(line)
            else:
                end_paragraph()
            i += 1
    end_paragraph()
    return _attach_table_captions(blocks)


def _attach_table_captions(blocks: list[_Block]) -> list[_Block]:
    """'Table 2: ...' right before a table belongs to it (and is repeated with its header)."""
    out: list[_Block] = []
    for block in blocks:
        prev = out[-1] if out else None
        if (
            block.kind == "table"
            and prev is not None
            and prev.kind == "prose"
            and _TABLE_CAPTION_RE.match(prev.lines[0])
        ):
            out[-1] = _Block("table", prev.lines + block.lines, head=len(prev.lines) + block.head)
        else:
            out.append(block)
    return out


@dataclass(frozen=True)
class _Unit:
    """The smallest piece packed into windows: a paragraph, a sentence or a whole block."""

    text: str
    para: int  # units of the same paragraph are joined with a space, others with a blank line
    atomic: bool  # table / code / math (never used as overlap)


def _units(blocks: list[_Block], cfg: ChunkSettings) -> list[_Unit]:
    budget = cfg.tokens - cfg.overlap  # room left in a window after the overlap
    units: list[_Unit] = []
    for para, block in enumerate(blocks):
        if block.kind != "prose":
            for piece in _split_block(block, cfg.hard_max):
                units.append(_Unit(piece, para, atomic=True))
            continue
        text = " ".join(line.strip() for line in block.lines)
        if count_tokens(text) <= budget:
            units.append(_Unit(text, para, atomic=False))
            continue
        for sentence in _SENTENCE_END_RE.split(text):
            pieces = [sentence] if count_tokens(sentence) <= budget else []
            pieces = pieces or split_by_tokens(sentence, budget)
            units.extend(_Unit(p.strip(), para, atomic=False) for p in pieces if p.strip())
    return units


def _split_block(block: _Block, hard_max: int) -> list[str]:
    """A table/code/math block as one piece, or, if too big, pieces of at most `hard_max`
    tokens: table rows with the header repeated, code/math lines inside their fences."""
    if count_tokens(block.text) <= hard_max:
        return [block.text]
    if block.kind == "table":
        head, body, tail = block.lines[: block.head], block.lines[block.head :], []
    else:  # keep the opening and closing fence ('```' or '$$') around every piece
        head, body, tail = block.lines[:1], block.lines[1:-1], block.lines[-1:]
    room = hard_max - count_tokens("\n".join(head + tail)) - 2  # 2: the joining newlines
    # A caption + header that alone fill the cap would leave no room for rows; keep a
    # minimum so splitting still ends (those pieces would exceed the cap, but no real
    # table has a 2000-token header).
    room = max(room, 32)
    pieces: list[list[str]] = []
    current: list[str] = []
    for line in body:
        for part in [line] if count_tokens(line) <= room else split_by_tokens(line, room):
            if current and count_tokens("\n".join(current + [part])) > room:
                pieces.append(current)
                current = []
            current.append(part)
    if current:
        pieces.append(current)
    return ["\n".join(head + piece + tail) for piece in pieces]


def _join(units: list[_Unit]) -> str:
    out = ""
    for i, unit in enumerate(units):
        sep = "" if i == 0 else (" " if unit.para == units[i - 1].para else "\n\n")
        out += sep + unit.text
    return out


def _overlap(window: list[_Unit], overlap: int) -> list[_Unit]:
    """The end of a window to repeat at the start of the next, at most `overlap` tokens:
    whole trailing units (paragraphs or sentences) if they fit; otherwise the last
    paragraph's trailing sentences; only if even its last sentence is too long, its last
    words. Tables, code and math are never repeated."""
    tail: list[_Unit] = []
    for unit in reversed(window):
        if unit.atomic or count_tokens(_join([unit, *tail])) > overlap:
            break
        tail.insert(0, unit)
    if tail or not window or window[-1].atomic:
        return tail
    last = window[-1]
    sentences: list[str] = []
    for sentence in reversed(_SENTENCE_END_RE.split(last.text)):
        if count_tokens(" ".join([sentence, *sentences])) > overlap:
            break
        sentences.insert(0, sentence)
    text = " ".join(sentences) if sentences else tail_within(last.text, overlap)
    return [_Unit(text, last.para, atomic=False)] if text else []


def _section_texts(lines: list[str], cfg: ChunkSettings) -> list[str]:
    """Pack a section's units into windows of at most `cfg.tokens` tokens (a table, code
    or math piece larger than that gets a chunk of its own, up to `cfg.hard_max`)."""
    windows: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in _units(_blocks(lines), cfg):
        if unit.atomic and count_tokens(unit.text) > cfg.tokens:
            if current:
                windows.append(current)
            windows.append([unit])
            current = []
            continue
        if current and count_tokens(_join([*current, unit])) > cfg.tokens:
            windows.append(current)
            current = [*_overlap(current, cfg.overlap), unit]
            if count_tokens(_join(current)) > cfg.tokens:
                current = [unit]
        else:
            current.append(unit)
    if current:
        windows.append(current)
    return [text for window in windows if (text := _join(window).strip())]

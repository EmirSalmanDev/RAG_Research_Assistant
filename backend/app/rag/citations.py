"""Numbered sources and the [n] citations that point at them (used by generate and verify).

Generate shows the model the retrieved chunks as sources [1]..[n]; the answer cites them
as "[2]" or "[1][3]" (also "[1, 3]"). This module formats those sources as data blocks,
finds the citations in an answer, splits the answer into sentences for verify, and
builds the citation objects stored on the message.

Evidence levels: every source says which kind of chunk it is (`chunk_type`): `section`
(full text), `abstract` (the abstract only) or `analysis` (a generated summary). An
answer whose cited sources are all abstracts can only be as detailed as the abstracts;
`abstract_only` says whether that is the case.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.rag.prompts import escape_block
from app.rag.search import SearchHit

# Answers link to the arXiv abstract page, never to a PDF (we do not re-serve papers).
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"

_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_LEADING_CITATIONS = re.compile(r"^((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)")
# A possible sentence end: '.', '!' or '?' (optionally followed by a closing quote or
# bracket) and then whitespace. Whether it really ends a sentence is decided in
# _is_sentence_end.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+")
# A list or heading marker at the start of a line: "1.", "2)", "-", "*", "+", "•", "#".
_LINE_MARKER = re.compile(r"^(?:\d{1,3}[.)]|[-*+•]|#{1,6}|>)\s+")
# Words that end with a period without ending the sentence (compared in lower case,
# without their final period).
_ABBREVIATIONS = frozenset({
    "e.g", "i.e", "et al", "al", "cf", "vs", "etc", "approx", "fig", "figs", "eq", "eqs",
    "sec", "secs", "tab", "ref", "refs", "no", "nos", "vol", "pp", "dr", "mr", "ms", "prof",
    "resp", "ca", "incl", "w.r.t",
})  # fmt: skip


@dataclass(frozen=True)
class Source:
    n: int  # 1-based number shown to the model and cited as [n]
    hit: SearchHit


def number_sources(hits: Sequence[SearchHit]) -> list[Source]:
    return [Source(i, hit) for i, hit in enumerate(hits, start=1)]


def format_source(source: Source) -> str:
    """One source as a data block. Title, section and text come from papers: escaped."""
    h = source.hit
    return (
        f'<source n="{source.n}" evidence="{h.chunk_type}">\n'
        f"Title: {escape_block(h.title)}\n"
        f"arXiv: {h.arxiv_id}\n"
        f"Section: {escape_block(h.section_path or 'Abstract')}\n"
        f"Evidence: {h.chunk_type}\n"
        f"Text:\n{escape_block(h.text)}\n"
        f"</source>"
    )


def format_sources(sources: Sequence[Source]) -> str:
    return "<sources>\n" + "\n\n".join(format_source(s) for s in sources) + "\n</sources>"


def cited_numbers(text: str) -> list[int]:
    """Every number cited in `text`, in order of first appearance."""
    found: dict[int, None] = {}
    for match in _CITATION.finditer(text):
        for part in re.split(r"[,;]", match[1]):
            found[int(part)] = None
    return list(found)


@dataclass(frozen=True)
class Sentence:
    index: int
    text: str
    citations: tuple[int, ...]


def _is_sentence_end(text: str, match: re.Match[str]) -> bool:
    """Is this '.', '!' or '?' + whitespace a real sentence end?

    Not after an abbreviation ("e.g.", "et al.", "Fig.") or a single capital letter (an
    initial, "A. Vaswani"), and not when the next word starts in lower case. Decimals
    ("0.9 BLEU") never get here: the point is not followed by whitespace.
    """
    if match.group().startswith("."):
        before = text[: match.start()]
        word = before.rsplit(None, 1)[-1] if before.strip() else ""
        word = word.lstrip("([\"'").lower()
        if word in _ABBREVIATIONS or re.fullmatch(r"[a-z]", word):  # "et al." ends in "al"
            return False
    rest = text[match.end() :]
    return not (rest[:1].islower())


def _split_line(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if _is_sentence_end(text, match):
            pieces.append(text[start : match.end()].strip())
            start = match.end()
    pieces.append(text[start:].strip())
    return [p for p in pieces if p]


def split_sentences(answer: str) -> list[Sentence]:
    """The answer's sentences, each with the numbers it cites.

    Line by line; a list or heading marker at the start of a line ("1.", "-", "##") stays
    with the line's first sentence and is never a sentence of its own. Inside a line, a
    sentence ends at '.', '!' or '?' plus whitespace, except after an abbreviation, after
    an initial, or before a lower-case word (see _is_sentence_end). Citations that start
    the next piece ("... model. [2] The next ...") belong to the sentence before.
    """
    pieces: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        marker_match = _LINE_MARKER.match(line)
        marker = marker_match.group() if marker_match else ""
        body = line[len(marker) :]
        for k, piece in enumerate(_split_line(body)):
            lead = _LEADING_CITATIONS.match(piece)
            if lead and pieces and not (k == 0 and marker):
                pieces[-1] = f"{pieces[-1]} {lead[1].strip()}"
                piece = piece[lead.end() :].strip()
            if k == 0 and marker:
                piece = f"{marker}{piece}"
            if piece.strip(" -*#>") and re.search(r"[A-Za-z]", piece):
                pieces.append(piece.strip())
    return [Sentence(i, p, tuple(cited_numbers(p))) for i, p in enumerate(pieces)]


def abstract_only(sources: Sequence[Source], cited: Sequence[int]) -> bool:
    """True when the answer cites at least one source and every cited source that exists
    is an abstract chunk (then the answer rests on abstracts alone)."""
    by_n = {s.n: s for s in sources}
    kinds = [by_n[n].hit.chunk_type for n in cited if n in by_n]
    return bool(kinds) and all(k == "abstract" for k in kinds)


def citation_objects(sources: Sequence[Source], cited: Sequence[int]) -> list[dict[str, Any]]:
    """What `messages.citations` stores: one object per cited source that exists."""
    by_n = {s.n: s for s in sources}
    return [source_object(by_n[n]) for n in cited if n in by_n]


def source_object(source: Source) -> dict[str, Any]:
    h = source.hit
    return {
        "n": source.n,
        "chunk_id": str(h.chunk_id),
        "arxiv_id": h.arxiv_id,
        "title": h.title,
        "section_path": h.section_path,
        "chunk_type": h.chunk_type,
        "published_at": h.published_at,
        "abs_url": ARXIV_ABS_URL.format(arxiv_id=h.arxiv_id),
    }

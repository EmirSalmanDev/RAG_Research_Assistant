"""Clean a paper's full-text markdown before chunking. Pure functions, no I/O.

Input is `paper_texts.raw_markdown` (as fetched or converted); the result's `markdown`
goes to `paper_texts.markdown` and `references` to `paper_texts.references`.

What cleaning does, in order:
1. normalise characters and whitespace (line endings, non-breaking spaces, PDF
   ligatures such as 'ﬁ' -> 'fi'); HTML comments are removed;
2. remove image links and converter picture placeholders, but keep caption lines
   ('Figure 2: ...');
3. remove PDF artifacts: the vertical arXiv stamp, page numbers, page separators
   ('-----') and running headers (short lines repeated on many pages);
4. re-join words hyphenated at a line break ('trans-' + 'former');
5. drop the front matter (title, authors, affiliations) and the Abstract section,
   because the abstract is indexed separately as its own chunk;
6. cut the reference list out into its own return value; appendices that follow the
   references stay in the body;
7. normalise heading levels from section numbers ('3' -> ##, '3.1' -> ###), so the
   heading tree is consistent whichever converter produced the text.

Fenced code and display math keep their content (only trailing spaces are trimmed).
Cleaning is idempotent: cleaning the cleaned markdown again returns it unchanged.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from app.ingest.markdown import is_table_row, parse_heading, protected_lines


@dataclass(frozen=True)
class CleanedText:
    markdown: str  # cleaned body: headings, paragraphs, tables, captions, appendices
    references: str | None  # the cut-out reference list, None if none was found


def clean_markdown(raw: str) -> CleanedText:
    text = normalize_characters(raw)
    lines = text.split("\n")
    lines = remove_images(lines)
    lines = remove_page_artifacts(lines)
    lines = join_hyphenated_words(lines)
    lines = drop_front_matter(lines)
    body, references, appendix = split_references(lines)
    lines = normalize_heading_levels(body, appendix)
    refs = _tidy(references) if references is not None else ""
    return CleanedText(markdown=_tidy(lines), references=refs or None)


# --- 1. characters --------------------------------------------------------------------

_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\ufeff"))  # soft hyphen etc.


def normalize_characters(text: str) -> str:
    # NFC only joins accents with their letters. NFKC would also turn math such as
    # 'x²' into 'x2', so ligatures are replaced one by one instead.
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    for ligature, letters in _LIGATURES.items():
        text = text.replace(ligature, letters)
    text = text.translate(_INVISIBLE).replace("\u00a0", " ")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return "\n".join(line.rstrip() for line in text.split("\n"))


# --- 2. images --------------------------------------------------------------------------

_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>", re.IGNORECASE)
_PICTURE_PLACEHOLDER_RE = re.compile(r"==>\s*picture\b.*?<==", re.IGNORECASE)


def remove_images(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line, protected in zip(lines, protected_lines(lines), strict=True):
        if not protected:
            stripped = _PICTURE_PLACEHOLDER_RE.sub("", _IMAGE_RE.sub("", line))
            if stripped.strip(" *") == "" and line.strip():
                continue  # the line held only an image: drop it entirely
            line = stripped.rstrip()
        out.append(line)
    return out


# --- 3. PDF page artifacts --------------------------------------------------------------

_ARXIV_STAMP_RE = re.compile(
    r"^arXiv:\s?(\d{4}\.\d{4,5}|[a-z-]+(\.[A-Z]{2})?/\d{7})(v\d+)?"
    r"(\s*\[[^\]]+\])?(\s+\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})?$"
)
_PAGE_NUMBER_RE = re.compile(r"^(\d{1,4}|-\s*\d{1,4}\s*-|page\s+\d+(\s+of\s+\d+)?)$", re.I)
_RULE_RE = re.compile(r"^([-*_])\1{2,}$")  # '-----' page separators / horizontal rules
RUNNING_HEADER_MIN_REPEATS = 3
RUNNING_HEADER_CHARS = (10, 100)  # a running header is at least 10, at most 100 characters
PDF_MIN_PAGE_ARTIFACTS = 2  # page numbers/separators that mark the text as PDF output


def remove_page_artifacts(lines: list[str]) -> list[str]:
    protected = protected_lines(lines)

    def candidate(i: int) -> bool:
        line = lines[i].strip()
        return (
            bool(line)
            and not protected[i]
            and not is_table_row(line)
            and parse_heading(lines[i]) is None
        )

    def page_artifact(line: str) -> bool:
        return bool(
            _ARXIV_STAMP_RE.match(line) or _PAGE_NUMBER_RE.match(line) or _RULE_RE.match(line)
        )

    artifacts = {i for i in range(len(lines)) if candidate(i) and page_artifact(lines[i].strip())}
    # A running header/footer is an identical line repeated on many pages (e.g. the
    # conference name). Only PDF output has pages, so this check runs only when page
    # artifacts were found; short lines such as 'Proof.' never count.
    repeated: set[str] = set()
    if len(artifacts) >= PDF_MIN_PAGE_ARTIFACTS:
        low, high = RUNNING_HEADER_CHARS
        counts = Counter(
            lines[i].strip()
            for i in range(len(lines))
            if candidate(i) and low <= len(lines[i].strip()) <= high
        )
        repeated = {line for line, n in counts.items() if n >= RUNNING_HEADER_MIN_REPEATS}
    return [
        line
        for i, line in enumerate(lines)
        if i not in artifacts and not (candidate(i) and line.strip() in repeated)
    ]


# --- 4. hyphenation ---------------------------------------------------------------------

_HYPHEN_END_RE = re.compile(r"[a-z]{2,}-$")
_LOWER_START_RE = re.compile(r"^[a-z]")


def join_hyphenated_words(lines: list[str]) -> list[str]:
    """'trans-' at the end of a line + 'former is ...' -> 'transformer is ...'.

    Only for prose: both lines must be ordinary text (no heading, table, code or math).
    Trade-off: a compound split exactly at its own hyphen ('state-' + 'of-the-art')
    loses that hyphen, which barely affects search.
    """
    protected = protected_lines(lines)

    def prose(i: int) -> bool:
        return not protected[i] and not is_table_row(lines[i]) and parse_heading(lines[i]) is None

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        while (
            i + 1 < len(lines)
            and prose(i)
            and prose(i + 1)
            and _HYPHEN_END_RE.search(line)
            and _LOWER_START_RE.match(lines[i + 1].lstrip())
        ):
            line = line[:-1] + lines[i + 1].lstrip()
            i += 1
        out.append(line)
        i += 1
    return out


# --- 5. front matter and abstract -------------------------------------------------------

_ABSTRACT_RE = re.compile(r"^(abstract)[.:\u2014\u2013-]?$", re.I)
_ABSTRACT_PARAGRAPH_RE = re.compile(r"^(\*\*|__)?abstract(\*\*|__)?\s*[.:\u2014\u2013-]\s*\S", re.I)
_INTRO_RE = re.compile(r"^((\d+|[IVX]+)\.?\s+)?introduction$", re.I)


def _label(line: str) -> str:
    """A heading's text, or a plain line without bold/italic markers:
    '## Abstract', '**Abstract**' and 'Abstract' all give 'Abstract'."""
    heading = parse_heading(line)
    return heading[1] if heading else re.sub(r"(\*\*|__|\*|_)", "", line).strip()


def drop_front_matter(lines: list[str]) -> list[str]:
    """Keep the text from the first section after the abstract.

    - With an abstract (heading '## Abstract', a line 'Abstract' or '**Abstract**', or a
      paragraph starting 'Abstract:'), the body starts at the first heading after it.
    - Without one, it starts at the 'Introduction' heading, or at a heading numbered '1'.
    - If none of these is found nothing is dropped except a leading '# Title' line, so
      an unusual layout loses no content.
    """
    protected = protected_lines(lines)
    headings = [
        i for i, line in enumerate(lines) if not protected[i] and parse_heading(line) is not None
    ]
    abstract_at = next(
        (
            i
            for i, line in enumerate(lines)
            if not protected[i]
            and (_ABSTRACT_RE.match(_label(line)) or _ABSTRACT_PARAGRAPH_RE.match(line.strip()))
        ),
        None,
    )
    if abstract_at is not None:
        start = next((i for i in headings if i > abstract_at), len(lines))
        return lines[start:]
    for i in headings:
        heading = parse_heading(lines[i])
        assert heading is not None
        if _INTRO_RE.match(heading[1]) or re.match(r"^1\.?\s+\S", heading[1]):
            return lines[i:]
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is not None and (h := parse_heading(lines[first])) is not None and h[0] == 1:
        return lines[first + 1 :]  # only a title heading
    return lines


# --- 6. references ----------------------------------------------------------------------

_REFERENCES_RE = re.compile(
    r"^((\d+|[IVX]+)\.?\s+)?(references|bibliography|references and notes|literature cited)$",
    re.I,
)


def split_references(lines: list[str]) -> tuple[list[str], list[str] | None, list[str]]:
    """Return (body before the references, the reference lines or None, what follows).

    The references start at a heading (or a lone 'References' / '**References**' line,
    as PDF converters write it) and end at the next heading: in a paper that is an
    appendix or supplementary section, which is kept.
    """
    protected = protected_lines(lines)
    for i, line in enumerate(lines):
        if protected[i]:
            continue
        if not _REFERENCES_RE.match(_label(line)):
            continue
        end = next(
            (
                j
                for j in range(i + 1, len(lines))
                if not protected[j] and parse_heading(lines[j]) is not None
            ),
            len(lines),
        )
        return lines[:i], lines[i + 1 : end], lines[end:]
    return lines, None, []


# --- 7. heading levels ------------------------------------------------------------------

# Section numbers have 1-2 digits, so a heading such as '2023 results' is not numbered.
_NUMBERED_RE = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2})*)\.?\s+\S")
# Case-sensitive letter: 'a note on ...' is not appendix section 'a'.
_LETTERED_RE = re.compile(r"^(?:(?i:appendix)\s+)?(?P<num>[A-Z](?:\.\d{1,2})*)[.:]?\s+\S")
_ROMAN_RE = re.compile(r"^(?P<num>[IVX]{1,4})\.\s+\S")
_APPENDIX_RE = re.compile(r"^(appendix|appendices|supplementary)\b", re.I)


def _section_depth(text: str, in_appendix: bool) -> int | None:
    """How deep a numbered heading is ('3' -> 1, '3.1' -> 2, 'A.2' -> 2); None if unnumbered.

    A single letter ('A Proofs') only counts as a number in the appendix, so a body
    heading such as 'A Simple Baseline' is not mistaken for an appendix section.
    """
    if m := _NUMBERED_RE.match(text):
        return m.group("num").count(".") + 1
    if m := _ROMAN_RE.match(text):
        return 1
    m = _LETTERED_RE.match(text)
    if m and (in_appendix or "." in m.group("num") or text.lower().startswith("appendix")):
        return m.group("num").count(".") + 1
    return None


def normalize_heading_levels(body: list[str], appendix: list[str]) -> list[str]:
    """Rewrite headings so level = section depth + 1 (## for '1', ### for '1.1', ...).

    Unnumbered headings keep their level relative to the shallowest heading in the
    text, which becomes ##: a paper whose sections are '### 1 Intro' and '#### Details'
    ends up with '## 1 Intro' and '### Details'. Level # is not used in the body (it was
    the paper title).
    """
    lines = body + appendix
    protected = protected_lines(lines)
    found = [
        (i, h) for i, line in enumerate(lines) if not protected[i] and (h := parse_heading(line))
    ]
    shift = min((level for _, (level, _) in found), default=2) - 2

    in_appendix = False
    out = list(lines)
    for i, (level, text) in found:
        in_appendix = in_appendix or i >= len(body) or bool(_APPENDIX_RE.match(text))
        depth = _section_depth(text, in_appendix)
        new_level = depth + 1 if depth is not None else level - shift
        out[i] = "#" * min(6, max(2, new_level)) + " " + text
    return out


def _tidy(lines: list[str]) -> str:
    """At most one blank line between blocks (code and math untouched); no blank lines
    at the start or the end."""
    protected = protected_lines(lines)
    out: list[str] = []
    for line, keep in zip(lines, protected, strict=True):
        if not keep and not line.strip() and (not out or not out[-1].strip()):
            continue  # a second blank line in a row
        out.append(line)
    return "\n".join(out).strip("\n")

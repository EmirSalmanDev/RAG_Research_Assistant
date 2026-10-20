"""BibTeX entries built from stored metadata: pure functions, no LLM, no network.

    @misc{vaswani2017attention,
      author = {Vaswani, Ashish and Shazeer, Noam},
      title = {{Attention Is All You Need}},
      year = {2017},
      eprint = {1706.03762},
      archivePrefix = {arXiv},
      primaryClass = {cs.CL},
      url = {https://arxiv.org/abs/1706.03762},
    }

Rules (docs/adr/0011):
- Key: the first author's last name, the year and the first significant word of the title,
  folded to lower-case ASCII (`vaswani2017attention`). A key that is already taken gets
  `a`, `b`, `c`, ... (`vaswani2017attentiona`): the first paper in the given order keeps the
  plain key. The same papers in the same order always give the same keys.
- Entry type `@misc` (these are preprints). The title is wrapped in a second pair of braces
  so BibTeX styles keep its capitals; LaTeX special characters are escaped (math in a title
  therefore appears as text: a stray `$` would break the document).
- arXiv papers carry `eprint`, `archivePrefix`, `primaryClass` and a link to the abstract
  page (never to a PDF). An uploaded document has no arXiv page: it gets
  `note = {Uploaded document}` and no `eprint` and no `url`.
- Unicode letters in names and titles are kept as they are (the file is UTF-8); only the key
  is ASCII-folded.
- A paper without authors is `Anonymous`.
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.ids import abs_url, is_local_id

ANONYMOUS = "Anonymous"
UPLOAD_NOTE = "Uploaded document"
FALLBACK_KEY_PART = "anon"  # a name that folds to nothing (for example Chinese characters)
FALLBACK_TITLE_WORD = "untitled"

# Words that never make a useful part of a key.
TITLE_STOPWORDS = frozenset(
    "a an the of on in for to and or with from by at is are as via using towards toward into "
    "over under about how what why do does we our your its it this that these those be can "
    "not no vs versus".split()
)  # fmt: skip

# Lower-case particles that belong to the last name: "Ludwig van Beethoven".
NAME_PARTICLES = frozenset(
    "van von de del della der den di da dos das do du la le bin ibn al el ter ten zu".split()
)  # fmt: skip
NAME_SUFFIXES = frozenset({"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"})

# Letters that Unicode decomposition does not split into base letter + accent.
_FOLD_EXTRA = str.maketrans(
    {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "đ": "d", "ł": "l", "ð": "d", "þ": "th", "ı": "i"}
)
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_SPECIAL = re.compile("[" + re.escape("".join(_LATEX_ESCAPES)) + "]")


@dataclass(frozen=True)
class BibPaper:
    """What an entry needs from a paper."""

    arxiv_id: str  # an arXiv id, or a `local-` id for an uploaded document
    title: str
    authors: Sequence[str]  # "First Last" or "Last, First", arXiv's order
    published_at: datetime
    primary_category: str | None = None


@dataclass(frozen=True)
class BibEntry:
    key: str
    arxiv_id: str
    fields: tuple[tuple[str, str], ...]  # (name, already-escaped value), in output order

    def render(self) -> str:
        body = "".join(f"  {name} = {{{value}}},\n" for name, value in self.fields)
        return f"@misc{{{self.key},\n{body}}}\n"


# --- names ------------------------------------------------------------------------------------


def author_names(authors: Iterable[object]) -> list[str]:
    """`papers.authors` holds [{"name": ...}] objects; the names, in order."""
    names: list[str] = []
    for author in authors:
        name = author.get("name") if isinstance(author, dict) else author
        if isinstance(name, str) and name.strip():
            names.append(" ".join(name.split()))
    return names


def split_name(name: str) -> tuple[str, str, str]:
    """(last, suffix, first) of a person's name. "Last, First" is taken as written;
    otherwise the last word is the last name, plus the lower-case particles before it
    ("van", "de", ...). A single word is a last name with no first name."""
    name = " ".join(name.split())
    if "," in name:
        last, _, first = name.partition(",")
        return last.strip(), "", first.strip()
    words = name.split(" ")
    suffix = ""
    if len(words) > 1 and words[-1].lower() in NAME_SUFFIXES:
        suffix = words.pop()
    start = len(words) - 1
    while start > 0 and words[start - 1] in NAME_PARTICLES:  # particles are written lower case
        start -= 1
    return " ".join(words[start:]), suffix, " ".join(words[:start])


def format_author(name: str) -> str:
    """One author as BibTeX wants it: "Last, First" ("Last, Jr., First" with a suffix)."""
    last, suffix, first = split_name(name)
    if not last and not first:
        return ""
    parts = [escape_latex(last)]
    if suffix:
        parts.append(escape_latex(suffix))
    if first:
        parts.append(escape_latex(first))
    text = ", ".join(parts)
    # A name that contains the word "and" would be read as two authors: brace it.
    return "{" + text + "}" if re.search(r"\band\b", text, re.IGNORECASE) else text


def format_authors(authors: Sequence[str]) -> str:
    formatted = [a for a in (format_author(n) for n in authors) if a]
    return " and ".join(formatted) if formatted else ANONYMOUS


# --- text -------------------------------------------------------------------------------------


def escape_latex(text: str) -> str:
    """`text` on one line, with the characters LaTeX treats specially escaped."""
    flat = " ".join(text.split())
    return _LATEX_SPECIAL.sub(lambda m: _LATEX_ESCAPES[m.group()], flat)


def fold_ascii(text: str) -> str:
    """Lower-case ASCII letters and digits only: "Müller-Ødegård" -> "mullerodegard"."""
    return fold_words(text).replace(" ", "")


def fold_words(text: str) -> str:
    """Lower-case ASCII with every other run of characters turned into one space."""
    spread = text.translate(_FOLD_EXTRA).lower().translate(_FOLD_EXTRA)
    ascii_text = unicodedata.normalize("NFKD", spread).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def title_word(title: str) -> str:
    """The first significant word of a title, folded ("Attention Is All You Need" ->
    "attention"). Falls back to the first word, then to "untitled"."""
    words = [fold_ascii(w) for w in re.split(r"[\s\-:/]+", title)]
    words = [w for w in words if w]
    for word in words:
        if word not in TITLE_STOPWORDS:
            return word
    return words[0] if words else FALLBACK_TITLE_WORD


def base_key(paper: BibPaper) -> str:
    last = FALLBACK_KEY_PART
    if paper.authors:
        folded = fold_ascii(split_name(paper.authors[0])[0])
        last = folded or FALLBACK_KEY_PART
    return f"{last}{paper.published_at.year}{title_word(paper.title)}"


def _suffix(n: int) -> str:
    """0 -> "a", 1 -> "b", ..., 25 -> "z", 26 -> "aa" (a bijective base-26 counter)."""
    letters = ""
    n += 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(ord("a") + rem) + letters
    return letters


def assign_keys(papers: Sequence[BibPaper]) -> list[str]:
    """One unique key per paper, in the order of `papers`."""
    taken: set[str] = set()
    keys: list[str] = []
    for paper in papers:
        key = base_key(paper)
        n = 0
        while key in taken:
            key = f"{base_key(paper)}{_suffix(n)}"
            n += 1
        taken.add(key)
        keys.append(key)
    return keys


# --- entries and the file ---------------------------------------------------------------------


def build_entries(papers: Sequence[BibPaper]) -> list[BibEntry]:
    """The entries of `papers`, in the same order (the caller fixes the order, so the keys
    are the same every time for the same collection)."""
    entries: list[BibEntry] = []
    for paper, key in zip(papers, assign_keys(papers), strict=True):
        fields: list[tuple[str, str]] = [
            ("author", format_authors(paper.authors)),
            ("title", "{" + escape_latex(paper.title) + "}"),
            ("year", str(paper.published_at.year)),
        ]
        if is_local_id(paper.arxiv_id):
            fields.append(("note", UPLOAD_NOTE))
        else:
            fields += [("eprint", paper.arxiv_id), ("archivePrefix", "arXiv")]
            if paper.primary_category:
                fields.append(("primaryClass", paper.primary_category))
            fields.append(("url", abs_url(paper.arxiv_id) or ""))
        entries.append(BibEntry(key=key, arxiv_id=paper.arxiv_id, fields=tuple(fields)))
    return entries


def render_bibtex(entries: Sequence[BibEntry]) -> str:
    """The .bib file: entries separated by a blank line, ending in a newline."""
    return "\n".join(e.render() for e in entries)


def order_papers(papers: Iterable[BibPaper]) -> list[BibPaper]:
    """The one stable order used for the .bib file and the review: oldest first, ties by id."""
    return sorted(papers, key=lambda p: (p.published_at, p.arxiv_id))


def export_filename(collection: str) -> str:
    """A safe file name for a collection's .bib ("My thesis" -> "my-thesis.bib")."""
    slug = fold_words(collection).replace(" ", "-")
    return f"{slug or 'library'}.bib"

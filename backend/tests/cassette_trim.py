"""Trim a recorded arXiv HTML page to a short LaTeXML excerpt before it is written to a
cassette (CLAUDE.md rule 14: never full paper text in the repo).

The excerpt keeps arXiv's real LaTeXML markup (tags and class names), so the HTML
converter is tested on the real structure, but only a few elements: title, abstract,
the first two sections (first paragraph each, one subsection), one display equation,
one figure caption, one table (header and two rows), two references and the first
appendix. At most 60 lines, with a header comment naming the source and licence.

Real text is kept only if the page states a permissive licence (CC BY, CC BY-SA, CC0).
Otherwise every text is replaced with invented text and only the structure remains.
"""

import re
from datetime import date
from typing import Any

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

MAX_LINES = 60
MAX_PARAGRAPH_CHARS = 400
PERMISSIVE = re.compile(
    r"\b(CC[ -]?BY(?![ -]?N[CD])(?:[ -]?SA)?|CC0"
    r"|creativecommons\.org/(licenses/by/|licenses/by-sa/|publicdomain/zero))",
    re.I,
)
NON_PERMISSIVE = re.compile(r"(by-nc|by-nd|BY[ -]?N[CD]|non-exclusive)", re.I)


def page_licence(html: str) -> str:
    """The licence as the page states it ('License: ...' text or a creativecommons link)."""
    soup = BeautifulSoup(html, "lxml")
    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        if "creativecommons.org" in href or "license" in href.lower():
            text = " ".join(link.get_text(" ").split())
            if text:
                return f"{text} ({href})"
    match = re.search(r"License:\s*([^<\n]{3,80})", html)
    return match.group(1).strip() if match else "unknown"


def is_permissive(licence: str) -> bool:
    return bool(PERMISSIVE.search(licence)) and not NON_PERMISSIVE.search(licence)


def _first(node: Tag | None, *args: Any, **kwargs: Any) -> Tag | None:
    found = node.find(*args, **kwargs) if node is not None else None
    return found if isinstance(found, Tag) else None


def _short(tag: Tag) -> str:
    """A paragraph's HTML, cut after MAX_PARAGRAPH_CHARS of text (inline math kept)."""
    total = 0
    for node in list(tag.descendants):
        if (
            isinstance(node, NavigableString)
            and node.parent is not None
            and node.parent.name != "math"
        ):
            if total >= MAX_PARAGRAPH_CHARS:
                node.replace_with("")
            total += len(node)
    for annotation in tag.find_all(["annotation", "annotation-xml", "semantics"]):
        annotation.decompose()  # MathML internals: the alttext attribute is enough
    return " ".join(str(tag).split())


# Numbering that is structure, not content: '1', '1.2', 'Appendix A', 'A.1', 'Figure 1:',
# 'Table 2:', '[3]'. Author-year labels such as 'Smith (2024)' are content and get replaced.
_NUMBERING = re.compile(
    r"^\s*(\d+(\.\d+)*\.?|Appendix [A-Z]|[A-Z](\.\d+)+|Figure \d+:?|Table \d+:?|\[\d+\])\s*$"
)


def _invent(tag: Tag, n: int) -> None:
    """Replace every text (and math alttext) in `tag` with invented text, keeping the tags
    and only numbering text; external links and alt/title texts are removed too."""
    for math in tag.find_all("math"):
        math["alttext"] = "x^{2}"
        math.clear()
    for node in list(tag.descendants):
        if isinstance(node, NavigableString) and node.strip():
            if _NUMBERING.match(node):
                continue  # section/figure/table numbers are structure
            node.replace_with(f"Invented text {n}.")
    _strip_attributes(tag)


def _strip_attributes(tag: Tag) -> None:
    for element in [tag, *tag.find_all(True)]:
        for name in ("alt", "title", "src"):
            if name in element.attrs:
                del element.attrs[name]
        href = element.get("href")
        if isinstance(href, str) and not href.startswith("#"):
            element["href"] = "#"


def structure_only(fragment: str) -> str:
    """Invent every text of an already-trimmed excerpt (used for non-permissive pages)."""
    excerpt = BeautifulSoup(f"<article>{fragment}</article>", "lxml")
    container = _first(excerpt, "article")
    assert container is not None
    for n, element in enumerate(container.find_all(_INVENTED_ELEMENTS), 1):
        _invent(element, n)
    for node in list(container.descendants):  # any text outside those elements
        if isinstance(node, NavigableString) and node.strip() and not _NUMBERING.match(node):
            if not re.fullmatch(r"Invented text \d+\.", node.strip()):
                node.replace_with("Invented text.")
    _strip_attributes(container)
    return "\n".join(" ".join(str(c).split()) for c in container.children if str(c).strip())


_INVENTED_ELEMENTS = ["h1", "h2", "h3", "h6", "p", "li", "tr", "figcaption", "span", "a", "cite"]


def trim_arxiv_html(html: str, source_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    article = _first(soup, class_="ltx_document")
    if article is None:
        return html if len(html) < 2000 else "<html><body>no LaTeXML document</body></html>"
    licence = page_licence(html)
    keep_text = is_permissive(licence)

    lines: list[str] = []

    def add(tag: Tag | None) -> None:
        if tag is not None:
            lines.append(_short(tag))

    add(_first(article, "h1", class_="ltx_title_document"))
    abstract = _first(article, class_="ltx_abstract")
    if abstract is not None:
        lines.append('<div class="ltx_abstract">')
        add(_first(abstract, class_="ltx_title_abstract"))
        add(_first(abstract, "p"))
        lines.append("</div>")
    for section in article.find_all("section", class_="ltx_section", recursive=True)[:2]:
        lines.append(f'<section class="ltx_section" id="{section.get("id", "")}">')
        add(_first(section, class_="ltx_title_section"))
        add(_first(section, "p", class_="ltx_p"))
        sub = _first(section, "section", class_="ltx_subsection")
        if sub is not None:
            lines.append('<section class="ltx_subsection">')
            add(_first(sub, class_="ltx_title_subsection"))
            add(_first(sub, "p", class_="ltx_p"))
            lines.append("</section>")
        lines.append("</section>")
    equation = _first(article, "table", class_="ltx_equation")
    math = _first(equation, "math")
    if math is not None:
        lines.append('<table class="ltx_equation ltx_eqn_table"><tr><td>')
        add(math)
        lines.append("</td></tr></table>")
    figure = _first(article, "figure", class_="ltx_figure")
    caption = _first(figure, "figcaption")
    if caption is not None:
        lines.append('<figure class="ltx_figure">')
        add(caption)
        lines.append("</figure>")
    table_figure = _first(article, "figure", class_="ltx_table")
    tabular = _first(table_figure, "table", class_="ltx_tabular")
    if table_figure is not None and tabular is not None:
        lines.append('<figure class="ltx_table">')
        add(_first(table_figure, "figcaption"))
        lines.append('<table class="ltx_tabular">')
        for row in tabular.find_all("tr")[:3]:
            add(row)
        lines.append("</table></figure>")
    bibliography = _first(article, "section", class_="ltx_bibliography")
    if bibliography is not None:
        lines.append('<section class="ltx_bibliography" id="bib">')
        add(_first(bibliography, class_="ltx_title_bibliography"))
        lines.append('<ul class="ltx_biblist">')
        for item in bibliography.find_all("li", class_="ltx_bibitem")[:2]:
            add(item)
        lines.append("</ul></section>")
    appendix = _first(article, "section", class_="ltx_appendix")
    if appendix is not None:
        lines.append('<section class="ltx_appendix">')
        add(_first(appendix, class_="ltx_title_appendix"))
        add(_first(appendix, "p", class_="ltx_p"))
        lines.append("</section>")

    body = "\n".join(lines)
    if not keep_text:
        body = structure_only(body)
    header = (
        f"<!-- Trimmed excerpt of {source_url}, recorded {date.today().isoformat()} for tests. "
        f"Licence stated on the page: {licence}. "
        + (
            "Text kept (permissive licence)."
            if keep_text
            else "Text replaced with invented text; structure only."
        )
        + " -->"
    )
    out = "\n".join(
        [header, '<html><body><article class="ltx_document">', body, "</article></body></html>"]
    )
    if out.count("\n") + 1 > MAX_LINES:
        raise RuntimeError(f"trimmed excerpt has {out.count(chr(10)) + 1} lines (max {MAX_LINES})")
    return out

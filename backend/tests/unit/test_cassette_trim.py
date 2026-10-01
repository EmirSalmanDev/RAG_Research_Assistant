"""The cassette recorder's HTML trimming (rule 14), on an invented LaTeXML page."""

from typing import Any

import pytest

from app.ingest.clean import clean_markdown
from app.ingest.fulltext import html_to_markdown
from tests.cassette_trim import MAX_LINES, is_permissive, trim_arxiv_html
from tests.conftest import before_record_request, before_record_response
from tests.unit.test_fulltext_convert import LATEXML

URL = "https://arxiv.org/html/2601.00001v1"
SECRET_PHRASE = "An invented abstract."  # stands in for real paper text


def _page(licence_link: str, sections: int = 12) -> str:
    """The invented LaTeXML page, made long (many sections) and with a licence link."""
    extra = "".join(
        f'<section class="ltx_section" id="S{n}"><h2 class="ltx_title ltx_title_section">'
        f'<span class="ltx_tag">{n} </span>Section {n}</h2><div class="ltx_para">'
        f'<p class="ltx_p">{"Long invented paragraph text. " * 40}</p></div></section>'
        for n in range(2, sections + 2)
    )
    page = LATEXML.replace(
        '</section>\n<section class="ltx_bibliography"',
        f'</section>{extra}\n<section class="ltx_bibliography"',
    )
    return page.replace("<body>", f"<body><header>{licence_link}</header>", 1)


@pytest.mark.parametrize(
    ("licence", "expected"),
    [
        ("License: CC BY 4.0", True),
        ("CC BY-SA 4.0", True),
        ("CC0 1.0", True),
        ("License: CC BY-NC-SA 4.0", False),
        ("CC BY-ND 4.0", False),
        ("arXiv.org perpetual non-exclusive license", False),
        ("unknown", False),
    ],
)
def test_licence_rules(licence: str, expected: bool) -> None:
    assert is_permissive(licence) is expected


def test_permissive_page_keeps_a_short_real_excerpt() -> None:
    link = '<a href="https://creativecommons.org/licenses/by/4.0/">License: CC BY 4.0</a>'
    out = trim_arxiv_html(_page(link), URL)
    assert out.count("\n") + 1 <= MAX_LINES
    assert out.startswith(f"<!-- Trimmed excerpt of {URL}") and "Text kept" in out
    assert SECRET_PHRASE in out
    assert "Section 5" not in out  # only the first two sections are kept
    md = html_to_markdown(out)
    cleaned = clean_markdown(md)
    assert cleaned.references and cleaned.markdown.startswith("## 1 Introduction")


def test_other_licences_keep_only_the_structure() -> None:
    link = (
        '<a href="https://arxiv.org/licenses/nonexclusive-distrib/1.0/">'
        "License: arXiv.org perpetual non-exclusive license</a>"
    )
    out = trim_arxiv_html(_page(link), URL)
    assert "structure only" in out and SECRET_PHRASE not in out and "Invented text" in out
    assert "x^{2}" in out and "E=mc^{2}" not in out  # math replaced too
    md = html_to_markdown(out)
    headings = [line for line in md.split("\n") if line.startswith("#")]
    assert len(headings) >= 5  # title, abstract, sections, references, appendix survive
    assert "## References" in md


def test_recorder_trims_html_and_refuses_pdfs() -> None:
    class Request:
        uri = URL

    before_record_request(Request())
    response: dict[str, Any] = {
        "headers": {"Content-Type": ["text/html; charset=utf-8"], "Content-Length": ["1"]},
        "body": {"string": _page("License: unknown").encode()},
    }
    trimmed = before_record_response(response)["body"]["string"].decode()
    assert trimmed.startswith(f"<!-- Trimmed excerpt of {URL}") and SECRET_PHRASE not in trimmed
    assert "Content-Length" not in response["headers"]

    pdf: dict[str, Any] = {
        "headers": {"Content-Type": ["application/pdf"]},
        "body": {"string": b"%PDF"},
    }
    with pytest.raises(RuntimeError, match="refusing to record a PDF"):
        before_record_response(pdf)

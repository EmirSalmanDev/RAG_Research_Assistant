"""html_to_markdown on invented LaTeXML markup, and pdf_to_markdown on a tiny invented PDF
built inside the test (no network, no files)."""

# ruff: noqa: E501  (the invented HTML below keeps arXiv's long one-line markup)

import pytest

from app.ingest.clean import clean_markdown
from app.ingest.fulltext import html_to_markdown, pdf_to_markdown
from app.ingest.markdown import parse_heading

# Invented text in arXiv's LaTeXML markup (class names as arXiv's HTML uses them).
LATEXML = """<html><head><title>x</title><script>var x = 1;</script></head><body>
<nav class="ltx_page_navbar">navigation that must not appear</nav>
<article class="ltx_document">
<h1 class="ltx_title ltx_title_document">Invented Paper</h1>
<div class="ltx_authors">Ada Example</div>
<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>
<p class="ltx_p">An invented abstract.</p></div>
<section class="ltx_section" id="S1">
<h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">1 </span>Introduction</h2>
<div class="ltx_para"><p class="ltx_p">We use <math alttext="x^{2}" display="inline"><mi>x</mi></math>
as shown<span class="ltx_note ltx_role_footnote"><sup class="ltx_note_mark">1</sup><span class="ltx_note_outer">
<span class="ltx_note_content"><sup class="ltx_note_mark">1</sup>Code at example.org.</span></span></span>
in <cite class="ltx_cite">[<a class="ltx_ref" href="#bib.bib1">1</a>]</cite>.</p>
<table class="ltx_equation ltx_eqn_table"><tr><td><math alttext="E=mc^{2}" display="block"><mi>E</mi></math></td></tr></table>
</div>
<figure class="ltx_figure"><img src="x.png"/><figcaption class="ltx_caption">
<span class="ltx_tag ltx_tag_figure">Figure 1: </span>An invented figure.</figcaption></figure>
<figure class="ltx_table"><figcaption class="ltx_caption"><span class="ltx_tag">Table 1: </span>Invented scores.</figcaption>
<table class="ltx_tabular"><tr><th>Model</th><th>Score | %</th></tr><tr><td>A</td><td>1.0</td></tr></table></figure>
<section class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection"><span class="ltx_tag">1.1 </span>Details</h3>
<div class="ltx_para"><ul class="ltx_itemize"><li class="ltx_item">first</li><li class="ltx_item">second</li></ul></div>
</section></section>
<section class="ltx_bibliography" id="bib"><h2 class="ltx_title ltx_title_bibliography">Bibliography</h2>
<ul class="ltx_biblist"><li class="ltx_bibitem" id="bib.bib1"><span class="ltx_tag">[1]</span>
<span class="ltx_bibblock">A. Invented. A paper. 2025.</span></li></ul></section>
<section class="ltx_appendix" id="A1"><h2 class="ltx_title ltx_title_appendix">
<span class="ltx_tag">Appendix A </span>Proofs</h2><div class="ltx_para"><p class="ltx_p">Invented proof.</p></div></section>
</article><footer class="ltx_page_footer">footer that must not appear</footer></body></html>"""


def test_latexml_becomes_markdown() -> None:
    md = html_to_markdown(LATEXML)
    headings = [h for line in md.split("\n") if (h := parse_heading(line))]
    assert headings == [
        (1, "Invented Paper"),
        (2, "Abstract"),
        (2, "1 Introduction"),
        (3, "1.1 Details"),
        (2, "References"),  # the bibliography always becomes '## References'
        (2, "Appendix A Proofs"),
    ]
    assert "We use $x^{2}$ as shown (Code at example.org.) in [1]." in md  # inline math
    assert "$$\nE=mc^{2}\n$$" in md  # display math as its alttext
    assert "Figure 1: An invented figure." in md and "x.png" not in md
    assert "Table 1: Invented scores.\n\n| Model | Score \\| % |\n| --- | --- |\n| A | 1.0 |" in md
    assert "- first" in md and "- second" in md
    assert "[1] A. Invented. A paper. 2025." in md
    assert "navigation" not in md and "footer" not in md and "var x" not in md


def test_converted_html_cleans_into_body_references_and_appendix() -> None:
    cleaned = clean_markdown(html_to_markdown(LATEXML))
    assert cleaned.references == "[1] A. Invented. A paper. 2025."
    assert cleaned.markdown.startswith("## 1 Introduction")
    assert "An invented abstract." not in cleaned.markdown  # indexed as its own chunk
    assert cleaned.markdown.endswith("## Appendix A Proofs\n\nInvented proof.")


def test_non_latexml_html_is_rejected() -> None:
    with pytest.raises(ValueError, match="LaTeXML"):
        html_to_markdown("<html><body><p>No LaTeXML here.</p></body></html>")


def _invented_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    page = doc.new_page()
    page.insert_text((72, 72), "1 Introduction", fontsize=18)
    page.insert_text((72, 110), "This invented PDF holds one paragraph of body text.", fontsize=11)
    page.insert_text((72, 126), "It exists only inside this test.", fontsize=11)
    data: bytes = doc.tobytes()  # type: ignore[no-untyped-call]
    doc.close()  # type: ignore[no-untyped-call]
    return data


def test_pdf_becomes_markdown_in_memory() -> None:
    md = pdf_to_markdown(_invented_pdf())
    assert parse_heading(md.split("\n")[0]) == (1, "1 Introduction")
    assert "This invented PDF holds one paragraph of body text." in md
    assert "It exists only inside this test." in md


def test_a_broken_pdf_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017 - PyMuPDF raises its own error types
        pdf_to_markdown(b"%PDF-1.7 not really a pdf")

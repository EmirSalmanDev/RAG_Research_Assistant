"""PDFs for the upload tests, generated here from invented text (never real paper text)."""

import pymupdf

SENTENCE = (
    "This invented sentence number {n} describes widgets, gadgets and an imaginary benchmark "
    "that exists only inside this test fixture. "
)


def make_pdf(pages: int = 3, seed: str = "a") -> bytes:
    """A small text PDF. Different `seed`s give different bytes (and so different ids)."""
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    for page_no in range(1, pages + 1):
        page = doc.new_page()
        heading = f"{page_no} Invented section {seed}\n\n"
        body = "".join(SENTENCE.format(n=page_no * 100 + i) for i in range(12))
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 770), heading + body, fontsize=11)  # type: ignore[no-untyped-call]
    return bytes(doc.tobytes())  # type: ignore[no-untyped-call]


def make_blank_pdf() -> bytes:
    """A PDF with a page and no text at all (what a scan looks like to a text parser)."""
    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    doc.new_page()
    return bytes(doc.tobytes())  # type: ignore[no-untyped-call]

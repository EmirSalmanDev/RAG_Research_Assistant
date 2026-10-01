"""clean.py on the markdown fixtures (all invented and labelled) and on small inline cases."""

from pathlib import Path

import pytest

from app.ingest.clean import (
    clean_markdown,
    drop_front_matter,
    join_hyphenated_words,
    normalize_characters,
    normalize_heading_levels,
    remove_images,
    remove_page_artifacts,
    split_references,
)
from app.ingest.markdown import parse_heading

FIXTURES = Path(__file__).parents[1] / "fixtures" / "markdown"
ALL_FIXTURES = sorted(p.name for p in FIXTURES.glob("*.md"))


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def _headings(markdown: str) -> list[tuple[int, str]]:
    return [h for line in markdown.split("\n") if (h := parse_heading(line))]


# --- whole fixtures ---------------------------------------------------------------------


def test_fixtures_are_labelled_invented() -> None:
    """Rule 14: no fixture may hold real paper text; each says so in its header."""
    assert ALL_FIXTURES
    for name in ALL_FIXTURES:
        assert "INVENTED TEXT" in _fixture(name).split("-->")[0], name


def test_hf_style() -> None:
    result = clean_markdown(_fixture("hf_style.md"))
    assert _headings(result.markdown) == [
        (2, "1 Introduction"),
        (2, "2 Method"),
        (3, "2.1 Router"),
        (3, "2.2 Training"),
        (2, "3 Experiments"),
        (2, "4 Conclusion"),
        (2, "A Additional Results"),
        (3, "A.1 Ablation on the Number of Adapters"),
    ]
    md = result.markdown
    assert "Sparse Routing Adapters for Long-Context" not in md  # title (front matter)
    assert "Mira Okonkwo" not in md and "Invented University" not in md  # authors
    assert "keeps memory use flat" not in md  # abstract: indexed as its own chunk
    assert "![" not in md and "overview.png" not in md  # image link removed...
    assert "Figure 1: Overview of SRA." in md  # ...caption kept
    assert "Table 1: Recall@10" in md and "| SRA | 48.7 | 3.3 |" in md  # table intact
    assert "$$\nr_i = \\operatorname{softmax}(W h)_i\n$$" in md  # display math intact
    assert result.references is not None
    assert result.references.startswith("[1] A. Example and B. Sample.")
    assert "[1] A. Example" not in md


def test_html_style() -> None:
    result = clean_markdown(_fixture("html_style.md"))
    assert [h for h in _headings(result.markdown)] == [
        (2, "1 Introduction"),
        (2, "2 Related Work"),
        (2, "3 Method"),
        (3, "3.1 Calibration Score"),
        (3, "3.2 Choosing the Threshold"),
        (2, "4 Results"),
        (2, "Appendix A Proof of Lemma 1"),
        (3, "A.1 Hyperparameters"),
    ]
    assert "Small language models often accept" not in result.markdown  # h6 abstract dropped
    assert "Figure 2: Accepted answers" in result.markdown
    assert result.references is not None and result.references.count("\n- ") == 1


def test_pdf_style_artifacts_are_removed() -> None:
    result = clean_markdown(_fixture("pdf_style.md"))
    md = result.markdown
    assert _headings(md) == [(2, "1 Introduction"), (2, "2 Setup"), (2, "3 Results")]
    assert "arXiv:2601.09999" not in md  # vertical stamp
    assert "Preprint. Under review" not in md  # running header (3 times)
    assert "-----" not in md  # page separators
    assert not any(line.strip().isdigit() for line in md.split("\n"))  # page numbers
    assert "picture" not in md  # picture placeholder
    assert "retrieval systems rarely" in md  # 'sys-' + 'tems' re-joined
    assert "we find that" in md  # 'ﬁ' ligature -> 'fi'
    assert "Lena Brandt" not in md  # front matter before the bold **Abstract**
    assert "Figure 1: Vocabulary size against recall." in md
    assert result.references is not None and result.references.startswith("[1] D. Invented.")
    assert "Recall stays within one point" in md  # body before the references kept


def test_no_references_heading() -> None:
    result = clean_markdown(_fixture("no_references.md"))
    assert result.references is None
    assert _headings(result.markdown) == [(2, "1 Introduction"), (2, "2 Observation")]
    assert result.markdown.endswith("stops falling.")


def test_appendix_after_references_is_kept() -> None:
    result = clean_markdown(_fixture("appendix_after_references.md"))
    assert result.references == "[1] F. Nobody. An invented reference. 2023."
    assert _headings(result.markdown) == [
        (2, "1 Introduction"),
        (2, "2 Conclusion"),
        (2, "A Proofs"),
        (2, "B Extra Experiments"),
        (3, "B.1 Larger Models"),
    ]
    assert "Results with larger invented models." in result.markdown


def test_empty_sections_keep_their_headings() -> None:
    md = clean_markdown(_fixture("empty_sections.md")).markdown
    assert _headings(md) == [
        (2, "1 Introduction"),
        (2, "2 Method"),
        (3, "2.1 Encoder"),
        (3, "2.2 Decoder"),
        (2, "3 Limitations"),
        (2, "4 Conclusion"),
    ]


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_cleaning_is_idempotent(name: str) -> None:
    once = clean_markdown(_fixture(name)).markdown
    assert clean_markdown(once).markdown == once


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_output_has_no_html_comment_and_single_blank_lines(name: str) -> None:
    md = clean_markdown(_fixture(name)).markdown
    assert "<!--" not in md and "\n\n\n" not in md
    assert md == md.strip("\n")


# --- 1. characters ----------------------------------------------------------------------


def test_normalize_characters() -> None:
    raw = "e\u0301t\u00e9\r\nof\ufb01ce\u00a0hall\u00ad way\u200b <!-- note --> x\u00b2  \r\n"
    assert normalize_characters(raw) == "\u00e9t\u00e9\noffice hall way  x\u00b2\n"


# --- 2. images ---------------------------------------------------------------------------


def test_remove_images_keeps_captions_and_text() -> None:
    lines = [
        "![](fig1.png)",
        "**![a chart](fig2.png)**",
        "<img src='x.png'>",
        "See ![icon](i.png) the table.",
        "Figure 3: A caption.",
        "```",
        "![kept in code](c.png)",
        "```",
    ]
    assert remove_images(lines) == [
        "See  the table.",
        "Figure 3: A caption.",
        "```",
        "![kept in code](c.png)",
        "```",
    ]


# --- 3. page artifacts --------------------------------------------------------------------


def test_page_artifacts_are_removed_but_tables_and_code_are_not() -> None:
    lines = [
        "arXiv:2601.00001v1 [cs.LG] 3 Feb 2026",
        "Text.",
        "12",
        "- 13 -",
        "Page 14 of 20",
        "-----",
        "| 15 |",
        "```",
        "16",
        "```",
    ]
    assert remove_page_artifacts(lines) == ["Text.", "| 15 |", "```", "16", "```"]


def test_running_headers_only_removed_in_pdf_like_text() -> None:
    header = "Workshop on Invented Topics 2026"
    pdf = ["1", header, "A.", "-----", header, "B.", "2", header, "C."]
    assert remove_page_artifacts(pdf) == ["A.", "B.", "C."]
    # Without page artifacts (not PDF output), a repeated line is content: kept.
    plain = [header, "A.", header, "B.", header]
    assert remove_page_artifacts(plain) == plain


def test_short_repeated_lines_are_not_running_headers() -> None:
    lines = ["1", "Proof.", "x", "2", "Proof.", "y", "3", "Proof."]
    assert remove_page_artifacts(lines) == ["Proof.", "x", "Proof.", "y", "Proof."]


# --- 4. hyphenation -----------------------------------------------------------------------


def test_join_hyphenated_words() -> None:
    lines = ["the trans-", "former and multi-", "head", "Large-", "Scale", "| a- |", "| b |"]
    assert join_hyphenated_words(lines) == [
        "the transformer and multihead",
        "Large-",  # next line starts uppercase: a real hyphenated name, kept
        "Scale",
        "| a- |",  # tables untouched
        "| b |",
    ]


# --- 5. front matter -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lines",
    [
        ["# Title", "Authors", "## Abstract", "Abs text.", "## 1 Intro", "Body."],
        ["# Title", "**Abstract**", "Abs text.", "## 1 Intro", "Body."],
        ["Title", "Abstract: abs text.", "## 1 Intro", "Body."],
        ["# Title", "Abstract\u2014abs text.", "## 1 Intro", "Body."],
    ],
)
def test_front_matter_and_abstract_are_dropped(lines: list[str]) -> None:
    assert drop_front_matter(lines) == ["## 1 Intro", "Body."]


def test_without_abstract_body_starts_at_introduction() -> None:
    lines = ["# Title", "Authors", "## Introduction", "Body."]
    assert drop_front_matter(lines) == ["## Introduction", "Body."]


def test_unknown_layout_only_drops_the_title() -> None:
    lines = ["# Title", "Some text.", "## Background", "More."]
    assert drop_front_matter(lines) == ["Some text.", "## Background", "More."]
    assert drop_front_matter(["Text only."]) == ["Text only."]


# --- 6. references ----------------------------------------------------------------------------


def test_references_as_lone_line_until_next_heading() -> None:
    lines = ["## 1 Intro", "Body.", "**REFERENCES**", "[1] Ref.", "## A Appendix", "App."]
    body, refs, after = split_references(lines)
    assert (body, refs, after) == (["## 1 Intro", "Body."], ["[1] Ref."], ["## A Appendix", "App."])


def test_references_inside_code_are_ignored() -> None:
    lines = ["## 1 Intro", "```", "References", "```"]
    assert split_references(lines) == (lines, None, [])


# --- 7. heading levels -------------------------------------------------------------------------


def test_heading_levels_follow_section_numbers() -> None:
    body = [
        "# 1 Intro",
        "#### 1.1 Background",
        "## A Simple Baseline",  # body: 'A' is a word, not an appendix letter
        "### I. Roman Section",
        "## 2023 Results",  # a year is not a section number
        "```",
        "# a comment in code",
        "```",
    ]
    appendix = ["# A Proofs", "# A.1 Lemma"]
    # Unnumbered headings keep their place relative to the shallowest heading ('# 1 Intro',
    # level 1 -> 2), so the two level-2 unnumbered ones become ###. Read as numbered,
    # 'A Simple Baseline' and '2023 Results' would have become ## instead.
    assert normalize_heading_levels(body, appendix) == [
        "## 1 Intro",
        "### 1.1 Background",
        "### A Simple Baseline",
        "## I. Roman Section",
        "### 2023 Results",
        "```",
        "# a comment in code",
        "```",
        "## A Proofs",
        "### A.1 Lemma",
    ]


def test_unnumbered_levels_are_shifted_relative_to_the_shallowest() -> None:
    body = ["### Intro", "#### Details", "### Method"]
    assert normalize_heading_levels(body, []) == ["## Intro", "### Details", "## Method"]

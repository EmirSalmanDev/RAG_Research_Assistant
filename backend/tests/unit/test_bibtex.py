"""BibTeX entries from metadata: keys, names, escaping, uploads, determinism. All names and
titles are invented."""

from datetime import UTC, datetime

from app.features.bibtex import (
    BibPaper,
    assign_keys,
    base_key,
    build_entries,
    escape_latex,
    export_filename,
    fold_ascii,
    format_authors,
    order_papers,
    render_bibtex,
    split_name,
    title_word,
)
from tests.bib_validate import balanced, keys, validate


def paper(
    arxiv_id: str = "2601.00001",
    title: str = "An Invented Study of Widgets",
    authors: tuple[str, ...] = ("Ada Example", "Bo Sample"),
    year: int = 2026,
    category: str | None = "cs.CL",
) -> BibPaper:
    return BibPaper(arxiv_id, title, authors, datetime(year, 3, 4, tzinfo=UTC), category)


def test_an_arxiv_entry_has_the_eprint_fields_and_the_abstract_link() -> None:
    [entry] = build_entries([paper()])
    assert entry.key == "example2026invented"
    assert render_bibtex([entry]) == (
        "@misc{example2026invented,\n"
        "  author = {Example, Ada and Sample, Bo},\n"
        "  title = {{An Invented Study of Widgets}},\n"
        "  year = {2026},\n"
        "  eprint = {2601.00001},\n"
        "  archivePrefix = {arXiv},\n"
        "  primaryClass = {cs.CL},\n"
        "  url = {https://arxiv.org/abs/2601.00001},\n"
        "}\n"
    )


def test_an_old_style_id_with_a_slash_stays_in_the_fields_not_in_the_key() -> None:
    [entry] = build_entries([paper("hep-th/9901001", authors=("Eve Old",), year=1999)])
    assert entry.key == "old1999invented"
    text = render_bibtex([entry])
    assert "eprint = {hep-th/9901001}" in text
    assert "url = {https://arxiv.org/abs/hep-th/9901001}" in text
    assert validate(text) == []


def test_an_uploaded_document_has_a_note_and_no_eprint_or_url() -> None:
    [entry] = build_entries([paper("local-0123456789ab", category=None)])
    text = render_bibtex([entry])
    assert "note = {Uploaded document}" in text
    assert "eprint" not in text and "url" not in text and "archivePrefix" not in text
    assert validate(text) == []


def test_unicode_names_are_kept_in_the_entry_and_folded_in_the_key() -> None:
    p = paper(authors=("Jürgen Müller", "Zoë Ødegård", "Łukasz Kaiser"), title="Über Alles")
    [entry] = build_entries([p])
    assert entry.key == "muller2026uber"
    assert "author = {Müller, Jürgen and Ødegård, Zoë and Kaiser, Łukasz}" in entry.render()
    assert fold_ascii("Ødegård-Łukasz ß") == "odegardlukaszss"
    assert fold_ascii("日本語") == ""


def test_a_name_that_folds_to_nothing_gets_a_fallback_key() -> None:
    assert base_key(paper(authors=("王 小明",))).startswith("anon2026")


def test_name_forms() -> None:
    assert split_name("Ludwig van Beethoven") == ("van Beethoven", "", "Ludwig")
    assert split_name("Beethoven, Ludwig van") == ("Beethoven", "", "Ludwig van")
    assert split_name("Martin Luther King Jr.") == ("King", "Jr.", "Martin Luther")
    assert split_name("Voltaire") == ("Voltaire", "", "")
    assert split_name("J. R. R. Tolkien") == ("Tolkien", "", "J. R. R.")
    assert format_authors(["Martin Luther King Jr."]) == "King, Jr., Martin Luther"
    assert format_authors(["Voltaire", "Ada Example"]) == "Voltaire and Example, Ada"
    assert format_authors(["Ludwig van Beethoven"]) == "van Beethoven, Ludwig"


def test_a_name_with_the_word_and_is_braced_so_it_is_one_author() -> None:
    assert format_authors(["Research and Co"]) == "{Co, Research and}"
    assert (
        format_authors(["Ada Example", "Research and Co"]) == "Example, Ada and {Co, Research and}"
    )


def test_missing_authors_are_anonymous() -> None:
    [entry] = build_entries([paper(authors=())])
    assert "author = {Anonymous}" in entry.render()
    assert entry.key.startswith("anon2026")
    assert validate(entry.render()) == []


def test_special_characters_are_escaped_and_the_braces_balance() -> None:
    title = "Cost & Speed: 100% of $x_1$ in {C#} ~ a^b \\ done\n(wrapped)"
    assert escape_latex(title) == (
        r"Cost \& Speed: 100\% of \$x\_1\$ in \{C\#\} \textasciitilde{} a\textasciicircum{}b "
        r"\textbackslash{} done (wrapped)"
    )
    text = render_bibtex(build_entries([paper(title=title)]))
    assert balanced(text) and validate(text) == []
    assert "\n  title = {{Cost \\& Speed" in text


def test_the_title_word_skips_stopwords() -> None:
    assert title_word("Attention Is All You Need") == "attention"
    assert title_word("A Survey of the Methods") == "survey"
    assert title_word("The Of And") == "the"  # only stopwords: the first word
    assert title_word("") == "untitled"
    assert title_word("Pre-training of BERT") == "pre"
    assert title_word("$$$") == "untitled"


def test_collisions_get_letters_in_order_and_the_first_keeps_the_plain_key() -> None:
    papers = [paper(f"2601.0000{i}") for i in range(1, 5)]
    assert assign_keys(papers) == [
        "example2026invented",
        "example2026inventeda",
        "example2026inventedb",
        "example2026inventedc",
    ]


def test_many_collisions_run_past_z() -> None:
    keys_ = assign_keys([paper(f"2601.{i:05d}") for i in range(30)])
    assert len(set(keys_)) == 30
    assert keys_[26] == "example2026inventedz" and keys_[27] == "example2026inventedaa"


def test_a_letter_suffix_never_collides_with_a_natural_key() -> None:
    other = paper("2601.00009", title="Inventeda Thing")  # natural key ...inventeda
    keys_ = assign_keys([paper("2601.00001"), paper("2601.00002"), other])
    assert len(set(keys_)) == 3


def test_the_same_papers_in_the_same_order_give_the_same_keys() -> None:
    papers = [paper(f"2601.0000{i}", year=2020 + i % 2) for i in range(1, 7)]
    first = render_bibtex(build_entries(order_papers(papers)))
    second = render_bibtex(build_entries(order_papers(list(reversed(papers)))))
    assert first == second and keys(first) == keys(second)


def test_order_is_oldest_first_then_id() -> None:
    a, b, c = paper("2601.00003", year=2025), paper("2601.00001"), paper("2601.00002")
    assert [p.arxiv_id for p in order_papers([b, c, a])] == [
        "2601.00003",
        "2601.00001",
        "2601.00002",
    ]


def test_a_whole_file_validates_and_has_unique_keys() -> None:
    papers = [
        paper("2601.00001"),
        paper("2601.00002"),
        paper("local-aaaaaaaaaaaa", category=None, authors=()),
        paper("hep-th/9901001", year=1999, title="Cost & Speed"),
    ]
    text = render_bibtex(build_entries(order_papers(papers)))
    assert validate(text) == []
    assert len(keys(text)) == len(set(keys(text))) == 4
    assert text.endswith("}\n") and "\n\n@misc{" in text


def test_the_validator_catches_the_mistakes_it_is_for() -> None:
    good = render_bibtex(build_entries([paper()]))
    assert validate(good + "\n" + good) == ["duplicate key example2026invented"]
    assert validate(good.replace("year = {2026},\n", "")) == ["example2026invented: missing year"]
    assert "unbalanced braces" in validate(good.replace("{{An", "{An"))


def test_export_filename() -> None:
    assert export_filename("My Thesis (draft)") == "my-thesis-draft.bib"
    assert export_filename("Tez çalışması") == "tez-calismasi.bib"
    assert export_filename("日本語") == "library.bib"

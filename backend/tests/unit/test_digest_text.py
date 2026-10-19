"""The card's fallback summary: the abstract's first sentence, cut to a length."""

import pytest

from app.features.digest import card_summary, first_sentence


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("We study X. Then we do Y.", "We study X."),
        ("One sentence only", "One sentence only"),
        ("Is it? Yes. Maybe.", "Is it?"),
        ("  Spaces\n and\tnewlines. Next one.", "Spaces and newlines."),
        ("Accuracy reaches 0.91 on the set. Next.", "Accuracy reaches 0.91 on the set."),
        ("See Fig. 3 for details. Next.", "See Fig. 3 for details."),
        ("Uses e.g. a small model. Next.", "Uses e.g. a small model."),
        ("", ""),
    ],
)
def test_first_sentence(text: str, expected: str) -> None:
    assert first_sentence(text, 200) == expected


def test_a_long_sentence_is_cut_at_a_word_with_an_ellipsis() -> None:
    sentence = "word " * 100
    cut = first_sentence(sentence, 50)
    assert len(cut) <= 50 and cut.endswith("…")
    assert not cut[:-1].endswith(" ") and cut[:-1].endswith("word")  # no half words


def test_a_long_sentence_without_spaces_is_still_cut() -> None:
    cut = first_sentence("x" * 500, 50)
    assert len(cut) == 50 and cut.endswith("…")


def test_card_summary_prefers_the_stored_tldr() -> None:
    assert card_summary("  The   TL;DR. ", "Abstract one. Two.") == ("The TL;DR.", "analysis")
    assert card_summary(None, "Abstract one. Two.") == ("Abstract one.", "abstract")
    assert card_summary("   ", "Abstract one. Two.") == ("Abstract one.", "abstract")

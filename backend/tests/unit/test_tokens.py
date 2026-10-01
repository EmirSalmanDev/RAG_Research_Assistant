"""tokens.py with the real tokenizer (tiktoken cl100k_base, read from its local cache)."""

import pytest

from app.ingest.chunk import ChunkSettings
from app.ingest.tokens import count_tokens, get_encoding, split_by_tokens, tail_within


def test_encoding_is_the_embedding_models() -> None:
    assert get_encoding().name == "cl100k_base"  # text-embedding-3-small's tokenizer
    assert count_tokens("hello world") == 2
    assert count_tokens("<|endoftext|>") > 1  # special-token text counted as plain text


def test_split_by_tokens_cuts_between_words() -> None:
    text = " ".join(f"word{i}" for i in range(300))
    pieces = split_by_tokens(text, 50)
    assert all(count_tokens(p) <= 50 for p in pieces)
    assert " ".join(pieces) == text  # nothing lost, no word cut


def test_split_by_tokens_cuts_a_giant_word_by_tokens() -> None:
    word = "x" * 2000  # one 'word' far over the limit
    pieces = split_by_tokens(f"before {word} after", 20)
    assert pieces[0] == "before" and pieces[-1] == "after"
    assert "".join(pieces[1:-1]) == word and all(count_tokens(p) <= 20 for p in pieces)


def test_tail_within_takes_whole_words_up_to_the_limit() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta"
    tail = tail_within(text, 5)
    assert text.endswith(tail) and count_tokens(tail) <= 5
    assert count_tokens(" ".join(text.split()[-(len(tail.split()) + 1) :])) > 5  # longest
    assert tail_within(text, 0) == ""


def test_tail_within_trims_a_long_last_word() -> None:
    tail = tail_within("short " + "y" * 500, 10)
    assert tail and set(tail) == {"y"} and count_tokens(tail) <= 10


@pytest.mark.parametrize(
    ("tokens", "overlap", "hard_max"), [(512, 512, 2048), (512, -1, 2048), (4096, 64, 2048)]
)
def test_chunk_settings_reject_impossible_values(tokens: int, overlap: int, hard_max: int) -> None:
    with pytest.raises(ValueError):
        ChunkSettings(tokens=tokens, overlap=overlap, min_tokens=64, hard_max=hard_max)

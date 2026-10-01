"""Token counting with the embedding model's own tokenizer (tiktoken).

tiktoken downloads the tokenizer file once and caches it in the directory named by the
TIKTOKEN_CACHE_DIR environment variable. Its default is the OS temp directory, which
macOS clears, so when the variable is unset we point it at ~/.cache/tiktoken. CI and the
Docker image set the variable themselves; tests run offline from the cached file.
"""

import os
from functools import lru_cache
from pathlib import Path

import tiktoken

from app.config import get_settings

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tiktoken"


@lru_cache
def get_encoding() -> tiktoken.Encoding:
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(DEFAULT_CACHE_DIR))
    # text-embedding-3-small uses the cl100k_base encoding.
    return tiktoken.encoding_for_model(get_settings().embedding_model)


def count_tokens(text: str) -> int:
    # encode_ordinary: special-token strings such as '<|endoftext|>' in paper text are
    # counted as plain text instead of raising an error.
    return len(get_encoding().encode_ordinary(text))


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Cut text into pieces of at most `max_tokens` tokens, between words. Only a single
    word longer than that (a long URL, a formula) is cut inside, by tokens.
    Used as the last resort for a sentence or a line too long to keep whole."""
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        if count_tokens(word) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []
            ids = get_encoding().encode_ordinary(word)
            pieces += [
                get_encoding().decode(ids[i : i + max_tokens])
                for i in range(0, len(ids), max_tokens)
            ]
            continue
        if current and count_tokens(" ".join([*current, word])) > max_tokens:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def tail_within(text: str, max_tokens: int) -> str:
    """The longest run of whole words at the end of `text` that has at most `max_tokens`
    tokens (measured on the returned string itself). If even the last word is longer,
    the end of that word, trimmed until it fits."""
    words = text.split()
    tail: list[str] = []
    for word in reversed(words):
        if count_tokens(" ".join([word, *tail])) > max_tokens:
            break
        tail.insert(0, word)
    if tail or not words or max_tokens <= 0:
        return " ".join(tail)
    ids = get_encoding().encode_ordinary(words[-1])
    end = get_encoding().decode(ids[-max_tokens:])
    while end and count_tokens(end) > max_tokens:
        end = end[1:]
    return end

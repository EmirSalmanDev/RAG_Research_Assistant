"""Small text helpers shared by compare and timeline."""

from app.ingest.tokens import get_encoding

CUT_MARK = "\n[text cut]"


def cut_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """`text` shortened to at most `max_tokens` tokens (cl100k_base), marked with
    CUT_MARK when something was removed. Returns (text, was_cut)."""
    ids = get_encoding().encode_ordinary(text)
    if len(ids) <= max_tokens:
        return text, False
    return get_encoding().decode(ids[:max_tokens]).rstrip() + CUT_MARK, True

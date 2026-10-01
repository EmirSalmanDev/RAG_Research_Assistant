"""chunk.py with the real tokenizer (tiktoken cl100k_base, read from its local cache).

Text is invented: the markdown fixtures (labelled as invented) and generated sentences.
"""

import re
import uuid
from pathlib import Path

import pytest

from app.config import get_settings
from app.ingest import chunk as chunk_module
from app.ingest.chunk import (
    ABSTRACT_ORD,
    FIRST_SECTION_ORD,
    NAMESPACE,
    Chunk,
    ChunkSettings,
    abstract_chunk,
    chunk_id,
    chunk_paper,
)
from app.ingest.clean import clean_markdown
from app.ingest.tokens import count_tokens, get_encoding

FIXTURES = Path(__file__).parents[1] / "fixtures" / "markdown"
TITLE = "Invented Title"
ID, VERSION = "2601.00001", 2
CFG = ChunkSettings(tokens=512, overlap=64, min_tokens=64, hard_max=2048)


@pytest.fixture(autouse=True)
def tokenizer_is_cached() -> None:
    """The tests run with the network blocked, so the tokenizer must come from the cache.
    (Function scope on purpose: the network block covers it; see tests/conftest.py.)"""
    try:
        get_encoding()
    except Exception as err:  # pragma: no cover - only when the cache is missing
        pytest.fail(
            "tiktoken cl100k_base is not cached; download it once with the network on "
            f'(uv run python -c "from app.ingest.tokens import get_encoding; get_encoding()"): '
            f"{err}"
        )


def _chunks(name: str, cfg: ChunkSettings = CFG) -> list[Chunk]:
    markdown = clean_markdown((FIXTURES / name).read_text()).markdown
    return chunk_paper(ID, VERSION, TITLE, markdown, cfg)


def _sentences(n: int, start: int = 0) -> list[str]:
    topics = ["routing", "memory", "latency", "recall", "tokenizers", "adapters", "batches"]
    return [
        f"Invented sentence {i} reports an imaginary finding about {topics[i % 7]} "
        f"measured on the made-up benchmark number {i * 7 % 13}."
        for i in range(start, start + n)
    ]


def _overlap_tokens(prev: str, nxt: str) -> int:
    """Tokens in the longest end of `prev` that `nxt` starts with (0 if none)."""
    for k in range(min(len(prev), len(nxt)), 0, -1):
        if nxt.startswith(prev[-k:]) and (k == len(prev) or not prev[-k - 1].isalnum()):
            return count_tokens(prev[-k:])
    return 0


# --- ids ------------------------------------------------------------------------------------------


def test_chunk_id_is_uuid5_over_the_project_namespace() -> None:
    assert chunk_id(ID, VERSION, 3) == uuid.uuid5(NAMESPACE, f"{ID}v{VERSION}#3")
    assert chunk_id(ID, VERSION, 3) == chunk_id(ID, VERSION, 3)  # deterministic
    # Pinned value: if this changes, every stored point id changes too.
    assert str(chunk_id("1706.03762", 7, 0)) == str(
        uuid.uuid5(uuid.UUID("da2a306b-299a-49f3-9ba9-5a1724f60504"), "1706.03762v7#0")
    )
    distinct = {chunk_id(ID, v, o) for v in (1, 2) for o in (0, 1, 2)}
    assert len(distinct) == 6  # version and ord both change the id


def test_ids_are_the_same_on_every_run() -> None:
    first = [(c.id, c.text) for c in _chunks("hf_style.md")]
    second = [(c.id, c.text) for c in _chunks("hf_style.md")]
    assert first == second


def test_abstract_chunk_is_ord_zero() -> None:
    c = abstract_chunk(ID, VERSION, TITLE, "  An invented abstract.  ")
    assert (c.ord, c.chunk_type, c.id) == (ABSTRACT_ORD, "abstract", chunk_id(ID, VERSION, 0))
    assert c.text == "An invented abstract."
    assert c.embedding_text == f"{TITLE} > Abstract\n\nAn invented abstract."
    assert c.token_count == count_tokens("An invented abstract.")


def test_section_chunks_are_numbered_from_one() -> None:
    chunks = _chunks("hf_style.md")
    assert [c.ord for c in chunks] == list(range(FIRST_SECTION_ORD, FIRST_SECTION_ORD + 5))
    assert all(c.id == chunk_id(ID, VERSION, c.ord) for c in chunks)
    assert all(c.chunk_type == "section" for c in chunks)


def test_too_many_chunks_would_collide_with_analysis_ords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chunk_module, "FIRST_ANALYSIS_ORD", 3)
    with pytest.raises(ValueError, match="ids would collide"):
        _chunks("hf_style.md")


# --- sections, paths and the embedding prefix -----------------------------------------------------


def test_section_paths_and_embedding_prefix() -> None:
    chunks = _chunks("hf_style.md")
    assert [c.section_path for c in chunks] == [
        "1 Introduction",
        "2 Method > 2.1 Router",  # empty '2 Method' merged into its first subsection
        "3 Experiments",  # short '2.2 Training' merged into the next section
        "A Additional Results",  # short '4 Conclusion' merged into the next section
        "A Additional Results > A.1 Ablation on the Number of Adapters",
    ]
    for c in chunks:
        assert c.embedding_text == f"{TITLE} > {c.section_path}\n\n{c.text}"
    # A merged section keeps its heading line in the text.
    assert chunks[2].text.startswith("### 2.2 Training\n\nWe train the adapters")


def test_empty_sections_merge_one_hop_and_lose_no_heading() -> None:
    chunks = _chunks("empty_sections.md")
    assert [c.section_path for c in chunks] == [
        "2 Method",
        "2 Method > 2.2 Decoder",
        "4 Conclusion",
    ]
    text = "\n".join(c.text for c in chunks)
    for heading in ("1 Introduction", "2.1 Encoder", "3 Limitations"):
        assert heading in text


@pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES.glob("*.md")))
def test_every_fixture_chunks_within_limits_and_loses_no_words(name: str) -> None:
    markdown = clean_markdown((FIXTURES / name).read_text()).markdown
    chunks = chunk_paper(ID, VERSION, TITLE, markdown, CFG)
    assert chunks
    for c in chunks:
        assert c.text.strip() and c.token_count == count_tokens(c.text)
        assert c.token_count <= CFG.hard_max
        assert count_tokens(c.embedding_text) < 8191  # the embedding model's input limit
    words = set(re.findall(r"\w+", markdown))
    covered = set(re.findall(r"\w+", " ".join(f"{c.section_path} {c.text}" for c in chunks)))
    assert words <= covered


# --- windows: size and overlap in tokens ----------------------------------------------------------


def test_long_section_windows_and_overlap_in_tokens() -> None:
    # One long section: 12 paragraphs of 10 invented sentences each.
    paragraphs = [" ".join(_sentences(10, start=10 * p)) for p in range(12)]
    markdown = "## 1 Long Section\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_paper(ID, VERSION, TITLE, markdown, CFG)
    sizes = [c.token_count for c in chunks]
    total = count_tokens("\n\n".join(paragraphs))
    print(f"\nlong section: {total} tokens -> {len(chunks)} chunks of {sizes} tokens")
    assert len(chunks) >= total // CFG.tokens + 1 and max(sizes) <= CFG.tokens
    assert all(size >= CFG.tokens - 200 for size in sizes[:-1])  # 'about 512'
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        overlap = _overlap_tokens(prev.text, nxt.text)
        assert 0 < overlap <= CFG.overlap, (overlap, prev.ord)
        assert nxt.text.split(".")[0] + "." in prev.text  # starts with a repeated sentence
        assert prev.text.endswith(".")  # cut on a sentence boundary
    assert all(c.section_path == "1 Long Section" for c in chunks)


def test_one_paragraph_of_sentences_is_cut_on_sentence_boundaries() -> None:
    markdown = "## 1 One Paragraph\n\n" + " ".join(_sentences(80))
    chunks = chunk_paper(ID, VERSION, TITLE, markdown, CFG)
    assert len(chunks) > 1 and max(c.token_count for c in chunks) <= CFG.tokens
    sentences = set(_sentences(80))
    for c in chunks:
        assert set(re.split(r"(?<=\.) ", c.text)) <= sentences  # only whole sentences


def test_a_sentence_without_end_is_cut_by_tokens() -> None:
    markdown = "## 1 Run-on\n\n" + " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_paper(ID, VERSION, TITLE, markdown, CFG)
    assert len(chunks) > 1 and max(c.token_count for c in chunks) <= CFG.tokens
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert 0 < _overlap_tokens(prev.text, nxt.text) <= CFG.overlap


def test_settings_come_from_config() -> None:
    s = get_settings()
    assert ChunkSettings.from_config() == ChunkSettings(
        s.chunk_tokens, s.chunk_overlap, s.chunk_min_tokens, s.chunk_hard_max_tokens
    )
    assert ChunkSettings.from_config() == CFG  # the defaults these tests assume


def test_smaller_settings_are_respected() -> None:
    small = ChunkSettings(tokens=100, overlap=10, min_tokens=10, hard_max=300)
    markdown = "## 1 Section\n\n" + " ".join(_sentences(40))
    chunks = chunk_paper(ID, VERSION, TITLE, markdown, small)
    assert max(c.token_count for c in chunks) <= 100
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert 0 < _overlap_tokens(prev.text, nxt.text) <= 10


# --- tables, code and math ------------------------------------------------------------------------


def _table(rows: int) -> list[str]:
    return ["Table 1: Invented scores.", "", "| Model | Score |", "| --- | --- |"] + [
        f"| invented-model-{i} | {i}.5 |" for i in range(rows)
    ]


def test_table_is_never_split_below_the_hard_max() -> None:
    before = " ".join(_sentences(20))  # ~600 tokens of prose before the table
    lines = ["## 1 Results", "", before, "", *_table(40), "", " ".join(_sentences(5, 100))]
    chunks = chunk_paper(ID, VERSION, TITLE, "\n".join(lines), CFG)
    table = "\n".join(_table(40)).replace("\n\n", "\n")  # the caption is attached to it
    holders = [c for c in chunks if "| invented-model-0 |" in c.text]
    assert len(holders) == 1 and table in holders[0].text  # whole table, caption included
    assert sum("| invented-model-39 |" in c.text for c in chunks) == 1  # not repeated


def test_giant_table_is_split_by_rows_with_the_header_repeated() -> None:
    chunks = [c for c in _chunks("giant_table.md") if "| config-" in c.text]
    head = [
        "Table 3: Invented accuracy, latency and memory for every configuration.",
        "| Configuration | Dataset | Accuracy | Latency (ms) | Memory (GB) | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    largest = max(c.token_count for c in chunks)
    sizes = [c.token_count for c in chunks]
    print(f"\ngiant table: {len(chunks)} pieces of {sizes} tokens, largest {largest}")
    assert len(chunks) >= 3 and largest <= CFG.hard_max
    rows: list[str] = []
    for c in chunks:
        lines = c.text.split("\n")
        assert lines[:3] == head  # caption, header row and separator in every piece
        assert all(line.startswith("| config-") for line in lines[3:])  # whole rows only
        rows += lines[3:]
    names = [row.split(" |")[0] for row in rows]
    assert names == [f"| config-{i:03d}" for i in range(1, 251)]  # every row once, in order


@pytest.mark.parametrize(("open_", "close"), [("```python", "```"), ("$$", "$$")])
def test_giant_code_and_math_are_split_by_lines_inside_their_fences(open_: str, close: str) -> None:
    body = [f"# invented line {i}: value_{i} = compute(value_{i - 1}) + {i}" for i in range(600)]
    markdown = "\n".join(["## 1 Listing", "", open_, *body, close])
    chunks = chunk_paper(ID, VERSION, TITLE, markdown, CFG)
    assert len(chunks) >= 2 and max(c.token_count for c in chunks) <= CFG.hard_max
    lines: list[str] = []
    for c in chunks:
        piece = c.text.split("\n")
        assert piece[0] == open_ and piece[-1] == close
        lines += piece[1:-1]
    assert lines == body  # every line once, in order; '# ...' lines were not headings
    assert all(c.section_path == "1 Listing" for c in chunks)

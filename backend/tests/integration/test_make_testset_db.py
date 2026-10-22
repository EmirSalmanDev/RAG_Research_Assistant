"""The test-set tool's reads from Postgres, on a scratch database with invented papers:
the pool of full-text papers, the golden papers left out of it, the ingest suggestions,
and the commands that refuse to generate from too few papers or only estimate (no key, no
network, no model call)."""

import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.ingest.chunk import Chunk, abstract_chunk, chunk_id, embedding_text
from app.ingest.index import save_chunks
from eval import make_testset
from eval.make_testset import (
    MakeTestsetError,
    golden_papers,
    ingest_suggestions,
    load_pool,
    main,
    papers_of,
)
from tests.digest_fakes import make_paper

pytestmark = pytest.mark.integration


def _fulltext_paper(s: Session, arxiv_id: str, sections: int) -> None:
    paper = make_paper(s, arxiv_id, title=f"Invented Study {arxiv_id}")
    chunks = [abstract_chunk(arxiv_id, 1, paper.title, paper.abstract)]
    for k in range(1, sections + 1):
        text = f"Invented section {k} of {arxiv_id}. " + "More invented words. " * 60
        chunks.append(
            Chunk(id=chunk_id(arxiv_id, 1, k), ord=k, chunk_type="section",
                  section_path=f"{k} Part", text=text, token_count=300,
                  embedding_text=embedding_text(paper.title, f"{k} Part", text))
        )  # fmt: skip
    save_chunks(s, arxiv_id, chunks)
    s.commit()


@pytest.fixture
def golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An invented golden file: one question refers to a section of 2601.00006 (a
    full-text paper), one to the abstract of 2601.00004 (an abstract-only paper)."""
    path = tmp_path / "golden.jsonl"
    items = [
        {"id": "q01", "question": "Invented golden question?", "type": "fact",
         "reference_chunk_ids": [str(chunk_id("2601.00006", 1, 1))]},
        {"id": "q02", "question": "Another invented golden question?", "type": "filter",
         "reference_chunk_ids": [str(chunk_id("2601.00004", 1, 0))]},
    ]  # fmt: skip
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    monkeypatch.setattr(make_testset, "GOLDEN_PATH", path)
    return path


@pytest.fixture
def corpus(migrated_session: Session, golden: Path) -> Session:
    _fulltext_paper(migrated_session, "2601.00001", 3)
    _fulltext_paper(migrated_session, "2601.00002", 2)
    _fulltext_paper(migrated_session, "2601.00006", 2)  # a golden paper
    for i, upvotes in enumerate((None, 50, 7), start=3):  # abstract-only papers
        paper = make_paper(migrated_session, f"2601.0000{i}", upvotes=upvotes, age_days=i)
        save_chunks(
            migrated_session, paper.arxiv_id,
            [abstract_chunk(paper.arxiv_id, 1, paper.title, paper.abstract)],
        )  # fmt: skip
        migrated_session.commit()
    return migrated_session


def _stale_golden(tmp_path: Path) -> Path:
    stale = tmp_path / "stale.jsonl"
    gone = {"id": "q09", "question": "Gone?", "type": "fact",
            "reference_chunk_ids": [str(chunk_id("2601.00099", 1, 1))]}  # fmt: skip
    stale.write_text(json.dumps(gone) + "\n")
    return stale


def test_the_pool_is_the_full_text_papers_sections_and_abstracts(corpus: Session) -> None:
    pool = load_pool(corpus)
    assert papers_of(pool) == ["2601.00001", "2601.00002", "2601.00006"]
    assert sorted((c.arxiv_id, c.chunk_type) for c in pool).count(("2601.00001", "section")) == 3
    assert {c.chunk_type for c in pool} == {"abstract", "section"}
    assert all(c.title.startswith("Invented Study") for c in pool)


def test_golden_papers_are_found_through_their_reference_chunks(
    corpus: Session, golden: Path, tmp_path: Path
) -> None:
    assert golden_papers(corpus, golden) == {"2601.00006", "2601.00004"}
    with pytest.raises(MakeTestsetError, match="not in the database"):
        golden_papers(corpus, _stale_golden(tmp_path))


def test_ingest_suggestions_put_the_most_upvoted_abstract_only_papers_first(
    corpus: Session,
) -> None:
    ids = [arxiv_id for arxiv_id, _, _ in ingest_suggestions(corpus, 3, set())]
    assert ids == ["2601.00004", "2601.00005", "2601.00003"]  # 50, 7, then no upvotes
    # A paper a golden question refers to is never suggested.
    ids = [arxiv_id for arxiv_id, _, _ in ingest_suggestions(corpus, 3, {"2601.00004"})]
    assert ids == ["2601.00005", "2601.00003"]


def test_generate_refuses_with_too_few_papers_and_lists_suggestions(
    corpus: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--min-papers", "4"]) == 3
    out = capsys.readouterr()
    # 2601.00006 has full text, but a golden question refers to it: it does not count.
    assert "that can seed questions: 2 (2601.00001, 2601.00002)" in out.out
    assert "golden questions refer to them: 2601.00004, 2601.00006" in out.out
    suggested = out.out.split("Suggested")[1]
    assert "missing 2" in out.out and "Nothing was generated" in out.err
    assert "2601.00005" in suggested and "2601.00004" not in suggested


def test_the_estimate_reads_the_local_stores_and_counts_only_new_seeds(
    corpus: Session, capsys: pytest.CaptureFixture[str], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # fmt: skip
    asked: list[str] = []

    def vectors(chunks: list[make_testset.SeedChunk]) -> dict[object, list[float]]:
        asked.extend(c.arxiv_id for c in chunks)  # the local index, faked: no Qdrant points
        return {c.chunk_id: [1.0, float(c.chunk_id.int % 7)] for c in chunks}

    monkeypatch.setattr(make_testset, "pool_vectors", vectors)
    candidates = tmp_path / "candidates.jsonl"
    args = ["generate", "--estimate", "--min-papers", "2", "--fact", "2", "--compare", "1",
            "--filter", "3", "--unanswerable", "1", "--candidates", str(candidates)]  # fmt: skip
    assert main(args) == 0
    out = capsys.readouterr().out
    # Two seedable papers: only their two abstracts can seed filter questions (3 asked).
    assert "6 seeds from 2 papers" in out and "fewer than asked: {'filter': '2/3'}" in out
    assert "0 already in candidates.jsonl" in out and "6 requests" in out
    assert "Nothing was sent" in out and not candidates.exists()
    assert set(asked) == {"2601.00001", "2601.00002"}  # never the golden paper's vectors
    # Pool rows carry the paper's primary category (the pairing rule's first key).
    assert {c.primary_category for c in load_pool(corpus)} == {"cs.CL"}


def test_a_golden_set_that_disagrees_with_the_database_stops_the_tool(
    corpus: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    monkeypatch.setattr(make_testset, "GOLDEN_PATH", _stale_golden(tmp_path))
    assert main(["plan"]) == 2
    assert "not in the database" in capsys.readouterr().err

"""The test-set tool's reads from Postgres, on a scratch database with invented papers:
the pool of full-text papers, the ingest suggestions, and the commands that refuse to
generate from too few papers or only estimate (no key, no network, no model call)."""

import pytest
from sqlalchemy.orm import Session

from app.ingest.chunk import Chunk, abstract_chunk, chunk_id, embedding_text
from app.ingest.index import save_chunks
from eval.make_testset import ingest_suggestions, load_pool, main, papers_of
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
def corpus(migrated_session: Session) -> Session:
    _fulltext_paper(migrated_session, "2601.00001", 3)
    _fulltext_paper(migrated_session, "2601.00002", 2)
    for i, upvotes in enumerate((None, 50, 7), start=3):  # abstract-only papers
        paper = make_paper(migrated_session, f"2601.0000{i}", upvotes=upvotes, age_days=i)
        save_chunks(
            migrated_session, paper.arxiv_id,
            [abstract_chunk(paper.arxiv_id, 1, paper.title, paper.abstract)],
        )  # fmt: skip
        migrated_session.commit()
    return migrated_session


def test_the_pool_is_the_full_text_papers_sections_and_abstracts(corpus: Session) -> None:
    pool = load_pool(corpus)
    assert papers_of(pool) == ["2601.00001", "2601.00002"]
    assert sorted((c.arxiv_id, c.chunk_type) for c in pool).count(("2601.00001", "section")) == 3
    assert {c.chunk_type for c in pool} == {"abstract", "section"}
    assert all(c.title.startswith("Invented Study") for c in pool)


def test_ingest_suggestions_put_the_most_upvoted_abstract_only_papers_first(
    corpus: Session,
) -> None:
    ids = [arxiv_id for arxiv_id, _, _ in ingest_suggestions(corpus, 3)]
    assert ids == ["2601.00004", "2601.00005", "2601.00003"]  # 50, 7, then no upvotes


def test_generate_refuses_with_too_few_papers_and_lists_suggestions(
    corpus: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--min-papers", "4"]) == 3
    out = capsys.readouterr()
    assert "full-text papers: 2" in out.out and "missing 2" in out.out
    assert "2601.00004" in out.out and "Nothing was generated" in out.err


def test_the_estimate_reads_postgres_only(
    corpus: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--estimate", "--min-papers", "2", "--fact", "2", "--compare", "1",
                 "--filter", "1", "--unanswerable", "1"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "5 seeds from 2 papers" in out and "Nothing was sent" in out

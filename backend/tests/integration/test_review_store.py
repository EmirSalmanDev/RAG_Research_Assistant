"""The review against a scratch database: the real store, fake model. The citations must
resolve to real chunks of the right paper and evidence level, and the BibTeX keys must be
the ones of the .bib export. All text is invented."""

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, PaperAnalysis
from app.features import library
from app.features.bibtex import build_entries, render_bibtex
from app.features.review import (
    DraftOut,
    DraftParagraph,
    DraftSection,
    DraftSentence,
    ReviewDeps,
    SqlReviewStore,
    run_review,
)
from app.ingest.chunk import abstract_chunk, analysis_chunks
from app.ingest.index import save_chunks
from tests.bib_validate import keys
from tests.chat_fakes import fake_chat, parsed_reply
from tests.digest_fakes import make_paper
from tests.tracing_fakes import RecordingTracer

pytestmark = pytest.mark.integration

FIELDS: dict[str, str | None] = {
    "tldr": "Invented tldr.",
    "contributions": "Invented contributions.",
    "method": "Invented method.",
    "results": "Invented results.",
    "limitations": "Invented limitations.",
}


@pytest.fixture
def s(migrated_session: Session) -> Session:
    for i, (title, authors) in enumerate(
        [
            ("Widgets Reconsidered", [{"name": "Ada Example"}]),
            ("Widgets Again", [{"name": "Ada Example"}, {"name": "Bo Sample"}]),
            ("Gadgets", [{"name": "Cy Third"}]),
        ],
        start=1,
    ):
        paper = make_paper(migrated_session, f"2610.0000{i}", age_days=10 - i, title=title)
        paper.authors = authors
        save_chunks(migrated_session, paper.arxiv_id, [
            abstract_chunk(paper.arxiv_id, 1, title, paper.abstract)
        ])  # fmt: skip
    # only the first paper has an analysis, with its chunks
    migrated_session.add(PaperAnalysis(arxiv_id="2610.00001", model="invented", **FIELDS))
    save_chunks(migrated_session, "2610.00001", analysis_chunks("2610.00001", 1, "Widgets", FIELDS))
    migrated_session.commit()
    for paper_id in ("2610.00003", "2610.00001", "2610.00002"):
        library.add_item(migrated_session, paper_id, "Thesis")
    migrated_session.commit()
    return migrated_session


def test_the_real_store_gives_chunks_of_the_right_paper_and_level(s: Session) -> None:
    ok = DraftOut(
        sections=[
            DraftSection(
                heading="Widgets",
                paragraphs=[
                    DraftParagraph(
                        sentences=[
                            DraftSentence(
                                text="Example studied widgets.", kind="claim", source_numbers=[1]
                            ),
                            DraftSentence(
                                text="Others looked at gadgets.",
                                kind="claim",
                                source_numbers=[2, 3],
                            ),
                        ]  # fmt: skip
                    )
                ],
            )
        ]
    )
    chat, completions, rows = fake_chat([parsed_reply(ok)])
    deps = ReviewDeps(chat, SqlReviewStore(), RecordingTracer())
    result = run_review("Thesis", deps=deps)

    assert [r.arxiv_id for r in result.references] == ["2610.00001", "2610.00002", "2610.00003"]
    assert [r.evidence for r in result.references] == ["analysis", "abstract", "abstract"]
    for ref in result.references:
        chunk = s.get(Chunk, ref.chunk_id)
        assert chunk is not None
        assert chunk.arxiv_id == ref.arxiv_id and chunk.chunk_type == ref.chunk_type
    # the analysis reference opens the paper's own analysis chunk (the summary, ord 10000)
    first = s.get(Chunk, result.references[0].chunk_id)
    assert first is not None and first.ord == 10000 and "Invented tldr." in first.text
    assert result.references[0].section_path == first.section_path == "Analysis > Summary"
    assert result.references[1].section_path == "Abstract"
    assert result.counts.papers_without_analysis == 2 and result.counts.dropped_uncited == 0
    body = completions.calls[0]["messages"][1]["content"]
    assert "TL;DR: Invented tldr." in body and "Abstract: An invented abstract." in body
    assert [r[0] for r in rows] == ["review-write"]

    # the keys are those of the .bib export of the same collection
    bib = render_bibtex(build_entries(library.collection_bib_papers(s, "Thesis")))
    assert [r.bibtex_key for r in result.references] == keys(bib)
    assert len(set(keys(bib))) == 3


def test_an_unknown_collection_and_a_small_one_are_refused(s: Session) -> None:
    chat, completions, _ = fake_chat([])
    deps = ReviewDeps(chat, SqlReviewStore(), RecordingTracer())
    from app.features.errors import InvalidRequest, UnknownCollection

    with pytest.raises(UnknownCollection):
        run_review("Nope", deps=deps)
    library.add_item(s, "2610.00001", "Tiny")
    s.commit()
    with pytest.raises(InvalidRequest):
        run_review("Tiny", deps=deps)
    assert completions.calls == []


def test_the_chunks_are_unchanged_by_a_review(s: Session) -> None:
    before = s.scalars(select(Chunk.id).order_by(Chunk.id)).all()
    chat, _, _ = fake_chat([parsed_reply(DraftOut(sections=[]))])
    run_review("Thesis", deps=ReviewDeps(chat, SqlReviewStore(), RecordingTracer()))
    s.expire_all()
    assert s.scalars(select(Chunk.id).order_by(Chunk.id)).all() == before

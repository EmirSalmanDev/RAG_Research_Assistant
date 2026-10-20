"""`make review` on a scratch database: the estimate reads Postgres only (no key, no network),
it is an upper bound of what a real run sends, and the paid path refuses cleanly."""

import json

import pytest
from sqlalchemy.orm import Session

from app.db.models import PaperAnalysis
from app.features import library
from app.features.review import MapOut, ReviewDeps, SqlReviewStore, ThemeNote, run_review
from app.features.review_cli import estimate_review, main
from app.ingest.chunk import abstract_chunk, analysis_chunks
from app.ingest.index import save_chunks
from tests.bib_validate import validate
from tests.chat_fakes import fake_chat, parsed_reply
from tests.digest_fakes import make_paper
from tests.review_fakes import draft, sentence
from tests.tracing_fakes import RecordingTracer

pytestmark = pytest.mark.integration

FIELDS: dict[str, str | None] = {
    "tldr": "Invented tldr.",
    "contributions": "Invented contributions.",
    "method": "Invented method.",
    "results": "Invented results.",
    "limitations": "Invented limitations.",
}


def _collection(s: Session, count: int, name: str = "Thesis") -> None:
    for i in range(1, count + 1):
        paper = make_paper(s, f"2610.{i:05d}", age_days=100 - i, title=f"Invented Study {i}")
        paper.authors = [{"name": f"Ada Author{chr(96 + i)}"}]
        save_chunks(
            s, paper.arxiv_id, [abstract_chunk(paper.arxiv_id, 1, paper.title, paper.abstract)]
        )
        if i % 2:
            s.add(PaperAnalysis(arxiv_id=paper.arxiv_id, model="invented", **FIELDS))
            save_chunks(s, paper.arxiv_id, analysis_chunks(paper.arxiv_id, 1, paper.title, FIELDS))
        s.commit()
        library.add_item(s, paper.arxiv_id, name)
        s.commit()


def test_the_estimate_of_a_small_collection_is_exactly_the_one_call_that_would_be_made(
    migrated_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    _collection(migrated_session, 4)
    assert main(["Thesis", "--estimate", "--focus", "widgets"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["requests"].startswith("1 chat") and out["papers"] == 4
    assert out["papers_without_analysis"] == 2 and out["chat_output_tokens_max"] == 4000

    chat, completions, rows = fake_chat([parsed_reply(draft(), 1, 1)])
    run_review("Thesis", "widgets", deps=ReviewDeps(chat, SqlReviewStore(), RecordingTracer()))
    from app.features.review import input_tokens

    sent = input_tokens(completions.calls[0]["messages"])
    assert out["chat_input_tokens_max"] == sent
    assert 0 < out["cost_usd_max"] < 0.01


def test_the_estimate_of_a_big_collection_bounds_the_real_input(
    migrated_session: Session,
) -> None:
    _collection(migrated_session, 12)
    unused_chat, _, _ = fake_chat([])
    deps = ReviewDeps(unused_chat, SqlReviewStore(), RecordingTracer())
    estimate = estimate_review("Thesis", None, None, deps)
    assert str(estimate["requests"]).startswith("3 chat (map 2, write 1)")

    # a real run with the largest notes the model could return (the output cap)
    notes = MapOut(notes=[ThemeNote(theme="T", statement="word " * 55, source_numbers=[1])] * 10)
    ok = draft(("S", [[sentence("A claim.", "claim", 1)]]))
    chat, completions, rows = fake_chat(
        [parsed_reply(notes), parsed_reply(notes), parsed_reply(ok)]
    )
    run_review("Thesis", None, deps=ReviewDeps(chat, SqlReviewStore(), RecordingTracer()))
    from app.features.review import input_tokens

    sent = sum(input_tokens(c["messages"]) for c in completions.calls)
    assert sent <= int(str(estimate["chat_input_tokens_max"]))


def test_bib_prints_the_collections_file(
    migrated_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    _collection(migrated_session, 3)
    assert main(["Thesis", "--bib"]) == 0
    text = capsys.readouterr().out
    assert validate(text) == [] and text.count("@misc{") == 3


def test_refusals_have_exit_codes_and_a_paid_run_needs_a_key(
    migrated_session: Session, capsys: pytest.CaptureFixture[str]
) -> None:
    _collection(migrated_session, 3)
    assert main(["Nope", "--estimate"]) == 2
    assert main(["Thesis", "--estimate", "--max-papers", "0"]) == 2
    assert main(["Thesis", "--bib", "--max-papers", "0"]) == 0  # --bib ignores it
    assert main(["Thesis"]) == 3  # no key in the tests
    assert "OPENAI_API_KEY" in capsys.readouterr().err

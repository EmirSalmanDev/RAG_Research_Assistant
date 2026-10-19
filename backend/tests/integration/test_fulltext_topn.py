"""Full-text selection by relevance: library and requested papers first, then the top N of
the window. Selection only; nothing is fetched. All data is invented."""

import pytest
from sqlalchemy.orm import Session

from app.db.models import DigestEntry, LibraryItem, Paper, PaperText
from app.features.digest import today_in_zone
from app.ingest.fulltext import select_for_fulltext
from tests.digest_fakes import NOW, make_paper

pytestmark = pytest.mark.integration


def _scored(s: Session, arxiv_id: str, score: float | None, **kw: object) -> Paper:
    paper = make_paper(s, arxiv_id, **kw)  # type: ignore[arg-type]
    paper.relevance_score = score
    s.commit()
    return paper


def test_top_n_by_score_then_newer_then_id(migrated_session: Session) -> None:
    s = migrated_session
    _scored(s, "2610.00001", 0.5, age_days=2)
    _scored(s, "2610.00002", 0.9, age_days=3)
    _scored(s, "2610.00003", 0.5, age_days=1)  # ties 00001, newer
    _scored(s, "2610.00004", 0.5, age_days=1)  # ties 00003 exactly: smaller id first
    _scored(s, "2610.00005", 0.1, age_days=1)
    assert select_for_fulltext(s, top_n=4, now=NOW) == [
        "2610.00002", "2610.00003", "2610.00004", "2610.00001",
    ]  # fmt: skip
    assert select_for_fulltext(s, top_n=2, now=NOW) == ["2610.00002", "2610.00003"]


def test_top_n_zero_keeps_the_old_rule(migrated_session: Session) -> None:
    s = migrated_session
    _scored(s, "2610.00001", 0.9)
    assert select_for_fulltext(s, now=NOW) == []
    assert select_for_fulltext(s, top_n=0, now=NOW) == []


def test_library_and_requested_come_first_and_are_not_repeated(migrated_session: Session) -> None:
    s = migrated_session
    _scored(s, "2610.00009", None, age_days=40)  # in the library, old and unscored
    _scored(
        s,
        "2610.00008",
        None,
        age_days=40,
    )
    s.get(Paper, "2610.00008").fulltext_status = "requested"  # type: ignore[union-attr]
    _scored(s, "2610.00001", 0.9)
    _scored(s, "2610.00002", 0.8)
    s.add(LibraryItem(arxiv_id="2610.00009", collection="Saved"))
    s.add(LibraryItem(arxiv_id="2610.00002", collection="Saved"))  # also in the top N
    s.commit()
    assert select_for_fulltext(s, top_n=2, now=NOW) == [
        "2610.00002", "2610.00008", "2610.00009",  # library / requested, by id
        "2610.00001",  # then the top N that is not yet chosen
    ]  # fmt: skip


def test_the_top_n_is_taken_before_removing_papers_that_already_have_text(
    migrated_session: Session,
) -> None:
    """Day 2: yesterday's picks hold text and stay in the top N, so only a paper that newly
    entered the top N is selected, not the next 20 in line."""
    s = migrated_session
    for i, score in enumerate((0.9, 0.8, 0.7, 0.6), start=1):
        _scored(s, f"2610.0000{i}", score)
    s.add(
        PaperText(arxiv_id="2610.00001", version=1, raw_markdown="x", markdown="x", token_count=1)
    )
    s.commit()
    assert select_for_fulltext(s, top_n=2, now=NOW) == ["2610.00002"]  # not 00003 / 00004


def test_dismissed_unscored_old_failed_and_unavailable_papers_are_left_out(
    migrated_session: Session,
) -> None:
    s = migrated_session
    _scored(s, "2610.00001", 0.99)  # will be dismissed
    _scored(s, "2610.00002", None)  # never scored
    _scored(s, "2609.00001", 0.99, age_days=30)  # outside the window
    _scored(s, "2610.00003", 0.98).fulltext_status = "failed"
    _scored(s, "2610.00004", 0.97).fulltext_status = "unavailable"
    _scored(s, "2610.00005", 0.5)
    s.add(DigestEntry(digest_date=today_in_zone(NOW), arxiv_id="2610.00001", rank=1, score=1.0,
                      status="dismissed"))  # fmt: skip
    s.commit()
    # failed / unavailable still use up places in the top 5: the N best are chosen first
    assert select_for_fulltext(s, top_n=5, now=NOW) == ["2610.00005"]

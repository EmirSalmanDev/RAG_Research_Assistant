"""Scoring a window, building the digest, the Save / Dismiss / Undo state machine and
reading the digest, on a scratch database with fake vectors. All data is invented."""

from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import DigestEntry, Interest, LibraryItem, Paper, PaperAnalysis
from app.features import app_settings as cfg
from app.features import digest
from app.features.digest import apply_action, build_digest, load_digest, score_window
from tests.digest_fakes import NOW, FakeDense, FakeVectors, make_interest, make_paper

pytestmark = pytest.mark.integration

TODAY = digest.today_in_zone(NOW)


def _ids(result: digest.BuildResult) -> list[str]:
    return [e.arxiv_id for e in result.entries]


def _weight(session: Session, interest_id: int) -> float:
    session.expire_all()
    interest = session.get(Interest, interest_id)
    assert interest is not None
    return interest.weight


@pytest.fixture
def world(migrated_session: Session) -> tuple[Session, FakeVectors, Interest, Interest]:
    """Two interests (one near [1,0], one near [0,1]) and five papers in the window."""
    s = migrated_session
    near_a = make_interest(s, "Alpha", embedding=[1.0, 0.0])
    near_b = make_interest(s, "Beta", embedding=[0.0, 1.0])
    vectors = FakeVectors()
    for arxiv_id, vec, age in (
        ("2610.00001", [1.0, 0.0], 1),  # perfect for Alpha
        ("2610.00002", [0.0, 1.0], 2),  # perfect for Beta
        ("2610.00003", [1.0, 1.0], 1),  # in between
        ("2610.00004", [1.0, 0.1], 3),  # almost Alpha
        ("2610.00005", [0.1, 1.0], 1),  # almost Beta
    ):
        make_paper(s, arxiv_id, age_days=age)
        vectors.vectors[arxiv_id] = vec
    return s, vectors, near_a, near_b


# --- scoring ----------------------------------------------------------------------------------


def test_score_window_stores_scores_for_the_window_only_and_is_idempotent(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> None:
    s, vectors, _, _ = world
    make_paper(s, "2609.00099", age_days=30)  # outside the 7-day window
    first = score_window(s, vectors, NOW)
    s.commit()
    assert first["papers"] == 5 and first["interests"] == 2 and first["missing_vectors"] == 0
    scores = dict(s.execute(select(Paper.arxiv_id, Paper.relevance_score)).all())
    assert scores["2609.00099"] is None  # not scored
    assert scores["2610.00001"] == pytest.approx(0.8)  # cos 1 with Alpha, nothing else
    assert scores["2610.00003"] == pytest.approx(0.8 * 0.7071067811865475)
    assert score_window(s, vectors, NOW) == first  # same counts ...
    s.commit()
    assert dict(s.execute(select(Paper.arxiv_id, Paper.relevance_score)).all()) == scores


def test_scoring_does_not_touch_updated_at(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> None:
    s, vectors, _, _ = world
    before = s.scalar(select(Paper.updated_at).where(Paper.arxiv_id == "2610.00001"))
    score_window(s, vectors, NOW)
    s.commit()
    assert s.scalar(select(Paper.updated_at).where(Paper.arxiv_id == "2610.00001")) == before


def test_zero_interests_scores_by_upvotes_only_and_reads_no_vectors(
    migrated_session: Session,
) -> None:
    s = migrated_session
    make_paper(s, "2610.00001", upvotes=100)
    make_paper(s, "2610.00002", upvotes=None)
    vectors = FakeVectors()
    result = score_window(s, vectors, NOW)
    s.commit()
    assert result["interests"] == 0 and vectors.calls == []
    scores = dict(s.execute(select(Paper.arxiv_id, Paper.relevance_score)).all())
    assert scores == {"2610.00001": pytest.approx(get_settings().relevance_upvote_boost),
                      "2610.00002": 0.0}  # fmt: skip


def test_inactive_interests_do_not_count(migrated_session: Session) -> None:
    s = migrated_session
    make_interest(s, "Off", embedding=[1.0, 0.0], active=False)
    make_paper(s, "2610.00001")
    result = score_window(s, FakeVectors({"2610.00001": [1.0, 0.0]}), NOW)
    assert result["interests"] == 0


def test_a_paper_without_a_vector_is_scored_without_one(migrated_session: Session) -> None:
    s = migrated_session
    make_interest(s, "Kw", embedding=[1.0, 0.0], keywords=["invented"])
    make_paper(s, "2610.00001", abstract="An invented abstract.")
    result = score_window(s, FakeVectors(), NOW)  # Qdrant has no point for it
    s.commit()
    assert result["missing_vectors"] == 1
    score = s.scalar(select(Paper.relevance_score))
    assert score == pytest.approx(get_settings().relevance_keyword_weight)


# --- interest embeddings --------------------------------------------------------------------------


def test_embeddings_are_made_once_for_stale_active_interests_only(
    migrated_session: Session,
) -> None:
    s = migrated_session
    new = make_interest(s, "New")
    edited = make_interest(s, "Edited", embedding=[0.3, 0.3])
    make_interest(s, "Fresh", embedding=[0.5, 0.5])
    make_interest(s, "Off", active=False)
    edited.keywords = ["changed"]  # its text changed: the stored vector is stale
    s.commit()

    dense = FakeDense()
    assert digest.ensure_interest_embeddings(s, dense) == 2
    s.commit()
    assert len(dense.calls) == 1  # one request for both
    texts, feature = dense.calls[0]
    assert feature == "embed-interest" and len(texts) == 2
    assert any("Keywords: changed" in t for t in texts)
    s.refresh(new)
    assert new.embedding == [1.0, 0.0] and new.embedding_model == get_settings().embedding_model

    assert digest.ensure_interest_embeddings(s, dense) == 0  # nothing stale: no request
    assert len(dense.calls) == 1


# --- building the digest -------------------------------------------------------------------------


def test_the_digest_is_ranked_with_a_best_interest(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> None:
    s, vectors, alpha, beta = world
    result = build_digest(s, vectors, now=NOW)
    s.commit()
    assert result.status == "built" and result.digest_date == TODAY
    # perfect matches first (newer 00001 before older 00002), then the near ones
    assert _ids(result)[:2] == ["2610.00001", "2610.00002"]
    assert [e.rank for e in result.entries] == [1, 2, 3, 4, 5]
    assert [e.score for e in result.entries] == sorted(
        (e.score for e in result.entries), reverse=True
    )
    by_id = {e.arxiv_id: e.best_interest_id for e in result.entries}
    assert by_id["2610.00001"] == alpha.id and by_id["2610.00002"] == beta.id
    assert by_id["2610.00004"] == alpha.id and by_id["2610.00005"] == beta.id
    rows = s.scalars(select(DigestEntry).order_by(DigestEntry.rank)).all()
    assert [r.arxiv_id for r in rows] == _ids(result)
    assert {r.status for r in rows} == {"new"}


def test_the_digest_holds_digest_size_papers_from_the_settings(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> None:
    s, vectors, _, _ = world
    cfg.save(s, {"digest_size": 3})
    s.commit()
    assert len(build_digest(s, vectors, now=NOW).entries) == 3


def test_a_second_build_for_the_same_day_changes_nothing(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> None:
    s, vectors, _, _ = world
    first = build_digest(s, vectors, now=NOW)
    s.commit()
    snapshot = [(r.arxiv_id, r.rank, r.score) for r in s.scalars(select(DigestEntry))]
    calls = len(vectors.calls)
    again = build_digest(s, vectors, now=NOW)
    s.commit()
    assert again.status == "exists" and _ids(again) == _ids(first)
    assert len(vectors.calls) == calls  # it did not even score again
    assert [(r.arxiv_id, r.rank, r.score) for r in s.scalars(select(DigestEntry))] == snapshot


def test_dry_run_writes_nothing(world: tuple[Session, FakeVectors, Interest, Interest]) -> None:
    s, vectors, _, _ = world
    result = build_digest(s, vectors, now=NOW, dry_run=True)
    s.commit()
    assert len(result.entries) == 5
    assert s.scalar(select(DigestEntry.id)) is None
    assert all(p.relevance_score is None for p in s.scalars(select(Paper)))


def test_dismissed_and_earlier_papers_never_come_back(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> None:
    s, vectors, _, _ = world
    yesterday = TODAY - timedelta(days=1)
    s.add_all([
        DigestEntry(digest_date=yesterday, arxiv_id="2610.00001", rank=1, score=1.0),
        DigestEntry(digest_date=yesterday, arxiv_id="2610.00002", rank=2, score=1.0,
                    status="dismissed"),
        DigestEntry(digest_date=TODAY + timedelta(days=1), arxiv_id="2610.00003", rank=1,
                    score=1.0, status="dismissed"),  # dismissed on any date counts
    ])  # fmt: skip
    s.commit()
    result = build_digest(s, vectors, now=NOW)
    assert sorted(_ids(result)) == ["2610.00004", "2610.00005"]


def test_papers_outside_the_window_are_not_candidates(migrated_session: Session) -> None:
    s = migrated_session
    make_paper(s, "2610.00001", age_days=1)
    make_paper(s, "2609.00001", age_days=8)
    make_paper(s, "2610.00009", age_days=-1)  # published after "now": not yet
    assert _ids(build_digest(s, FakeVectors(), now=NOW)) == ["2610.00001"]


def test_no_interests_ranks_by_upvotes_then_newer_then_id(migrated_session: Session) -> None:
    s = migrated_session
    make_paper(s, "2610.00004", age_days=2, upvotes=50)
    make_paper(s, "2610.00003", age_days=1, upvotes=None)
    make_paper(s, "2610.00002", age_days=1, upvotes=None)  # ties 00003: same age, smaller id
    make_paper(s, "2610.00001", age_days=2, upvotes=None)
    result = build_digest(s, FakeVectors(), now=NOW)
    assert _ids(result) == ["2610.00004", "2610.00002", "2610.00003", "2610.00001"]
    assert {e.best_interest_id for e in result.entries} == {None}


def test_an_empty_window_gives_an_empty_digest(migrated_session: Session) -> None:
    result = build_digest(migrated_session, FakeVectors(), now=NOW)
    assert result.status == "empty" and result.entries == []


def test_an_earlier_date_is_built_from_what_that_day_could_see(migrated_session: Session) -> None:
    s = migrated_session
    make_paper(s, "2610.00001", age_days=0.5)  # published 12 h before NOW (2026-10-18 17:00 UTC)
    old_day = TODAY - timedelta(days=2)
    assert build_digest(s, FakeVectors(), now=NOW, digest_date=old_day).entries == []
    assert _ids(build_digest(s, FakeVectors(), now=NOW, digest_date=TODAY)) == ["2610.00001"]


def test_rebuild_replaces_new_entries_and_keeps_acted_on_ones(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> None:
    s, vectors, alpha, _ = world
    build_digest(s, vectors, now=NOW)
    s.commit()
    apply_action(s, TODAY, "2610.00004", "save")  # an entry lower down is saved
    apply_action(s, TODAY, "2610.00001", "dismiss")
    s.commit()
    make_paper(s, "2610.00006", age_days=0.1)
    vectors.vectors["2610.00006"] = [1.0, 0.0]  # a new, perfect match for Alpha

    result = build_digest(s, vectors, now=NOW, rebuild=True)
    s.commit()
    assert result.kept == 2
    rows = {r.arxiv_id: r for r in s.scalars(select(DigestEntry))}
    assert rows["2610.00004"].status == "saved" and rows["2610.00001"].status == "dismissed"
    assert "2610.00006" in rows  # the newcomer is in
    assert sorted(r.rank for r in rows.values()) == list(range(1, len(rows) + 1))
    assert len(rows) == 6  # 5 + 1; nothing duplicated (the unique key would have failed)
    # the kept entries keep their weight changes
    assert _weight(s, alpha.id) == pytest.approx(1.0 + 0.10 - 0.05)


# --- Save, Dismiss, Undo -----------------------------------------------------------------------


@pytest.fixture
def built(
    world: tuple[Session, FakeVectors, Interest, Interest],
) -> tuple[Session, Interest, Interest]:
    s, vectors, alpha, beta = world
    build_digest(s, vectors, now=NOW)
    s.commit()
    return s, alpha, beta


def _entry(s: Session, arxiv_id: str) -> DigestEntry:
    s.expire_all()
    row = s.scalar(select(DigestEntry).where(DigestEntry.arxiv_id == arxiv_id))
    assert row is not None
    return row


def test_save_nudges_the_best_interest_and_adds_to_the_library(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, alpha, beta = built
    apply_action(s, TODAY, "2610.00001", "save")
    s.commit()
    entry = _entry(s, "2610.00001")
    assert entry.status == "saved" and entry.weight_delta == pytest.approx(0.10)
    assert entry.actioned_at is not None
    assert _weight(s, alpha.id) == pytest.approx(1.10) and _weight(s, beta.id) == 1.0
    item = s.get(LibraryItem, ("2610.00001", "Saved"))
    assert item is not None and item.status == "to_read"


def test_dismiss_lowers_the_weight_and_leaves_the_library_alone(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, _, beta = built
    apply_action(s, TODAY, "2610.00002", "dismiss")
    s.commit()
    entry = _entry(s, "2610.00002")
    assert entry.status == "dismissed" and entry.weight_delta == pytest.approx(-0.05)
    assert _weight(s, beta.id) == pytest.approx(0.95)
    assert s.scalar(select(LibraryItem.arxiv_id)) is None


def test_repeating_an_action_never_counts_twice(built: tuple[Session, Interest, Interest]) -> None:
    s, alpha, _ = built
    for _ in range(3):
        apply_action(s, TODAY, "2610.00001", "save")
        s.commit()
    assert _weight(s, alpha.id) == pytest.approx(1.10)
    apply_action(s, TODAY, "2610.00004", "dismiss")  # also Alpha's best paper
    apply_action(s, TODAY, "2610.00004", "dismiss")
    s.commit()
    assert _weight(s, alpha.id) == pytest.approx(1.05)
    for _ in range(3):  # undo on a new entry changes nothing either
        apply_action(s, TODAY, "2610.00003", "undo")
    assert _weight(s, alpha.id) == pytest.approx(1.05)


def test_undo_gives_back_exactly_the_delta(built: tuple[Session, Interest, Interest]) -> None:
    s, alpha, _ = built
    apply_action(s, TODAY, "2610.00001", "save")
    s.commit()
    apply_action(s, TODAY, "2610.00001", "undo")
    s.commit()
    entry = _entry(s, "2610.00001")
    assert (entry.status, entry.weight_delta, entry.actioned_at) == ("new", 0.0, None)
    assert _weight(s, alpha.id) == 1.0  # exactly, not approximately
    assert s.get(LibraryItem, ("2610.00001", "Saved")) is None

    apply_action(s, TODAY, "2610.00001", "dismiss")
    apply_action(s, TODAY, "2610.00001", "undo")
    s.commit()
    assert _weight(s, alpha.id) == 1.0


def test_float_noise_does_not_build_up_over_many_round_trips(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, alpha, _ = built
    for _ in range(50):
        apply_action(s, TODAY, "2610.00001", "save")
        apply_action(s, TODAY, "2610.00001", "undo")
        apply_action(s, TODAY, "2610.00001", "dismiss")
        apply_action(s, TODAY, "2610.00001", "undo")
    s.commit()
    assert _weight(s, alpha.id) == 1.0


def test_switching_between_save_and_dismiss_reverses_the_first_nudge(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, alpha, _ = built
    apply_action(s, TODAY, "2610.00001", "dismiss")
    apply_action(s, TODAY, "2610.00001", "save")
    s.commit()
    assert _weight(s, alpha.id) == pytest.approx(1.10)  # not 1.05
    assert _entry(s, "2610.00001").status == "saved"
    apply_action(s, TODAY, "2610.00001", "dismiss")
    s.commit()
    assert _weight(s, alpha.id) == pytest.approx(0.95)
    assert s.get(LibraryItem, ("2610.00001", "Saved")) is None  # leaving "saved" removes it


def test_weights_stop_at_the_limits_and_undo_returns_what_was_applied(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, alpha, beta = built
    alpha.weight = 1.95
    beta.weight = 0.52
    s.commit()
    apply_action(s, TODAY, "2610.00001", "save")  # +0.10 but only 0.05 fits
    apply_action(s, TODAY, "2610.00002", "dismiss")  # -0.05 but only 0.02 fits
    s.commit()
    assert _weight(s, alpha.id) == 2.0 and _entry(s, "2610.00001").weight_delta == pytest.approx(
        0.05
    )
    assert _weight(s, beta.id) == 0.5 and _entry(s, "2610.00002").weight_delta == pytest.approx(
        -0.02
    )
    apply_action(s, TODAY, "2610.00001", "undo")
    apply_action(s, TODAY, "2610.00002", "undo")
    s.commit()
    assert _weight(s, alpha.id) == 1.95 and _weight(s, beta.id) == 0.52

    alpha.weight = 2.0
    s.commit()
    apply_action(s, TODAY, "2610.00001", "save")  # already at the limit: nothing to record
    s.commit()
    assert _entry(s, "2610.00001").weight_delta == 0.0 and _weight(s, alpha.id) == 2.0


def test_saving_an_entry_whose_interest_was_deleted_still_works(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, alpha, _ = built
    s.delete(alpha)
    s.commit()
    assert _entry(s, "2610.00001").best_interest_id is None
    apply_action(s, TODAY, "2610.00001", "save")
    s.commit()
    entry = _entry(s, "2610.00001")
    assert (entry.status, entry.weight_delta) == ("saved", 0.0)
    apply_action(s, TODAY, "2610.00001", "undo")
    assert _entry(s, "2610.00001").status == "new"


def test_undo_keeps_a_library_item_the_owner_has_edited(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, _, _ = built
    apply_action(s, TODAY, "2610.00001", "save")
    s.commit()
    item = s.get(LibraryItem, ("2610.00001", "Saved"))
    assert item is not None
    item.notes = "read this first"
    s.commit()
    apply_action(s, TODAY, "2610.00001", "undo")
    s.commit()
    assert s.get(LibraryItem, ("2610.00001", "Saved")) is not None


def test_a_saved_paper_is_picked_for_full_text(built: tuple[Session, Interest, Interest]) -> None:
    from app.ingest.fulltext import select_for_fulltext

    s, _, _ = built
    assert "2610.00001" not in select_for_fulltext(s)
    apply_action(s, TODAY, "2610.00001", "save")
    s.commit()
    assert "2610.00001" in select_for_fulltext(s)  # in the library = needs full text


def test_an_unknown_entry_is_an_error(built: tuple[Session, Interest, Interest]) -> None:
    s, _, _ = built
    with pytest.raises(digest.EntryNotFoundError):
        apply_action(s, TODAY, "2610.99999", "save")
    with pytest.raises(digest.EntryNotFoundError):
        apply_action(s, TODAY - timedelta(days=1), "2610.00001", "save")


# --- reading the digest ---------------------------------------------------------------------------


def test_load_digest_falls_back_to_the_latest_available(
    built: tuple[Session, Interest, Interest],
) -> None:
    s, _, _ = built
    view = load_digest(s, TODAY + timedelta(days=3), NOW)  # nothing for that day
    assert view.digest_date == TODAY and len(view.cards) == 5
    assert view.requested_date == TODAY + timedelta(days=3)
    older = load_digest(s, TODAY - timedelta(days=5), NOW)  # before the first digest
    assert older.digest_date == TODAY  # the newest of all, rather than nothing
    assert view.has_interests and view.recent_dates == [TODAY]


def test_load_digest_without_any_digest_is_empty_and_builds_nothing(
    migrated_session: Session,
) -> None:
    s = migrated_session
    make_paper(s, "2610.00001")
    view = load_digest(s, None, NOW)
    assert (view.digest_date, view.cards, view.has_interests) == (None, [], False)
    assert s.scalar(select(DigestEntry.id)) is None  # reading never builds


def test_cards_carry_summary_trending_and_interest_name(migrated_session: Session) -> None:
    s = migrated_session
    make_interest(s, "Alpha", embedding=[1.0, 0.0])
    make_paper(s, "2610.00001", upvotes=10, abstract="First sentence here. Second one follows.")
    make_paper(s, "2610.00002", upvotes=9, abstract="Other abstract. More.")
    s.add(PaperAnalysis(arxiv_id="2610.00002", tldr="The stored TL;DR.", model="m"))
    s.commit()
    vectors = FakeVectors({"2610.00001": [1.0, 0.0], "2610.00002": [1.0, 0.0]})
    build_digest(s, vectors, now=NOW)
    s.commit()
    cards = {c.paper.arxiv_id: c for c in load_digest(s, TODAY, NOW).cards}
    one, two = cards["2610.00001"], cards["2610.00002"]
    assert (one.summary, one.summary_source) == ("First sentence here.", "abstract")
    assert (two.summary, two.summary_source) == ("The stored TL;DR.", "analysis")
    assert one.trending is True and two.trending is False  # 10 >= 10, 9 < 10
    assert one.best_interest_name == "Alpha"

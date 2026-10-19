"""Interests CRUD and the stored settings, on a scratch database. All text is invented."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import LLMUsage
from app.features import app_settings as cfg
from app.features import interests as repo

pytestmark = pytest.mark.integration


def _make(s: Session, name: str = "Invented topic", **kw: object) -> int:
    row = repo.create_interest(
        s,
        name=name,
        description=kw.get("description", "About inventions."),  # type: ignore[arg-type]
        keywords=kw.get("keywords", ["alpha", "Beta", "ALPHA", "  "]),  # type: ignore[arg-type]
        categories=kw.get("categories", ["cs.CL"]),  # type: ignore[arg-type]
    )
    s.commit()
    return row.id


def test_create_normalises_keywords_and_sets_defaults(migrated_session: Session) -> None:
    s = migrated_session
    interest = repo.get_interest(s, _make(s))
    assert interest is not None
    assert interest.keywords == ["alpha", "Beta"]  # trimmed, blank dropped, repeat dropped
    assert (interest.weight, interest.is_active, interest.embedding) == (1.0, True, None)
    assert interest.created_at is not None
    assert not repo.has_current_embedding(interest)


def test_duplicate_names_are_refused(migrated_session: Session) -> None:
    s = migrated_session
    first = _make(s, "One")
    other = _make(s, "Two")
    with pytest.raises(repo.DuplicateInterestError):
        _make(s, "One")
    with pytest.raises(repo.DuplicateInterestError):  # a rename onto an existing name
        repo.update_interest(s, other, name="One", description=None, keywords=[], categories=[],
                             is_active=True)  # fmt: skip
    s.rollback()
    # renaming to its own name is fine
    updated = repo.update_interest(s, first, name="One", description="x", keywords=["k"],
                                   categories=[], is_active=False)  # fmt: skip
    assert updated is not None and updated.is_active is False


def test_edit_changes_the_embedding_hash_only_for_text_fields(migrated_session: Session) -> None:
    s = migrated_session
    interest_id = _make(s)
    interest = repo.get_interest(s, interest_id)
    assert interest is not None
    interest.embedding = [0.5] * 4
    interest.embedding_model = get_settings().embedding_model
    interest.embedding_hash = repo.current_hash(interest)
    s.commit()
    assert repo.has_current_embedding(interest)

    # categories / active flag: the embedded text is unchanged
    repo.update_interest(
        s,
        interest_id,
        name=interest.name,
        description=interest.description,
        keywords=list(interest.keywords),
        categories=["cs.LG"],
        is_active=False,
    )
    assert repo.has_current_embedding(interest)
    # keywords: the text changes, the stored vector is stale
    repo.update_interest(
        s,
        interest_id,
        name=interest.name,
        description=interest.description,
        keywords=["gamma"],
        categories=["cs.LG"],
        is_active=False,
    )
    assert not repo.has_current_embedding(interest)


def test_update_and_delete_unknown_ids(migrated_session: Session) -> None:
    s = migrated_session
    assert repo.update_interest(s, 999, name="x", description=None, keywords=[], categories=[],
                                is_active=True) is None  # fmt: skip
    assert repo.delete_interest(s, 999) is False
    interest_id = _make(s)
    assert repo.delete_interest(s, interest_id) is True
    assert repo.list_interests(s) == []


def test_embedding_text_and_hash() -> None:
    assert repo.embedding_text(" A ", None, []) == "A"
    assert repo.embedding_text("A", "B", ["c", "d"]) == "A\n\nB\n\nKeywords: c, d"
    assert repo.text_hash("m1", "t") != repo.text_hash("m2", "t")
    assert repo.text_hash("m1", "t") == repo.text_hash("m1", "t")


def test_settings_default_to_config_then_override_then_reset(migrated_session: Session) -> None:
    s = migrated_session
    base = cfg.load(s)
    config = get_settings()
    assert (base.digest_size, base.fulltext_top_n) == (config.digest_size, config.fulltext_top_n)
    assert base.categories == config.arxiv_categories and not base.categories_overridden
    assert base.budget_cap_usd is None

    cfg.save(s, {"digest_size": 5, "categories": ["cs.CV"], "budget_cap_usd": 7.5})
    s.commit()
    new = cfg.load(s)
    assert (new.digest_size, new.categories, new.budget_cap_usd) == (5, ["cs.CV"], 7.5)
    assert new.categories_overridden and cfg.arxiv_categories(s) == ["cs.CV"]

    cfg.save(s, {"digest_size": 6})  # a second save replaces the value, no duplicate row
    cfg.save(s, {"categories": None, "budget_cap_usd": None})  # None = back to the default
    s.commit()
    back = cfg.load(s)
    assert (back.digest_size, back.categories, back.budget_cap_usd) == (
        6, config.arxiv_categories, None,
    )  # fmt: skip
    assert s.execute(text("SELECT count(*) FROM app_settings")).scalar() == 1


def test_category_format() -> None:
    for good in ("cs.CL", "stat.ML", "hep-th", "astro-ph.GA", "physics.comp-ph", "cs"):
        assert cfg.is_valid_category(good), good
    for bad in ("", "CS.CL", "cs.", "cs.CL.X", "cs CL", "cs/CL", "cs.C1", "../x", "cs.CL\n"):
        assert not cfg.is_valid_category(bad), bad


def test_month_to_date_spend_counts_only_this_month(migrated_session: Session) -> None:
    s = migrated_session
    now = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    # Istanbul is UTC+3: this row is 30 Sep 22:00 UTC = 1 Oct 01:00 there, so it counts
    for when, cost in (
        (datetime(2026, 9, 30, 22, 0, tzinfo=UTC), "0.25"),
        (datetime(2026, 9, 30, 20, 59, tzinfo=UTC), "9.00"),  # 30 Sep 23:59 in Istanbul
        (datetime(2026, 10, 18, 5, 0, tzinfo=UTC), "0.5"),
    ):
        s.add(LLMUsage(feature="t", model="m", tokens_in=1, tokens_out=0, cost_usd=Decimal(cost),
                       created_at=when))  # fmt: skip
    s.commit()
    assert cfg.month_start(now) == datetime(2026, 9, 30, 21, 0, tzinfo=UTC)
    assert cfg.month_to_date_spend(s, now) == Decimal("0.75")

"""eval/scope.py: one configuration's settings for the whole process, and back again."""

import os

import pytest

from app.config import get_settings
from app.ingest.embed import get_dense_embedder
from eval.scope import ScopeError, applied_settings


def test_overrides_reach_every_reader_of_get_settings_and_are_undone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEARCH_MODE", raising=False)
    monkeypatch.setenv("SEARCH_CANDIDATES", "50")  # set before: must come back as it was
    before = get_settings()
    with applied_settings(
        {"search_mode": "bm25", "search_candidates": 20, "plan_rephrasings_enabled": False,
         "arxiv_categories": ["cs.CL"]}
    ) as s:  # fmt: skip
        assert s is get_settings()  # every module's get_settings() sees the same object
        assert (s.search_mode, s.search_candidates, s.plan_rephrasings_enabled) == (
            "bm25",
            20,
            False,
        )
        assert s.arxiv_categories == ["cs.CL"]
    after = get_settings()
    assert after is not s and after.search_mode == before.search_mode
    assert "SEARCH_MODE" not in os.environ
    assert os.environ["SEARCH_CANDIDATES"] == "50"


def test_settings_come_back_after_an_error_inside(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    with pytest.raises(RuntimeError, match="boom"):
        with applied_settings({"qdrant_collection": "chunks_variant"}):
            raise RuntimeError("boom")
    assert get_settings().qdrant_collection != "chunks_variant"
    assert "QDRANT_COLLECTION" not in os.environ
    with applied_settings({}):  # and a new scope can start
        pass


def test_the_shared_dense_embedder_is_rebuilt_for_the_new_model() -> None:
    outside = get_dense_embedder()
    with applied_settings({"embedding_model": "text-embedding-3-large", "embedding_dim": 3072}):
        inside = get_dense_embedder()
        assert inside is not outside
        assert (inside._model, inside._dim) == ("text-embedding-3-large", 3072)
    assert get_dense_embedder()._model == get_settings().embedding_model


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"no_such_setting": 1}, "unknown settings"),
        ({"database_url": "postgresql://elsewhere"}, "may not change"),
        ({"openai_api_key": "invented"}, "may not change"),
        ({"eval_model": "another-judge"}, "may not change"),  # the yardstick stays fixed
        ({"rerank_threads": None}, "cannot set"),
        ({"search_mode": "fuzzy"}, None),  # pydantic refuses the value itself
    ],
)
def test_wrong_overrides_change_nothing(overrides: dict[str, object], message: str | None) -> None:
    before = get_settings().model_dump()
    with pytest.raises(Exception, match=message):
        with applied_settings(overrides):
            pass
    assert get_settings().model_dump() == before
    assert "SEARCH_MODE" not in os.environ


def test_scopes_do_not_nest() -> None:
    with applied_settings({"search_mode": "dense"}):
        with pytest.raises(ScopeError, match="nest"):
            with applied_settings({"search_mode": "bm25"}):
                pass
        assert get_settings().search_mode == "dense"  # the outer scope is untouched


def test_the_bm25_model_is_never_a_variant() -> None:
    with pytest.raises(ScopeError, match="may not change"):
        with applied_settings({"sparse_model": "another/bm25"}):
            pass

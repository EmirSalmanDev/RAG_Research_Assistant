"""The relevance scoring function: pure, so every case runs without a database or API.
All papers and interests are invented."""

import itertools
import math
from datetime import UTC, datetime, timedelta

import pytest

from app.config import Settings
from app.features.relevance import (
    InterestVector,
    PaperFeatures,
    cosine,
    keyword_share,
    rank_key,
    score_paper,
    upvote_boost,
)

S = Settings(_env_file=None)
DAY = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def paper(**kw: object) -> PaperFeatures:
    base: dict[str, object] = dict(
        arxiv_id="2610.00001", title="An invented title", abstract="An invented abstract.",
        categories=["cs.CL"], hf_upvotes=None, published_at=DAY, vector=[1.0, 0.0],
    )  # fmt: skip
    return PaperFeatures(**(base | kw))  # type: ignore[arg-type]


def interest(i: int = 1, **kw: object) -> InterestVector:
    base: dict[str, object] = dict(
        id=i, weight=1.0, keywords=[], categories=[], embedding=[1.0, 0.0]
    )
    return InterestVector(**(base | kw))  # type: ignore[arg-type]


# --- building blocks ------------------------------------------------------------------------


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine([1, 1], [1, 0]) == pytest.approx(math.sqrt(0.5))
    assert cosine([2, 0], [5, 0]) == pytest.approx(1.0)  # length does not matter
    assert cosine([0, 0], [1, 0]) == 0.0  # a zero vector has no direction
    assert cosine([], [1.0]) == 0.0
    assert cosine([1, 0, 0], [1, 0]) == 0.0  # different sizes (a model change): no crash


@pytest.mark.parametrize(
    ("keywords", "text", "expected"),
    [
        ([], "anything", 0.0),
        (["", "  "], "anything", 0.0),  # blank keywords are not keywords
        (["rag"], "Retrieval storage", 0.0),  # whole words only: "rag" is not in "storage"
        (["rag"], "A RAG system", 1.0),  # case-insensitive
        (["rag"], "rag-based, (RAG).", 1.0),  # punctuation is a boundary
        (["rag"], "ragged", 0.0),
        (["rag"], "fine_rag", 0.0),  # an underscore is part of a word
        (["kv cache"], "the KV   cache grows", 1.0),  # a phrase; any whitespace between
        (["kv cache"], "kv caches", 0.0),
        (["c++"], "written in C++ and Rust", 1.0),  # regex characters are literal
        (["c++"], "written in C++11", 0.0),
        (["a.b"], "axb", 0.0),
        (["agent", "tool use", "planning"], "Agents use tool use and planning.", 2 / 3),
        (["agent", "agent", "AGENT"], "an agent", 1.0),  # repeats count once
        (["agent", "agent", "other"], "an agent", 0.5),  # ... also in the denominator
        (["agent", "tool"], "agent tool", 1.0),
    ],
)
def test_keyword_share(keywords: list[str], text: str, expected: float) -> None:
    assert keyword_share(keywords, text) == pytest.approx(expected)


def test_keyword_share_is_capped_at_one() -> None:
    assert keyword_share(["a", "b"], "a b") <= 1.0


def test_upvote_boost_grows_on_a_log_scale_and_is_capped() -> None:
    assert upvote_boost(None, S) == 0.0
    assert upvote_boost(0, S) == 0.0
    assert upvote_boost(-4, S) == 0.0  # nonsense input does not lower a score
    top = S.relevance_upvote_boost
    assert upvote_boost(S.relevance_upvote_cap, S) == pytest.approx(top)
    assert upvote_boost(100_000, S) == pytest.approx(top)  # the cap
    ladder = [upvote_boost(n, S) for n in (1, 3, 10, 30, 100)]
    assert ladder == sorted(ladder) and len(set(ladder)) == 5
    assert upvote_boost(10, S) == pytest.approx(top * math.log(11) / math.log(101))


# --- the score of a paper ---------------------------------------------------------------------


def test_with_no_interests_the_score_is_the_upvote_boost_only() -> None:
    assert score_paper(paper(), [], S).score == 0.0
    result = score_paper(paper(hf_upvotes=100), [], S)
    assert result.score == pytest.approx(S.relevance_upvote_boost)
    assert result.best_interest_id is None


def test_a_perfect_semantic_match_scores_the_cosine_weight() -> None:
    result = score_paper(paper(), [interest(7)], S)
    assert result.score == pytest.approx(S.relevance_cosine_weight)  # cos 1, no keywords
    assert result.best_interest_id == 7


def test_keywords_and_category_add_up() -> None:
    p = paper(title="Agents", abstract="Tool use for agents.", vector=[0.0, 1.0])
    i = interest(1, keywords=["agents", "tool use"], categories=["cs.CL"], embedding=[1.0, 0.0])
    # cos 0, both keywords found, category shared
    expected = S.relevance_keyword_weight * 1.0 + S.relevance_category_bonus
    assert score_paper(p, [i], S).score == pytest.approx(expected)


def test_a_category_mismatch_is_not_a_penalty() -> None:
    same = score_paper(paper(categories=["cs.CL"]), [interest(categories=["cs.CL"])], S).score
    other = score_paper(paper(categories=["cs.CV"]), [interest(categories=["cs.CL"])], S).score
    none = score_paper(paper(categories=["cs.CV"]), [interest(categories=[])], S).score
    assert same == pytest.approx(other + S.relevance_category_bonus)
    assert other == pytest.approx(none)  # an interest without categories accepts any paper


def test_weight_scales_the_semantic_part_but_not_the_category_bonus() -> None:
    base = S.relevance_cosine_weight
    light = score_paper(paper(), [interest(weight=0.5, categories=["cs.CL"])], S).score
    heavy = score_paper(paper(), [interest(weight=2.0, categories=["cs.CL"])], S).score
    assert light == pytest.approx(0.5 * base + S.relevance_category_bonus)
    assert heavy == pytest.approx(2.0 * base + S.relevance_category_bonus)


def test_upvotes_add_to_the_best_interest_value() -> None:
    plain = score_paper(paper(), [interest()], S).score
    liked = score_paper(paper(hf_upvotes=100), [interest()], S).score
    assert liked == pytest.approx(plain + S.relevance_upvote_boost)


def test_the_score_is_the_maximum_over_interests_not_the_sum() -> None:
    near = interest(1, embedding=[1.0, 0.0])
    far = interest(2, embedding=[0.0, 1.0])
    assert score_paper(paper(), [near, far], S).score == pytest.approx(
        score_paper(paper(), [near], S).score
    )


def test_a_missing_vector_counts_as_zero_similarity() -> None:
    p = paper(vector=None, abstract="About agents.")
    result = score_paper(p, [interest(3, keywords=["agents"])], S)
    assert result.score == pytest.approx(S.relevance_keyword_weight)
    assert result.best_interest_id == 3  # the keyword is evidence


def test_an_interest_without_an_embedding_still_counts_by_keywords() -> None:
    p = paper(abstract="About agents.")
    result = score_paper(p, [interest(3, keywords=["agents"], embedding=None)], S)
    assert result.score == pytest.approx(S.relevance_keyword_weight)


def test_no_evidence_means_no_best_interest() -> None:
    result = score_paper(paper(vector=None), [interest(1, categories=["cs.CV"])], S)
    assert result.score == 0.0
    assert result.best_interest_id is None
    # a shared category alone is evidence
    result = score_paper(paper(vector=None), [interest(1, categories=["cs.CL"])], S)
    assert result.best_interest_id == 1


def test_best_interest_ignores_weights() -> None:
    p = paper(vector=[1.0, 0.0])
    close = interest(1, weight=0.5, embedding=[1.0, 0.1])
    far = interest(2, weight=2.0, embedding=[0.2, 1.0])
    result = score_paper(p, [close, far], S)
    assert result.best_interest_id == 1  # the closer interest, although it is lighter
    heavy_value = 2.0 * S.relevance_cosine_weight * cosine(p.vector or [], far.embedding or [])
    light_value = 0.5 * S.relevance_cosine_weight * cosine(p.vector or [], close.embedding or [])
    assert result.score == pytest.approx(max(heavy_value, light_value))


def test_ties_between_interests_go_to_the_smaller_id_in_any_input_order() -> None:
    twins = [interest(5), interest(2), interest(9)]
    for order in itertools.permutations(twins):
        assert score_paper(paper(), list(order), S).best_interest_id == 2


def test_a_category_match_breaks_a_tie_in_the_unweighted_part() -> None:
    with_cat = interest(8, categories=["cs.CL"])
    without = interest(3)
    assert score_paper(paper(), [without, with_cat], S).best_interest_id == 8


def test_scoring_is_idempotent() -> None:
    p, interests = paper(hf_upvotes=12), [interest(1, keywords=["invented"]), interest(2)]
    assert score_paper(p, interests, S) == score_paper(p, interests, S)


def test_changed_settings_change_the_score_without_code_changes() -> None:
    other = Settings(_env_file=None, relevance_cosine_weight=0.5)
    assert score_paper(paper(), [interest()], other).score == pytest.approx(0.5)


# --- ordering -------------------------------------------------------------------------------------


def test_rank_order_is_score_then_newer_then_arxiv_id() -> None:
    rows = [
        ("2610.00004", 0.5, DAY),
        ("2610.00002", 0.5, DAY),  # ties with 00004 and is older than 00003 below
        ("2610.00003", 0.5, DAY + timedelta(hours=1)),
        ("2610.00001", 0.9, DAY - timedelta(days=3)),
        ("2610.00005", 0.1, DAY + timedelta(days=1)),
    ]
    expected = ["2610.00001", "2610.00003", "2610.00002", "2610.00004", "2610.00005"]
    for order in itertools.permutations(rows):  # the input order never matters
        ranked = sorted(order, key=lambda r: rank_key(*r))
        assert [r[0] for r in ranked] == expected

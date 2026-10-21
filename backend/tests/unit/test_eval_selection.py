"""eval/selection.py: the rule that picks "the best of the above", fixed in advance."""

from collections.abc import Sequence
from typing import Any

import pytest

from eval.selection import (
    HeldOutInSelection,
    RerankerResult,
    Rule,
    VariantResults,
    choose,
    paired_bootstrap,
    qualify_reranker,
)


def _results(name: str, recall: Sequence[float | None], faith: Sequence[float | None] | None = None,
             split: str = "generated", **extra: Any) -> VariantResults:  # fmt: skip
    faith = faith if faith is not None else [1.0] * len(recall)
    questions = {
        f"gen-{i:03d}": {"split": split, "context_recall": r, "faithfulness": f, **extra}
        for i, (r, f) in enumerate(zip(recall, faith, strict=True))
    }
    return VariantResults(name, None, questions)


N = 30


def test_the_bootstrap_is_repeatable_and_brackets_the_mean() -> None:
    diffs = [0.1, -0.05, 0.2, 0.0, 0.15] * 6
    a, b = paired_bootstrap(diffs, Rule()), paired_bootstrap(diffs, Rule())
    assert a == b
    assert a.low < a.mean < a.high and a.n == 30


def test_the_incumbent_stays_when_nothing_is_significantly_better() -> None:
    incumbent = _results("a3", [0.5, 0.6] * (N // 2))
    noisy = _results("a1", [0.6, 0.5] * (N // 2))  # same mean, other questions
    decision = choose(incumbent, [noisy])
    assert decision.winner == "a3"
    assert decision.comparisons[0].outcome == "not better"


def test_a_clearly_better_challenger_wins_and_the_best_of_several_is_taken() -> None:
    incumbent = _results("a3", [0.4] * N)
    good = _results("a2", [0.7] * N)
    better = _results("a4", [0.9] * N)
    decision = choose(incumbent, [good, better])
    assert decision.winner == "a4"
    assert [c.outcome for c in decision.comparisons] == ["better", "better"]


def test_a_challenger_worse_on_faithfulness_is_disqualified() -> None:
    incumbent = _results("a3", [0.4] * N, [1.0] * N)
    unfaithful = _results("a5", [0.9] * N, [0.5] * N)
    decision = choose(incumbent, [unfaithful])
    assert decision.winner == "a3"
    assert decision.comparisons[0].outcome == "disqualified"


def test_too_few_paired_questions_is_undecided() -> None:
    incumbent = _results("a3", [0.2] * N)
    # The challenger failed on most questions (a failed turn says nothing): 10 of 30 pairs.
    flaky = _results("a1", [1.0] * 10 + [0.0] * 20)
    for qid in list(flaky.questions)[10:]:
        flaky.questions[qid]["error"] = "search failed: X"  # type: ignore[index]
    decision = choose(incumbent, [flaky])
    assert decision.comparisons[0].outcome == "undecided" and decision.winner == "a3"


def test_held_out_questions_never_reach_the_rule() -> None:
    with pytest.raises(HeldOutInSelection):
        choose(_results("a3", [0.4] * N, split="heldout"), [_results("a1", [0.9] * N)])
    with pytest.raises(HeldOutInSelection):
        choose(_results("a3", [0.4] * N), [_results("a1", [0.9] * N, split="heldout")])


def _reranker(name: str, **kw: Any) -> RerankerResult:
    base: dict[str, Any] = {
        "variant": name, "candidates": 50, "median_rerank_s": 20.0, "peak_rss_mib": 1500.0,
        "isolated": True, "ranking_changes": False, "same_lists_as_reference": True,
        "answers": None,
    }  # fmt: skip
    return RerankerResult(**(base | kw))


def test_the_reranker_bar_speed_memory_and_quality() -> None:
    ref = _reranker("r-reference", answers=_results("r-reference", [0.6] * N))
    out = qualify_reranker(ref, [
        _reranker("r-threads-2", median_rerank_s=4.0),  # fast, same lists: qualifies
        _reranker("r-arena-off", median_rerank_s=3.0, same_lists_as_reference=False),
        _reranker("r-candidates-10", candidates=10, median_rerank_s=1.0, ranking_changes=True,
                  same_lists_as_reference=None, answers=_results("x", [0.6] * N)),
        _reranker("r-candidates-20", candidates=20, median_rerank_s=3.5, ranking_changes=True,
                  same_lists_as_reference=None, answers=_results("y", [0.1] * N)),
        _reranker("r-small-model", median_rerank_s=2.0, ranking_changes=True,
                  same_lists_as_reference=None, answers=_results("z", [0.6] * N),
                  isolated=False),
    ])  # fmt: skip
    by = {row["variant"]: row for row in out["variants"]}
    assert by["r-threads-2"]["qualifies"]
    assert "lists differ" in by["r-arena-off"]["reasons"][0]
    assert "10 candidates < 20" in by["r-candidates-10"]["reasons"][0]
    assert any("significantly worse" in r for r in by["r-candidates-20"]["reasons"])
    assert any("not run alone" in r for r in by["r-small-model"]["reasons"])
    assert out["recommended"] == "r-threads-2"


def test_without_a_qualifying_variant_the_reference_is_kept() -> None:
    out = qualify_reranker(
        _reranker("r-reference"), [_reranker("r-threads-4", median_rerank_s=9.0)]
    )
    assert out["recommended"] == "r-reference"

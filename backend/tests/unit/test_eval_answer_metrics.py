"""The answer metrics' arithmetic (eval/metrics.py), checked against hand-computed values."""

import math

import pytest

from eval.metrics import average_precision, cosine, response_relevancy, share


def test_share_is_the_fraction_of_true_and_undefined_when_empty() -> None:
    assert share([True, False, True, True]) == 0.75
    assert share([False]) == 0.0
    assert share([]) is None  # no statements: undefined, not a perfect or a zero score


@pytest.mark.parametrize(
    ("useful", "expected"),
    [
        ([True, False, True], (1 / 1 + 2 / 3) / 2),  # precision@1 = 1, precision@3 = 2/3
        ([False, True], (1 / 2) / 1),  # the one useful source is second
        ([True, True, False, False], 1.0),  # all useful sources first
        ([False, False, True, True], (1 / 3 + 2 / 4) / 2),
        ([False, False], 0.0),  # nothing useful
        ([], 0.0),
    ],
)
def test_average_precision_rewards_useful_sources_at_the_top(
    useful: list[bool], expected: float
) -> None:
    assert average_precision(useful) == pytest.approx(expected)


def test_average_precision_drops_when_the_same_useful_source_moves_down() -> None:
    assert average_precision([True, False, False]) > average_precision([False, False, True])


def test_cosine() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0  # a zero vector has no direction
    with pytest.raises(ValueError, match="different length"):
        cosine([1.0], [1.0, 0.0])


def test_response_relevancy_is_the_mean_cosine_unless_noncommittal() -> None:
    question = [1.0, 0.0]
    generated = [[1.0, 0.0], [0.0, 1.0]]  # one on topic (1.0), one off topic (0.0)
    assert response_relevancy(question, generated, noncommittal=False) == pytest.approx(0.5)
    assert response_relevancy(question, generated, noncommittal=True) == 0.0
    assert response_relevancy(question, [], noncommittal=False) is None
    assert response_relevancy(question, [], noncommittal=True) == 0.0  # evasive: still 0

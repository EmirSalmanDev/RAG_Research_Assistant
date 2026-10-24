"""Reference coverage (eval/metrics.py): a retrieval metric that does not depend on chunk ids,
so chunk sizes can be compared (ablation 7). All texts are invented."""

import pytest

from eval.metrics import reference_coverage

REFERENCE = (
    "Low-rank adaptation freezes the pretrained weights and trains two small matrices in "
    "every layer. The update is the product of the two matrices, scaled by a constant. At "
    "inference the product is merged into the weights, so there is no extra latency."
)
WORDS = REFERENCE.split()
FIRST_HALF, SECOND_HALF = " ".join(WORDS[:20]), " ".join(WORDS[20:])
OTHER = "Mixture of experts routes every token to two of sixteen feed-forward experts."


def test_the_whole_passage_retrieved_is_full_coverage() -> None:
    assert reference_coverage([OTHER, REFERENCE], [REFERENCE], 8) == 1.0


def test_a_passage_found_in_two_smaller_chunks_is_almost_fully_covered() -> None:
    """Only the n-grams that straddle the cut between the two pieces are lost."""
    covered = reference_coverage([FIRST_HALF, SECOND_HALF], [REFERENCE], 8)
    assert covered is not None and 0.8 < covered < 1.0
    half = reference_coverage([FIRST_HALF, OTHER], [REFERENCE], 8)
    assert half is not None and 0.3 < half < 0.6


def test_a_bigger_chunk_holding_the_passage_covers_it() -> None:
    assert reference_coverage([f"{OTHER} {REFERENCE} {OTHER}"], [REFERENCE], 8) == 1.0


def test_only_the_first_k_count_and_case_and_punctuation_do_not() -> None:
    assert reference_coverage([OTHER, REFERENCE.upper()], [REFERENCE], 1) == 0.0
    assert reference_coverage([OTHER, REFERENCE.upper().replace(",", "")], [REFERENCE], 2) == 1.0


def test_short_references_and_no_reference() -> None:
    assert reference_coverage(["they use a learned router"], ["learned router"], 8) == 1.0
    assert reference_coverage(["nothing here"], ["learned router"], 8) == 0.0
    assert reference_coverage([REFERENCE], [], 8) is None
    assert reference_coverage([], [REFERENCE], 8) == pytest.approx(0.0)

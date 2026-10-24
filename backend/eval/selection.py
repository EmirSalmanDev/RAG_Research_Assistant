"""Choosing "the best of the above" between ablation variants, by a rule fixed in advance.

The rule was written down (here and in docs/adr/0012) before any ablation had been run.
It looks at generated questions only: `choose` raises if it is handed a held-out (golden)
question, so the golden set can never decide a setting (CLAUDE.md rule 11).

The rule, for one step (the incumbent is what the app runs today):
1. Questions: the generated questions both runs answered without an error and without a
   stage falling back (a plan that failed and fell back to the default plan did not run
   the configuration under test), with the metric defined in both (paired: the same
   question under both configurations).
   If fewer than `min_paired_share` (80 %) of the generated questions where either run
   has the metric are paired, the comparison is "undecided" and the challenger loses.
2. Primary metric: context recall (does what reached the generator state the reference
   answer?). It is judged per question, does not depend on chunk ids (so chunk sizes can
   be compared), and is defined whenever a question has a reference answer and the turn
   did not fail, also when the generator refused (docs/adr/0012, decision 12).
3. A challenger is "better" only if the paired bootstrap interval of its mean difference
   to the incumbent lies above 0 (10,000 resamples of the paired questions, fixed seed,
   percentile interval). With m challengers in a step each interval is at 1 - 0.05/m
   (Bonferroni: 98.75 % for the four challengers of the retrieval step), so the chance
   that noise alone crowns any challenger stays at or below 5 %.
4. Guard: a better challenger is disqualified if it is significantly worse on
   faithfulness (the interval of that difference lies below 0).
5. Winner: the better, not disqualified challenger with the highest mean primary metric
   (ties: the order the step lists its variants); with none, the incumbent stays.
So with few questions most differences are "not significant" and the incumbent stays,
which is the honest result of a small test set. A percentile bootstrap on about 30
values is somewhat too narrow; the correction above leans against that, and the stored
intervals let a reader judge them.

The reranker comparison (moved from M4) asks another question, whether a cheaper runtime
setting is good enough, so it has its own rule (`qualify_reranker`): the owner's speed and
memory bar from M4 (at least 20 candidates, median rerank time per question at most 5 s,
peak memory at most 2,048 MiB), and quality: a variant whose ranking can differ (fewer
candidates, another model) must not be significantly worse than the reference on context
recall; a runtime-only variant (threads, arena) must return exactly the reference's lists.
The rule recommends; changing the app's defaults stays the owner's decision.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

PerQuestion = Mapping[str, Mapping[str, Any]]  # question id -> that question's stored metrics


class HeldOutInSelection(ValueError):
    """A held-out (golden) question reached the selection: never allowed."""


@dataclass(frozen=True)
class Rule:
    primary: str = "context_recall"
    guard: str = "faithfulness"
    confidence: float = 0.95
    resamples: int = 10_000
    seed: int = 20261021
    min_paired_share: float = 0.8


RULE = Rule()  # the rule as fixed in advance


@dataclass(frozen=True)
class Interval:
    n: int
    mean: float
    low: float
    high: float


def paired_bootstrap(
    diffs: Sequence[float], rule: Rule, confidence: float | None = None
) -> Interval:
    """Mean of the paired differences and its percentile bootstrap interval (at the
    rule's confidence unless another is given)."""
    n = len(diffs)
    if n == 0:
        raise ValueError("no paired questions")
    rng = random.Random(rule.seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(rule.resamples))
    tail = (1 - (rule.confidence if confidence is None else confidence)) / 2
    low = means[int(tail * rule.resamples)]
    high = means[min(rule.resamples - 1, int((1 - tail) * rule.resamples))]
    return Interval(n, sum(diffs) / n, low, high)


@dataclass(frozen=True)
class VariantResults:
    """One variant's answer run: per generated question, its stored metrics."""

    variant: str
    run_id: int | None
    questions: PerQuestion  # question id -> metrics (as stored in eval_results.metrics)


def _check_generated(results: VariantResults) -> None:
    held_out = sorted(q for q, m in results.questions.items() if m.get("split") != "generated")
    if held_out:
        raise HeldOutInSelection(
            f"{results.variant}: selection may only look at generated questions, got {held_out}"
        )


def _value(metrics: Mapping[str, Any], metric: str) -> float | None:
    if metrics.get("error") or metrics.get("stage_errors"):
        return None  # a failed turn, or one that fell back, did not run the configuration
    value = metrics.get(metric)
    return float(value) if isinstance(value, int | float) else None


def compare(
    incumbent: VariantResults, challenger: VariantResults, metric: str, rule: Rule,
    confidence: float | None = None,
) -> tuple[Interval | None, int, int]:  # fmt: skip
    """(interval of challenger - incumbent, paired questions, questions where either has
    the metric). The interval is None when there is no pair."""
    pairs: list[float] = []
    either = 0
    for qid in sorted(set(incumbent.questions) | set(challenger.questions)):
        a = _value(incumbent.questions.get(qid, {}), metric)
        b = _value(challenger.questions.get(qid, {}), metric)
        if a is None and b is None:
            continue
        either += 1
        if a is not None and b is not None:
            pairs.append(b - a)
    return (paired_bootstrap(pairs, rule, confidence) if pairs else None), len(pairs), either


@dataclass
class Comparison:
    challenger: str
    paired: int
    eligible: int
    primary: Interval | None
    guard: Interval | None
    outcome: str  # better | not better | undecided | disqualified


@dataclass
class Decision:
    winner: str
    incumbent: str
    reason: str
    rule: Rule
    comparisons: list[Comparison] = field(default_factory=list)
    primary_confidence: float | None = None  # after the Bonferroni correction

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "incumbent": self.incumbent,
            "reason": self.reason,
            "rule": asdict(self.rule),
            "primary_confidence": self.primary_confidence,
            "comparisons": [asdict(c) for c in self.comparisons],
        }


def choose(
    incumbent: VariantResults, challengers: Sequence[VariantResults], rule: Rule = RULE
) -> Decision:
    """The step's winner by the rule above (challengers in the step's order)."""
    for results in (incumbent, *challengers):
        _check_generated(results)
    if not incumbent.questions:
        raise ValueError(f"{incumbent.variant}: no generated questions to compare on")
    comparisons: list[Comparison] = []
    better: list[tuple[float, int, str]] = []
    # Bonferroni: m challengers share the error rate of one comparison.
    corrected = 1 - (1 - rule.confidence) / max(1, len(challengers))
    for order, challenger in enumerate(challengers):
        primary, paired, eligible = compare(incumbent, challenger, rule.primary, rule, corrected)
        guard, _, _ = compare(incumbent, challenger, rule.guard, rule)
        if primary is None or eligible == 0 or paired / eligible < rule.min_paired_share:
            outcome = "undecided"
        elif primary.low <= 0:
            outcome = "not better"
        elif guard is not None and guard.high < 0:
            outcome = "disqualified"
        else:
            outcome = "better"
            better.append((-primary.mean, order, challenger.variant))
        comparisons.append(
            Comparison(challenger.variant, paired, eligible, primary, guard, outcome)
        )
    if not better:
        return Decision(
            incumbent.variant, incumbent.variant,
            "no challenger is significantly better on the primary metric", rule, comparisons,
            corrected,
        )  # fmt: skip
    winner = min(better)[2]
    return Decision(
        winner, incumbent.variant,
        f"significantly better on {rule.primary} and not worse on {rule.guard}", rule,
        comparisons, corrected,
    )  # fmt: skip


# --- the reranker comparison --------------------------------------------------------------------


@dataclass(frozen=True)
class RerankerBar:
    """The owner's bar for the reranker settings (M4 Step C, fixed 2026-10-04)."""

    min_candidates: int = 20
    max_median_rerank_s: float = 5.0
    max_peak_rss_mib: float = 2048.0


BAR = RerankerBar()


@dataclass(frozen=True)
class RerankerResult:
    variant: str
    candidates: int
    median_rerank_s: float | None  # per question, over the run (warm: the model is loaded first)
    peak_rss_mib: float | None
    isolated: bool  # the variant ran alone in its process (else peak memory is not its own)
    ranking_changes: bool  # fewer candidates or another model
    same_lists_as_reference: bool | None  # runtime-only variants: retrieval lists identical
    answers: VariantResults | None  # ranking-changing variants: their answer run


def qualify_reranker(
    reference: RerankerResult,
    variants: Sequence[RerankerResult],
    bar: RerankerBar = BAR,
    rule: Rule = RULE,
) -> dict[str, Any]:
    """Per variant: does it meet the bar, and is its quality kept? Recommends the fastest
    variant that does both (none: keep the reference)."""
    rows: list[dict[str, Any]] = []
    for v in variants:
        reasons: list[str] = []
        if v.candidates < bar.min_candidates:
            reasons.append(f"{v.candidates} candidates < {bar.min_candidates}")
        if v.median_rerank_s is None or v.median_rerank_s > bar.max_median_rerank_s:
            reasons.append(f"median rerank {v.median_rerank_s} s > {bar.max_median_rerank_s} s")
        if not v.isolated:
            reasons.append("not run alone in its process: peak memory not its own")
        elif v.peak_rss_mib is None or v.peak_rss_mib > bar.max_peak_rss_mib:
            reasons.append(f"peak memory {v.peak_rss_mib} MiB > {bar.max_peak_rss_mib} MiB")
        if v.ranking_changes:
            if v.answers is None or reference.answers is None:
                reasons.append("no answer runs to compare quality")
            else:
                for results in (reference.answers, v.answers):
                    _check_generated(results)
                interval, _, _ = compare(reference.answers, v.answers, rule.primary, rule)
                if interval is None:
                    reasons.append("no paired questions")
                elif interval.high < 0:
                    reasons.append(f"significantly worse on {rule.primary}")
        elif v.same_lists_as_reference is not True:
            reasons.append("retrieval lists differ from the reference")
        rows.append(
            {
                "variant": v.variant,
                "qualifies": not reasons,
                "reasons": reasons,
                "median_rerank_s": v.median_rerank_s,
                "peak_rss_mib": v.peak_rss_mib,
            }
        )
    qualifying = [r for r in rows if r["qualifies"]]
    best = min(qualifying, key=lambda r: r["median_rerank_s"]) if qualifying else None
    return {
        "reference": reference.variant,
        "recommended": best["variant"] if best else reference.variant,
        "bar": asdict(bar),
        "rule": asdict(rule),
        "variants": rows,
    }

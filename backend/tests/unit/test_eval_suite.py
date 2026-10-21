"""eval/suite.py: the smoke set, the report CI posts, and paid runs needing a cap."""

from typing import Any

import pytest

from eval import suite as suite_module
from eval.golden import GoldenItem, SetQuestion
from eval.suite import SmokeSetError, _without_texts, markdown, smoke_set


def _q(qid: str, qtype: Any, split: Any = "generated") -> SetQuestion:
    refs = [] if qtype == "unanswerable" else ["0214d8ce-b010-59e2-b4d6-06b4e659c8d5"]
    return SetQuestion(GoldenItem(id=qid, question=f"Invented {qid}?", type=qtype,
                                  reference_chunk_ids=refs), split)  # fmt: skip


GOLDEN = [_q(f"q{i:02d}", "fact", "heldout") for i in range(1, 21)]
GENERATED = (
    [_q(f"gen-{i:03d}", "fact") for i in range(1, 21)]
    + [_q(f"gen-{i:03d}", "compare") for i in range(21, 29)]
    + [_q(f"gen-{i:03d}", "filter") for i in range(29, 33)]
    + [_q(f"gen-{i:03d}", "unanswerable") for i in range(33, 41)]
)


def test_the_smoke_set_is_fifteen_generated_questions_of_every_type() -> None:
    chosen = smoke_set([*GOLDEN, *GENERATED])
    assert len(chosen) == 15
    assert {q.split for q in chosen} == {"generated"}  # never a held-out question
    types = [q.item.type for q in chosen]
    assert {"fact", "compare", "filter", "unanswerable"} <= set(types)
    assert types.count("filter") == 4  # round robin: a small type is taken whole first
    assert smoke_set(list(reversed(GENERATED))) == chosen  # the file order does not matter


def test_held_out_questions_never_change_the_smoke_set() -> None:
    chosen = {q.item.id for q in smoke_set(GENERATED)}
    again = {q.item.id for q in smoke_set([*GENERATED, *GOLDEN])}
    assert chosen == again


def test_too_few_generated_questions_is_a_clear_error() -> None:
    with pytest.raises(SmokeSetError, match="needs 15 generated questions, there are 14"):
        smoke_set([*GOLDEN, *GENERATED[:14]])


def test_the_report_tables_and_the_export_without_chunk_texts() -> None:
    run = {
        "id": 31, "name": "smoke:configured:answers", "git_sha": "0123456789abcdef",
        "config": {"kind": "answers"},
        "metrics": {"status": "complete", "spend_usd": "0.061",
                    "all": {"n_questions": 15, "faithfulness": 0.9, "latency_ms_p50": 4210.0},
                    "generated": {"n_questions": 15, "faithfulness": 0.9}},
    }  # fmt: skip
    text = markdown([run])
    assert "**smoke:configured:answers** (eval_runs 31, commit `0123456789ab`)" in text
    assert "| all | 15 | 0.900 |" in text and "4210.0" in text
    assert "| heldout |" not in text
    stripped = _without_texts(
        {"hits": [{"chunk_id": "x", "text": "paper text"}], "sources": [{"n": 1, "text": "t"}],
         "faithfulness": 1.0}
    )  # fmt: skip
    assert stripped == {"hits": [{"chunk_id": "x"}], "sources": [{"n": 1}], "faithfulness": 1.0}


def test_a_paid_suite_run_needs_a_cap(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(suite_module, "load_question_set", lambda: [*GOLDEN, *GENERATED])
    assert suite_module.main(["smoke"]) == 2
    assert "needs --max-usd" in capsys.readouterr().err
    monkeypatch.setattr(suite_module, "load_question_set", lambda: GOLDEN)
    assert suite_module.main(["smoke", "--max-usd", "0.3"]) == 2
    assert "needs 15 generated questions" in capsys.readouterr().err

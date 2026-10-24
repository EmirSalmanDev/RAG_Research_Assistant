"""Loading and checking golden.jsonl (invented questions in temporary files)."""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from eval.golden import GoldenError, for_tuning, load_golden, load_question_set

C1 = "0214d8ce-b010-59e2-b4d6-06b4e659c8d5"
C2 = "1b2f8d4e-6a3c-5e7f-9a1b-2c3d4e5f6a7b"


def _write(tmp_path: Path, *items: dict[str, Any], raw: str = "") -> Path:
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in items) + raw)
    return path


def _q(n: int, **kw: Any) -> dict[str, Any]:
    return {"id": f"q{n:02d}", "question": f"Invented question {n}?", **kw}


def test_loads_items_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    lines = [
        json.dumps(_q(1, reference_chunk_ids=[C1, C2], reference_answer="An answer.", type="fact")),
        "",
        json.dumps(_q(2, type="unanswerable", notes="not in the corpus")),
    ]
    path.write_text("\n".join(lines) + "\n")
    first, second = load_golden(path)
    assert first.reference_chunk_ids == [uuid.UUID(C1), uuid.UUID(C2)]
    assert (first.scored, second.scored) == (True, False)
    assert (second.type, second.notes, second.reference_answer) == (
        "unanswerable",
        "not in the corpus",
        None,
    )


def test_untyped_question_without_references_is_unscored(tmp_path: Path) -> None:
    (item,) = load_golden(_write(tmp_path, _q(1)))
    assert not item.scored


def test_find_papers_questions_are_accepted_and_scored(tmp_path: Path) -> None:
    (item,) = load_golden(_write(tmp_path, _q(1, type="find_papers", reference_chunk_ids=[C1])))
    assert (item.type, item.scored) == ("find_papers", True)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (_q(1, reference_chunk_ids=["not-a-uuid"]), "line 1"),
        (_q(1, reference_chunk_ids=[C1, C1]), "same id twice"),
        (_q(1, type="fact"), "needs reference_chunk_ids"),
        (_q(1, type="find_papers"), "needs reference_chunk_ids"),
        (_q(1, type="comparison", reference_chunk_ids=[C1]), "line 1"),  # the type is "compare"
        (_q(1, type="unanswerable", reference_chunk_ids=[C1]), "cannot have reference chunks"),
        (_q(1, type="opinion", reference_chunk_ids=[C1]), "line 1"),
        (_q(1, extra_field="typo"), "line 1"),
        ({"id": "q01", "question": "   "}, "question is blank"),
    ],
)
def test_bad_items_are_rejected_with_their_line(
    tmp_path: Path, item: dict[str, Any], message: str
) -> None:
    with pytest.raises(GoldenError, match=message):
        load_golden(_write(tmp_path, item))


def test_invalid_json_names_the_line(tmp_path: Path) -> None:
    with pytest.raises(GoldenError, match="line 2"):
        load_golden(_write(tmp_path, _q(1), raw="\n{not json"))


@pytest.mark.parametrize("field", ["id", "question"])
def test_repeated_ids_or_questions_are_rejected(tmp_path: Path, field: str) -> None:
    first, second = _q(1), _q(2)
    second[field] = first[field]
    with pytest.raises(GoldenError, match=f"repeated {field}"):
        load_golden(_write(tmp_path, first, second))


def test_missing_or_empty_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(GoldenError, match="does not exist"):
        load_golden(tmp_path / "golden.jsonl")
    with pytest.raises(GoldenError, match="no questions"):
        load_golden(_write(tmp_path, raw="\n\n"))


# --- the full set: held-out golden questions + reviewed generated ones ------------------------


def _file(path: Path, *items: dict[str, Any]) -> Path:
    path.write_text("\n".join(json.dumps(i) for i in items) + "\n")
    return path


def test_the_question_set_tags_each_question_with_its_file(tmp_path: Path) -> None:
    golden = _file(tmp_path / "golden.jsonl", _q(1, reference_chunk_ids=[C1]), _q(2))
    generated = _file(
        tmp_path / "generated.jsonl",
        {"id": "gen-001", "question": "A generated question?", "reference_chunk_ids": [C2],
         "type": "fact"},
    )  # fmt: skip
    questions = load_question_set(golden, generated)
    assert [(q.item.id, q.split) for q in questions] == [
        ("q01", "heldout"), ("q02", "heldout"), ("gen-001", "generated"),
    ]  # fmt: skip


def test_without_a_generated_file_only_the_golden_questions_load(tmp_path: Path) -> None:
    golden = _file(tmp_path / "golden.jsonl", _q(1, reference_chunk_ids=[C1]))
    for generated in (tmp_path / "missing.jsonl", _file(tmp_path / "empty.jsonl")):
        assert [q.split for q in load_question_set(golden, generated)] == ["heldout"]


@pytest.mark.parametrize("field", ["id", "question"])
def test_a_generated_question_can_never_repeat_a_golden_one(tmp_path: Path, field: str) -> None:
    golden = _file(tmp_path / "golden.jsonl", _q(1, reference_chunk_ids=[C1]))
    clash = _q(1, reference_chunk_ids=[C2]) | (
        {"question": "Something else?"} if field == "id" else {"id": "gen-001"}
    )
    generated = _file(tmp_path / "generated.jsonl", clash)
    with pytest.raises(GoldenError, match=f"repeated {field}"):
        load_question_set(golden, generated)


def test_only_generated_questions_may_be_used_for_tuning(tmp_path: Path) -> None:
    golden = _file(tmp_path / "golden.jsonl", _q(1, reference_chunk_ids=[C1]), _q(2))
    generated = _file(
        tmp_path / "generated.jsonl",
        {"id": "gen-001", "question": "A generated question?", "type": "unanswerable"},
    )
    tuning = for_tuning(load_question_set(golden, generated))
    assert [q.item.id for q in tuning] == ["gen-001"]
    assert for_tuning(load_question_set(golden, tmp_path / "missing.jsonl")) == []

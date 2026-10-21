"""The golden set: questions with the chunks that answer them.

`backend/eval/golden.jsonl` holds one JSON object per line. Only the owner changes it;
the code only reads it. The 20 questions are AI-drafted, verified against stored chunk
text; they are a held-out check: nothing is ever tuned to make them score better.

    {"id": "q01", "question": "How does LoRA reduce the trainable parameters?",
     "reference_answer": "It freezes the weights and learns two low-rank matrices ...",
     "reference_chunk_ids": ["0214d8ce-b010-59e2-b4d6-06b4e659c8d5"],
     "type": "fact", "notes": "optional free text"}

- `reference_chunk_ids`: ids from the `chunks` table (= Qdrant point ids). An empty list
  marks an unanswerable question: it is run and stored, but not scored, because recall
  and nDCG need at least one right answer.
- `type` (optional): fact, compare, filter, find_papers (the question asks which papers
  cover something) or unanswerable. `unanswerable` and an empty list must go together.
- `reference_answer`, `notes`: optional; for the answer metrics of a later milestone.

Chunk ids contain the paper version, so a new arXiv version (or re-chunking) changes
them; the runner refuses to run when a reference id no longer exists.

The full question set has two parts (docs/adr/0012), kept in two files so they can never
be mixed up:
- `golden.jsonl` (split "heldout"): the 20 first questions. Only the owner edits them, and
  nothing is ever tuned on them: they are the held-out check.
- `generated.jsonl` (split "generated"): questions written by the test-set generator and
  accepted one by one by the owner in the review tool (`eval/make_testset.py`). Same
  format. Only these may be used to choose between settings (`for_tuning`).
"""

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"
GENERATED_PATH = Path(__file__).parent / "generated.jsonl"

QuestionType = Literal["fact", "compare", "filter", "find_papers", "unanswerable"]
Split = Literal["heldout", "generated"]


class GoldenError(ValueError):
    """golden.jsonl is missing or malformed; the message says where."""


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    reference_chunk_ids: list[uuid.UUID] = Field(default_factory=list)
    type: QuestionType | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "GoldenItem":
        if not self.question.strip():
            raise ValueError("question is blank")
        if len(set(self.reference_chunk_ids)) != len(self.reference_chunk_ids):
            raise ValueError("reference_chunk_ids lists the same id twice")
        if self.type == "unanswerable" and self.reference_chunk_ids:
            raise ValueError("an unanswerable question cannot have reference chunks")
        if self.type not in (None, "unanswerable") and not self.reference_chunk_ids:
            raise ValueError(f"a {self.type!r} question needs reference_chunk_ids")
        return self

    @property
    def scored(self) -> bool:
        """Retrieval metrics need at least one reference chunk."""
        return bool(self.reference_chunk_ids)


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    """Read and check the golden set. Errors name the line, so they are easy to fix."""
    if not path.is_file():
        raise GoldenError(f"{path} does not exist (the owner writes the golden set)")
    items: list[GoldenItem] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue  # blank lines are allowed, e.g. between groups of questions
        try:
            items.append(GoldenItem.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as err:
            raise GoldenError(f"{path.name} line {number}: {err}") from err
    if not items:
        raise GoldenError(f"{path} has no questions")
    _no_repeats(items, path.name)
    return items


def _no_repeats(items: Sequence[GoldenItem], where: str) -> None:
    for field in ("id", "question"):
        values = [getattr(i, field) for i in items]
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise GoldenError(f"{where}: repeated {field}: {repeated}")


@dataclass(frozen=True)
class SetQuestion:
    """One question of the full set and the file it came from."""

    item: GoldenItem
    split: Split


def load_question_set(
    golden: Path = GOLDEN_PATH, generated: Path = GENERATED_PATH
) -> list[SetQuestion]:
    """The held-out golden questions, then the accepted generated ones.

    The generated file is optional (it exists once the owner has reviewed questions).
    Ids and question texts must be unique across both files, so a generated question can
    never stand in for a golden one.
    """
    questions = [SetQuestion(item, "heldout") for item in load_golden(golden)]
    if generated.is_file() and generated.read_text().strip():
        questions += [SetQuestion(item, "generated") for item in load_golden(generated)]
    _no_repeats([q.item for q in questions], f"{golden.name} + {generated.name}")
    return questions


def for_tuning(questions: Sequence[SetQuestion]) -> list[SetQuestion]:
    """The questions a choice between settings may look at: the generated ones only.

    Every comparison that picks a setting (a reranker, a candidate count, a chunk size)
    must go through this, so the held-out golden questions never influence a choice;
    they are only reported, next to the generated ones (CLAUDE.md rule 11).
    """
    return [q for q in questions if q.split == "generated"]

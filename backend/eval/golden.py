"""The golden set: hand-written questions with the chunks that answer them.

`backend/eval/golden.jsonl` holds one JSON object per line. Only the owner writes it;
the code only reads it. The 20 hand-written questions are a held-out check: nothing is
ever tuned to make them score better.

    {"id": "q01", "question": "How does LoRA reduce the trainable parameters?",
     "reference_answer": "It freezes the weights and learns two low-rank matrices ...",
     "reference_chunk_ids": ["0214d8ce-b010-59e2-b4d6-06b4e659c8d5"],
     "type": "fact", "notes": "optional free text"}

- `reference_chunk_ids`: ids from the `chunks` table (= Qdrant point ids). An empty list
  marks an unanswerable question: it is run and stored, but not scored, because recall
  and nDCG need at least one right answer.
- `type` (optional): fact, compare, filter or unanswerable. `unanswerable` and an empty
  list must go together.
- `reference_answer`, `notes`: optional; for the answer metrics of a later milestone.

Chunk ids contain the paper version, so a new arXiv version (or re-chunking) changes
them; the runner refuses to run when a reference id no longer exists.
"""

import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"

QuestionType = Literal["fact", "compare", "filter", "unanswerable"]


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
    for field in ("id", "question"):
        values = [getattr(i, field) for i in items]
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            raise GoldenError(f"{path.name}: repeated {field}: {repeated}")
    return items

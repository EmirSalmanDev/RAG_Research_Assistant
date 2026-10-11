"""messages.meta: validated on write, read back by version."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.rag.meta import MessageMeta, PlanMeta, load_meta
from app.rag.plan import QueryPlan
from app.rag.search import SearchFilters
from app.rag.verify import VerifyResult

PLAN = QueryPlan(
    intent="explain",
    rephrasings=("a", "b"),
    filters=SearchFilters(categories=("cs.CL",), published_from=datetime(2026, 9, 1, tzinfo=UTC)),
    dropped=("category 'cs.CV': not in the index",),
)


def _meta() -> MessageMeta:
    return MessageMeta(
        intent="explain",
        standalone_question="Explain LoRA",
        plan=PlanMeta.from_plan(PLAN),
        sources=8,
        abstract_only=True,
        prompts={"plan": "plan_v1", "generate": "generate_explain_v1"},
        skipped={"condense": "no history", "rerank": "disabled"},
        errors={"verify": "RetryableError"},
        verification=VerifyResult(
            status="unavailable",
            sentences=[],
            invalid_citations=[],
            uncited_sentences=[],
            unsupported_sentences=[],
        ),
    )


def test_meta_round_trips_through_json() -> None:
    data = _meta().model_dump(mode="json")
    assert data["version"] == 1
    assert data["plan"]["published_from"] == "2026-09-01T00:00:00+00:00"
    assert load_meta(data) == _meta()


def test_unknown_fields_and_stages_are_refused() -> None:
    data = _meta().model_dump(mode="json")
    with pytest.raises(ValidationError):
        MessageMeta.model_validate(data | {"surprise": 1})
    with pytest.raises(ValidationError):
        MessageMeta.model_validate(data | {"skipped": {"retrieve": "x"}})


def test_load_meta_reads_none_and_refuses_unknown_versions() -> None:
    assert load_meta(None) is None
    with pytest.raises(ValueError, match="unknown messages.meta version 2"):
        load_meta(_meta().model_dump(mode="json") | {"version": 2})

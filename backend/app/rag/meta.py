"""The shape of `messages.meta` (assistant messages; migration 0003).

The column is jsonb, so Postgres accepts any JSON: the type checking happens here.
Every write goes through `MessageMeta` (unknown fields are refused), and every read goes
through `load_meta`, which picks the model by `version`. When the shape changes in a way
old rows do not fit, add `MessageMetaV2` with `version: Literal[2]` and teach `load_meta`
both; rows written as version 1 then stay readable.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from app.rag.plan import Intent, QueryPlan
from app.rag.verify import VerifyResult

Stage = Literal["condense", "plan", "search", "rerank", "generate", "verify"]
STAGES: tuple[Stage, ...] = ("condense", "plan", "search", "rerank", "generate", "verify")
Refusal = Literal["no_sources", "insufficient_sources"]


class PlanMeta(BaseModel):
    """The checked plan that searched, as plain JSON (dates as ISO strings)."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    rephrasings: list[str]
    arxiv_ids: list[str]
    categories: list[str]
    published_from: str | None
    published_to: str | None
    dropped: list[str]  # what the checks removed from the model's plan

    @classmethod
    def from_plan(cls, plan: QueryPlan) -> "PlanMeta":
        f = plan.filters
        return cls(
            intent=plan.intent,
            rephrasings=list(plan.rephrasings),
            arxiv_ids=list(f.arxiv_ids),
            categories=list(f.categories),
            published_from=f.published_from.isoformat() if f.published_from else None,
            published_to=f.published_to.isoformat() if f.published_to else None,
            dropped=list(plan.dropped),
        )


class MessageMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    intent: Intent
    standalone_question: str
    plan: PlanMeta
    sources: int  # how many numbered sources generate was given (0 on a refusal)
    refusal: Refusal | None = None
    abstract_only: bool = False  # every cited source is an abstract chunk
    prompts: dict[Stage, str] = {}  # stage -> prompt id used, e.g. "generate_v2"
    skipped: dict[Stage, str] = {}  # stage -> why it did not run
    errors: dict[Stage, str] = {}  # stage -> the error it fell back from
    verification: VerifyResult | None = None


def load_meta(data: dict[str, Any] | None) -> MessageMeta | None:
    """Read a stored meta (None stays None). Raises if the row does not fit its version."""
    if data is None:
        return None
    version = data.get("version")
    if version == 1:
        return MessageMeta.model_validate(data)
    raise ValueError(f"unknown messages.meta version {version!r}")

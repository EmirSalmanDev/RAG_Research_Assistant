"""Plan stage: one Structured Outputs call that turns the question into a search plan.

The model returns (in the exact JSON shape of `PlanOutput`):
- intent: qa | explain | compare | find_papers. It picks the generate prompt and is
  stored on the message. For now compare and find_papers use the qa prompt.
- two rephrasings, searched as well as the question (more ways to hit the right chunk);
- filters: arXiv ids, arXiv categories and a published_at date range.

The model's filters are never trusted as they are, because a wrong filter silently
hides every right answer. `check_plan` keeps an arXiv id only if it is written in the
question, a category only if it is one the index holds, and a date only if it parses
and is not in the future. What it removes is recorded (the trace and the message meta
show it). There is no author filter: chunk payloads have no author field.

With `plan_enabled` off, or if the call fails, the default plan is used: intent qa, no
filters, no rephrasings (docs/adr/0006).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Literal, get_args

from pydantic import BaseModel

from app.llm.client import ChatClient, ChatMessage, ChatUsage
from app.rag.prompts import escape_block, load_prompt
from app.rag.search import SearchFilters

PLAN_PROMPT = "plan_v1"
FEATURE = "chat-plan"
MAX_OUTPUT_TOKENS = 400
MAX_REPHRASINGS = 2

Intent = Literal["qa", "explain", "compare", "find_papers"]
INTENTS: tuple[Intent, ...] = get_args(Intent)

# New-style ids (2017 on: 1706.03762, optional version) and old-style ids (cs/0112017).
_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5}|[a-z][a-z-]*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b")


class PlanOutput(BaseModel):
    """The JSON shape the model must fill. Every field is required (Structured Outputs'
    strict mode); 'nothing' is an empty list or null."""

    intent: Intent
    rephrasings: list[str]
    arxiv_ids: list[str]
    categories: list[str]
    published_from: str | None  # YYYY-MM-DD
    published_to: str | None  # YYYY-MM-DD


@dataclass(frozen=True)
class QueryPlan:
    intent: Intent = "qa"
    rephrasings: tuple[str, ...] = ()
    filters: SearchFilters = field(default_factory=SearchFilters)
    dropped: tuple[str, ...] = ()  # what check_plan removed from the model's answer


DEFAULT_PLAN = QueryPlan()


@dataclass(frozen=True)
class PlanResult:
    plan: QueryPlan
    skipped: str | None = None
    usage: ChatUsage | None = None
    prompt: str | None = None


def build_messages(question: str, *, today: date, categories: Sequence[str]) -> list[ChatMessage]:
    user = (
        f"Today's date: {today.isoformat()}\n"
        f"arXiv categories in the index: {', '.join(categories)}\n\n"
        f"<question>\n{escape_block(question)}\n</question>"
    )
    return [
        {"role": "system", "content": load_prompt(PLAN_PROMPT).text},
        {"role": "user", "content": user},
    ]


def _arxiv_ids_in(text: str) -> set[str]:
    return {m[1] for m in _ARXIV_ID.finditer(text)}


def _parse_day(value: str | None, name: str, today: date, dropped: list[str]) -> date | None:
    if value is None or not value.strip():
        return None
    try:
        day = date.fromisoformat(value.strip())
    except ValueError:
        dropped.append(f"{name} {value!r}: not a YYYY-MM-DD date")
        return None
    if day > today:
        dropped.append(f"{name} {value!r}: in the future")
        return None
    return day


def check_plan(
    out: PlanOutput,
    question: str,
    *,
    categories: Sequence[str],
    today: date,
    use_rephrasings: bool,
) -> QueryPlan:
    """The model's plan, minus everything that cannot be right (see the module doc)."""
    dropped: list[str] = []

    in_question = _arxiv_ids_in(question)
    ids: list[str] = []
    for raw in out.arxiv_ids:
        found = _arxiv_ids_in(raw.strip())
        arxiv_id = next(iter(found)) if len(found) == 1 else None  # drops a version suffix
        if arxiv_id is not None and arxiv_id in in_question:
            ids.append(arxiv_id)
        else:
            dropped.append(f"arxiv_id {raw!r}: not written in the question")

    known = {c.lower(): c for c in categories}
    cats: list[str] = []
    for raw in out.categories:
        category = known.get(raw.strip().lower())
        if category is not None:
            cats.append(category)
        else:
            dropped.append(f"category {raw!r}: not in the index")

    start = _parse_day(out.published_from, "published_from", today, dropped)
    end = _parse_day(out.published_to, "published_to", today, dropped)
    if start is not None and end is not None and start > end:
        dropped.append(f"date range {start}..{end}: start after end")
        start = end = None

    rephrasings: list[str] = []
    if use_rephrasings:
        seen = {question.strip().lower()}
        for r in out.rephrasings:
            text = r.strip()
            if text and text.lower() not in seen and len(rephrasings) < MAX_REPHRASINGS:
                seen.add(text.lower())
                rephrasings.append(text)

    filters = SearchFilters(
        arxiv_ids=tuple(dict.fromkeys(ids)),
        categories=tuple(dict.fromkeys(cats)),
        # Whole days in UTC, both ends inclusive (as GET /search does).
        published_from=datetime.combine(start, time.min, UTC) if start else None,
        published_to=datetime.combine(end, time.max, UTC) if end else None,
    )
    return QueryPlan(out.intent, tuple(rephrasings), filters, tuple(dropped))


def plan_query(
    question: str,
    *,
    client: ChatClient,
    enabled: bool,
    use_rephrasings: bool,
    categories: Sequence[str],
    today: date,
) -> PlanResult:
    """The checked search plan for `question` (the default plan when disabled)."""
    if not enabled:
        return PlanResult(DEFAULT_PLAN, skipped="disabled")
    out, usage = client.parse(
        build_messages(question, today=today, categories=categories),
        PlanOutput,
        feature=FEATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    plan = check_plan(
        out, question, categories=categories, today=today, use_rephrasings=use_rephrasings
    )
    return PlanResult(plan, usage=usage, prompt=PLAN_PROMPT)

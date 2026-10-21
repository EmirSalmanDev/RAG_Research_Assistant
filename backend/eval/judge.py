"""The judge: structured model calls that turn an answer into yes/no verdicts.

Each answer metric needs the model's judgement on many small items (a statement, a
source, a citation). One Structured Outputs call per metric and answer returns one
verdict per item; `eval/metrics.py` turns the verdicts into a score. The steps follow the
RAGAS metrics of the same names (docs/adr/0012):

- faithfulness: `statements` breaks the answer into standalone claims, `faithfulness`
  checks each claim against the sources the answer was given;
- response relevancy: `relevancy` writes questions that the answer would answer, and says
  whether the answer is noncommittal (the questions are then embedded and compared);
- context precision: `context_precision` says, per source, whether it helps to reach the
  reference answer;
- context recall: `context_recall` says, per sentence of the reference answer, whether the
  sources state it;
- citation accuracy (our own metric): `citations` says, per [n] in the answer, whether
  that one source supports the sentence citing it.

Every prompt is a versioned file (rule 7), the sources and answers are data blocks the
model is told never to obey (rule 8), and every request is logged to llm_usage at the
evaluation prices (rule 9, `eval/llm.py`). The judge never sees the golden reference
CHUNK ids, only texts, and it has no access to retrieval: it judges what it is shown.

A call that answers for the wrong items (a missing or extra index) raises JudgeError;
the caller records the metric as not available for that question instead of guessing.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.llm.client import ChatClient, ChatMessage, ChatUsage
from app.rag.citations import Source, format_sources
from app.rag.prompts import escape_block, load_prompt

PROMPTS = {
    "statements": "judge_statements_v1",
    "faithfulness": "judge_faithfulness_v1",
    "relevancy": "judge_relevancy_v1",
    "context_precision": "judge_context_precision_v1",
    "context_recall": "judge_context_recall_v1",
    "citations": "judge_citation_v1",
}
FEATURE = "eval-judge"  # llm_usage.feature is "eval-judge-<step>"


class JudgeError(RuntimeError):
    """The judge's answer does not match what was asked; `usage` is what it cost."""

    def __init__(self, message: str, usage: ChatUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


# --- the JSON shapes the model must fill ------------------------------------------------------


class StatementsOutput(BaseModel):
    statements: list[str]


class SupportVerdict(BaseModel):
    index: int
    supported: bool
    reason: str


class SupportOutput(BaseModel):
    verdicts: list[SupportVerdict]


class RelevancyOutput(BaseModel):
    questions: list[str]
    noncommittal: bool


class UsefulVerdict(BaseModel):
    n: int
    useful: bool
    reason: str


class UsefulOutput(BaseModel):
    verdicts: list[UsefulVerdict]


class AttributedVerdict(BaseModel):
    index: int
    attributed: bool
    reason: str


class AttributedOutput(BaseModel):
    verdicts: list[AttributedVerdict]


# --- what the judge returns -------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One judged item: its key (a statement or sentence index, or a source number), the
    yes/no verdict and the model's short reason."""

    key: int
    ok: bool
    reason: str


@dataclass(frozen=True)
class CitationItem:
    """One [n] of an answer: the sentence (by index), the source it cites, and every
    source number that sentence cites (the judge rule differs for one or several)."""

    index: int
    sentence: str
    source: int
    sentence_cites: tuple[int, ...]


def _check_keys(got: Sequence[int], asked: Sequence[int], step: str, usage: ChatUsage) -> None:
    if sorted(got) != sorted(asked):
        raise JudgeError(f"{step}: verdicts for {sorted(got)}, asked for {sorted(asked)}", usage)


def _sentence_blocks(items: Sequence[tuple[int, str]]) -> str:
    blocks = "\n".join(
        f'<sentence index="{i}">\n{escape_block(text)}\n</sentence>' for i, text in items
    )
    return f"<sentences>\n{blocks}\n</sentences>"


@dataclass
class Judge:
    """The six judge steps on one ChatClient (normally `eval/llm.eval_chat_client()`).

    `spent` collects the usage of every request, so a run can report the judge's cost
    apart from the cost of the answers it judges.
    """

    client: ChatClient
    max_tokens: int
    relevancy_questions: int
    spent: list[ChatUsage] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.client.model

    def _ask[M: BaseModel](self, step: str, user: str, schema: type[M]) -> tuple[M, ChatUsage]:
        messages: list[ChatMessage] = [
            {"role": "system", "content": load_prompt(PROMPTS[step]).text},
            {"role": "user", "content": user},
        ]
        out, usage = self.client.parse(
            messages, schema, feature=f"{FEATURE}-{step}", max_tokens=self.max_tokens
        )
        self.spent.append(usage)
        return out, usage

    def statements(self, question: str, answer: str) -> list[str]:
        """The answer's claims as standalone statements (empty for an answer without any)."""
        user = (
            f"<question>\n{escape_block(question)}\n</question>\n\n"
            f"<answer>\n{escape_block(answer)}\n</answer>"
        )
        out, _ = self._ask("statements", user, StatementsOutput)
        return [s.strip() for s in out.statements if s.strip()]

    def faithfulness(self, statements: Sequence[str], sources: Sequence[Source]) -> list[Verdict]:
        """Per statement: can it be inferred from the sources the answer was given?"""
        if not statements:
            return []
        items = list(enumerate(statements))
        user = f"{format_sources(sources)}\n\n{_sentence_blocks(items)}"
        out, usage = self._ask("faithfulness", user, SupportOutput)
        _check_keys([v.index for v in out.verdicts], [i for i, _ in items], "faithfulness", usage)
        by_index = {v.index: v for v in out.verdicts}
        return [Verdict(i, by_index[i].supported, by_index[i].reason.strip()) for i, _ in items]

    def relevancy(self, answer: str) -> tuple[list[str], bool]:
        """Questions the answer would answer, and whether it is noncommittal."""
        n = self.relevancy_questions
        user = f"Number of questions: {n}\n\n<answer>\n{escape_block(answer)}\n</answer>"
        out, _ = self._ask("relevancy", user, RelevancyOutput)
        return [q.strip() for q in out.questions if q.strip()][:n], out.noncommittal

    def context_precision(
        self, question: str, reference: str, sources: Sequence[Source]
    ) -> list[Verdict]:
        """Per source, in the order given: does it help to reach the reference answer?"""
        if not sources:
            return []
        user = (
            f"<question>\n{escape_block(question)}\n</question>\n\n"
            f"<reference>\n{escape_block(reference)}\n</reference>\n\n{format_sources(sources)}"
        )
        out, usage = self._ask("context_precision", user, UsefulOutput)
        _check_keys([v.n for v in out.verdicts], [s.n for s in sources], "precision", usage)
        by_n = {v.n: v for v in out.verdicts}
        return [Verdict(s.n, by_n[s.n].useful, by_n[s.n].reason.strip()) for s in sources]

    def context_recall(
        self, question: str, reference_sentences: Sequence[str], sources: Sequence[Source]
    ) -> list[Verdict]:
        """Per reference sentence: do the sources state it? No sources: none is."""
        if not reference_sentences:
            return []
        items = list(enumerate(reference_sentences))
        if not sources:  # nothing was retrieved, so nothing can be attributed
            return [Verdict(i, False, "no sources were retrieved") for i, _ in items]
        user = (
            f"<question>\n{escape_block(question)}\n</question>\n\n"
            f"{format_sources(sources)}\n\n{_sentence_blocks(items)}"
        )
        out, usage = self._ask("context_recall", user, AttributedOutput)
        _check_keys([v.index for v in out.verdicts], [i for i, _ in items], "recall", usage)
        by_index = {v.index: v for v in out.verdicts}
        return [Verdict(i, by_index[i].attributed, by_index[i].reason.strip()) for i, _ in items]

    def citations(self, items: Sequence[CitationItem], sources: Sequence[Source]) -> list[Verdict]:
        """Per citation item (keyed by its position in `items`): does that one source
        support the sentence? Only the cited sources are shown; every item's source must
        be one of them."""
        if not items:
            return []
        by_n = {s.n: s for s in sources}
        missing = sorted({it.source for it in items if it.source not in by_n})
        if missing:
            raise ValueError(f"citation items name sources that were not given: {missing}")
        cited = [by_n[n] for n in sorted({it.source for it in items})]
        blocks = "\n".join(
            f'<sentence index="{k}" source="{it.source}" '
            f'sentence_cites="{",".join(map(str, it.sentence_cites))}">\n'
            f"{escape_block(it.sentence)}\n</sentence>"
            for k, it in enumerate(items)
        )
        user = f"{format_sources(cited)}\n\n<sentences>\n{blocks}\n</sentences>"
        out, usage = self._ask("citations", user, SupportOutput)
        _check_keys([v.index for v in out.verdicts], list(range(len(items))), "citations", usage)
        by_index = {v.index: v for v in out.verdicts}
        return [
            Verdict(k, by_index[k].supported, by_index[k].reason.strip()) for k in range(len(items))
        ]


def default_judge() -> Judge:
    """The configured judge (no request is sent until a step is called)."""
    from app.config import get_settings
    from eval.llm import eval_chat_client

    s = get_settings()
    return Judge(
        eval_chat_client(),
        max_tokens=s.eval_judge_max_output_tokens,
        relevancy_questions=s.eval_relevancy_questions,
    )

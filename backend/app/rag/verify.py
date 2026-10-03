"""Verify stage: are the answer's citations real, and do they support their sentences?

Two checks:
1. In code (free, always): every [n] must point to a source that was given, and
   sentences without any citation are listed (the prompt asks for one after every
   factual sentence, so these are the claims nobody can check).
2. One Structured Outputs call (`verify_enabled`): for each sentence with valid
   citations, the model reads the cited sources and says supported or not, with a
   short reason. Unsupported sentences are flagged for the UI, never removed.

The result goes into the message meta and the stream's `verification` event. If the
call fails, the pipeline keeps the code checks and marks the model check "unavailable"
(docs/adr/0006); the answer itself is unaffected.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from app.llm.client import ChatClient, ChatMessage, ChatResponseError, ChatUsage
from app.rag.citations import Sentence, Source, format_sources, split_sentences
from app.rag.prompts import escape_block, load_prompt

VERIFY_PROMPT = "verify_v1"
FEATURE = "chat-verify"
MAX_OUTPUT_TOKENS = 1000

VerifyStatus = Literal["checked", "skipped", "unavailable"]


class Verdict(BaseModel):
    index: int
    supported: bool
    reason: str


class VerifyOutput(BaseModel):
    """The JSON shape the model must fill: one verdict per sentence index."""

    verdicts: list[Verdict]


class SentenceCheck(BaseModel):
    index: int
    text: str
    citations: list[int]
    invalid_citations: list[int]  # cited numbers with no such source
    supported: bool | None  # None: not checked by the model (no valid citation, or no call)
    reason: str = ""


class VerifyResult(BaseModel):
    """Stored in the message meta and sent as the `verification` event."""

    status: VerifyStatus  # of the model check; the code checks always ran
    sentences: list[SentenceCheck]
    invalid_citations: list[int]
    uncited_sentences: list[int]  # indexes of sentences with no citation
    unsupported_sentences: list[int]  # indexes the model judged unsupported
    reason: str = ""  # why skipped or unavailable
    prompt: str | None = None

    model_config = {"frozen": True}


def code_checks(answer: str, sources: Sequence[Source]) -> list[SentenceCheck]:
    """Check 1: citations that point nowhere, and sentences without citations."""
    valid = {s.n for s in sources}
    return [
        SentenceCheck(
            index=s.index,
            text=s.text,
            citations=list(s.citations),
            invalid_citations=[n for n in s.citations if n not in valid],
            supported=None,
        )
        for s in split_sentences(answer)
    ]


def result(
    checks: Sequence[SentenceCheck],
    status: VerifyStatus,
    *,
    reason: str = "",
    prompt: str | None = None,
) -> VerifyResult:
    invalid = sorted({n for c in checks for n in c.invalid_citations})
    return VerifyResult(
        status=status,
        sentences=list(checks),
        invalid_citations=invalid,
        uncited_sentences=[c.index for c in checks if not c.citations],
        unsupported_sentences=[c.index for c in checks if c.supported is False],
        reason=reason,
        prompt=prompt,
    )


def build_messages(sentences: Sequence[Sentence], sources: Sequence[Source]) -> list[ChatMessage]:
    """Only the cited sources, then the sentences to judge; both as data blocks."""
    blocks = "\n".join(
        f'<sentence index="{s.index}" cites="{",".join(map(str, s.citations))}">\n'
        f"{escape_block(s.text)}\n</sentence>"
        for s in sentences
    )
    user = f"{format_sources(sources)}\n\n<sentences>\n{blocks}\n</sentences>"
    return [
        {"role": "system", "content": load_prompt(VERIFY_PROMPT).text},
        {"role": "user", "content": user},
    ]


def verify(
    answer: str,
    sources: Sequence[Source],
    *,
    client: ChatClient,
    enabled: bool,
) -> tuple[VerifyResult, ChatUsage | None]:
    """Both checks. Raises if the model call fails or answers for the wrong sentences
    (the pipeline then calls `unavailable`)."""
    checks = code_checks(answer, sources)
    if not enabled:
        return result(checks, "skipped", reason="disabled"), None
    by_n = {s.n: s for s in sources}
    to_check = [
        Sentence(c.index, c.text, tuple(n for n in c.citations if n in by_n))
        for c in checks
        if any(n in by_n for n in c.citations)
    ]
    if not to_check:
        return result(checks, "skipped", reason="no sentence cites a source"), None
    cited = sorted({n for s in to_check for n in s.citations})
    out, usage = client.parse(
        build_messages(to_check, [by_n[n] for n in cited]),
        VerifyOutput,
        feature=FEATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    verdicts = {v.index: v for v in out.verdicts}
    asked = {s.index for s in to_check}
    if set(verdicts) != asked:
        raise ChatResponseError(
            f"verdicts for sentences {sorted(verdicts)}, asked for {sorted(asked)}", usage
        )
    checked = [
        c.model_copy(
            update={
                "supported": verdicts[c.index].supported,
                "reason": verdicts[c.index].reason.strip(),
            }
        )  # fmt: skip
        if c.index in asked
        else c
        for c in checks
    ]
    return result(checked, "checked", prompt=VERIFY_PROMPT), usage


def unavailable(answer: str, sources: Sequence[Source], error: str) -> VerifyResult:
    """The code checks alone, when the model check failed."""
    return result(code_checks(answer, sources), "unavailable", reason=error)

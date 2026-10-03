"""Generate stage: write the cited answer from the numbered sources, streamed.

The prompt depends on the intent: `explain` gets its own prompt (longer, step by step);
qa, compare and find_papers share `generate_v1` for now (docs/adr/0006). The system
prompt is the same for every call and comes first, so OpenAI's prompt caching can reuse
it; the sources and the question follow in the user message, as data blocks.

Abstract-only evidence: when every source given is an abstract, the user message adds a
fixed note, so the model knows before it writes. After the answer, the pipeline checks
the CITED sources too (`citations.abstract_only`), because that is what the rule is about.

Insufficient sources: the prompt tells the model to answer exactly INSUFFICIENT_SOURCES
when the sources do not answer the question. The answer is streamed, so the start of
the text is held back while it could still be that marker; only text that cannot be the
marker reaches the user. If the whole answer is the marker, nothing was shown and the
pipeline turns it into the same refusal as an empty search.
"""

from collections.abc import Generator, Sequence

from app.llm.client import ChatClient, ChatMessage, ChatResponseError, ChatStream, ChatUsage
from app.rag.citations import Source, format_sources
from app.rag.plan import Intent
from app.rag.prompts import escape_block, load_prompt

FEATURE = "chat-generate"
INSUFFICIENT_MARKER = "INSUFFICIENT_SOURCES"
GENERATE_PROMPTS: dict[Intent, str] = {
    "qa": "generate_v1",
    "compare": "generate_v1",  # its own prompt (a comparison table) comes later
    "find_papers": "generate_v1",  # its own prompt (paper cards) comes later
    "explain": "generate_explain_v1",
}
ABSTRACT_ONLY_NOTE = (
    "Every source below is an abstract only (evidence: abstract). Follow the rule for "
    "answers that rest only on abstracts."
)


def build_messages(question: str, sources: Sequence[Source], intent: Intent) -> list[ChatMessage]:
    parts = [format_sources(sources)]
    if all(s.hit.chunk_type == "abstract" for s in sources):
        parts.insert(0, f"<note>\n{ABSTRACT_ONLY_NOTE}\n</note>")
    parts.append(f"<question>\n{escape_block(question)}\n</question>")
    return [
        {"role": "system", "content": load_prompt(GENERATE_PROMPTS[intent]).text},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


class GenerateStream:
    """Iterate for the answer's text pieces; afterwards read `text`, `insufficient` and
    `usage`. Raises ChatResponseError at the end if the model wrote nothing."""

    def __init__(self, stream: ChatStream, prompt: str) -> None:
        self._stream = stream
        self.prompt = prompt
        self.text = ""
        self.insufficient = False

    @property
    def usage(self) -> ChatUsage | None:
        return self._stream.usage

    def __iter__(self) -> Generator[str]:
        held = ""  # the start of the answer, kept back while it may be the marker
        holding = True
        parts: list[str] = []
        try:
            for piece in self._stream:
                parts.append(piece)
                if not holding:
                    yield piece
                    continue
                held += piece
                start = held.lstrip()
                if INSUFFICIENT_MARKER.startswith(start) or start.startswith(INSUFFICIENT_MARKER):
                    continue  # still possibly (or certainly) the marker
                holding = False
                yield held
        finally:
            self._stream.close()  # also on early exit: logs the usage
        self.text = "".join(parts).strip()
        if holding and self.text.startswith(INSUFFICIENT_MARKER):
            self.insufficient = True
            return
        if not self.text:
            raise ChatResponseError("the model returned no text", self.usage)
        if holding:  # a short answer that only looked like the start of the marker
            yield held

    def close(self) -> None:
        self._stream.close()


def generate(
    question: str,
    sources: Sequence[Source],
    intent: Intent,
    *,
    client: ChatClient,
    max_tokens: int,
    max_tokens_explain: int,
) -> GenerateStream:
    """Open the streamed answer (the request is sent now; iterate the result for text)."""
    if not sources:
        raise ValueError("generate needs at least one source (no sources means a refusal)")
    prompt = GENERATE_PROMPTS[intent]
    limit = max_tokens_explain if intent == "explain" else max_tokens
    stream = client.stream(
        build_messages(question, sources, intent), feature=FEATURE, max_tokens=limit
    )
    return GenerateStream(stream, prompt)

"""Generate stage: write the cited answer from the numbered sources, streamed.

The prompt depends on the intent: `explain` gets its own prompt (longer, step by step);
qa, compare and find_papers share `generate_v2` for now (docs/adr/0006). The system
prompt is the same for every call and comes first, so OpenAI's prompt caching can reuse
it; the sources and the question follow in the user message, as data blocks.

Abstract-only evidence: ONLY when every source given is an abstract does the user message
carry the note (prompt file `generate_abstract_note_v1`) that asks for the one-sentence
disclaimer. The system prompt never mentions such a disclaimer and tells the model not to
comment on the kind of sources unless a note asks it to. (v1 put the disclaimer rule, with
a ready-made sentence, in every system prompt, conditioned on the sources the model would
CITE, which it cannot know when it writes its first sentence: it then wrote "abstracts
only" over a mix of sections and abstracts.) After the answer, the pipeline computes
`abstract_only` in code from the CITED sources (`citations.abstract_only`).

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
    "qa": "generate_v2",
    "compare": "generate_v2",  # its own prompt (a comparison table) comes later
    "find_papers": "generate_v2",  # its own prompt (paper cards) comes later
    "explain": "generate_explain_v2",
}
ABSTRACT_NOTE_PROMPT = "generate_abstract_note_v1"  # only when ALL sources are abstracts


def needs_abstract_note(sources: Sequence[Source]) -> bool:
    return bool(sources) and all(s.hit.chunk_type == "abstract" for s in sources)


def prompt_id(sources: Sequence[Source], intent: Intent) -> str:
    """What the trace and the message meta record, e.g. 'generate_v2' or, with the note,
    'generate_v2+generate_abstract_note_v1'."""
    base = GENERATE_PROMPTS[intent]
    return f"{base}+{ABSTRACT_NOTE_PROMPT}" if needs_abstract_note(sources) else base


def build_messages(question: str, sources: Sequence[Source], intent: Intent) -> list[ChatMessage]:
    parts = [format_sources(sources)]
    if needs_abstract_note(sources):
        parts.insert(0, f"<note>\n{load_prompt(ABSTRACT_NOTE_PROMPT).text}\n</note>")
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
    prompt = prompt_id(sources, intent)
    limit = max_tokens_explain if intent == "explain" else max_tokens
    stream = client.stream(
        build_messages(question, sources, intent), feature=FEATURE, max_tokens=limit
    )
    return GenerateStream(stream, prompt)

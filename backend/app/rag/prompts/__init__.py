"""Prompt files: one per prompt and version, `<name>_v<N>.txt` in this folder (rule 7).

Prompts are never inline strings in the code: each lives in its own file, so a change
shows up as a reviewable diff. Editing a prompt means adding a new file with the next
version number (`generate_v2.txt`) and pointing the stage at it; the old file stays, so
earlier traces and evaluation runs still name a prompt that exists. The version a call
used is recorded on its trace and in the message's meta.

Each prompt is the system message: stable instructions only, the same for every call,
so OpenAI's automatic prompt caching can reuse it. Everything that changes per call
(sources, question, today's date) goes in the user message, inside delimited blocks.

All third-party text (paper chunks, earlier answers built from them) is untrusted
input. It only ever appears inside a block such as `<source n="1">...</source>`, and
every prompt tells the model to ignore instructions found inside blocks. `escape_block`
neutralises our own tag names inside such text, so a chunk cannot close its block early
and pretend that what follows is an instruction.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
_ID = re.compile(r"^(?P<name>[a-z][a-z_]*)_v(?P<version>[1-9][0-9]*)$")

# Every tag name used to delimit data in a prompt. Inside data, "<source" becomes
# "&lt;source", which the model reads as text, not as a block boundary.
BLOCK_TAGS = ("sources", "source", "question", "conversation", "message", "latest",
              "sentences", "sentence", "note")  # fmt: skip
# Blocks that only the evaluation's prompts use (the judge and the test-set generator). They
# are escaped only where those prompts are built, so chat prompts stay exactly as they were.
EVAL_TAGS = ("answer", "reference")


@dataclass(frozen=True)
class Prompt:
    name: str  # "generate"
    version: int  # 1
    text: str

    @property
    def id(self) -> str:
        """The file stem, e.g. 'generate_v1': what traces and meta record."""
        return f"{self.name}_v{self.version}"


@lru_cache
def load_prompt(prompt_id: str) -> Prompt:
    """Read `<prompt_id>.txt`, e.g. load_prompt('generate_v1')."""
    match = _ID.match(prompt_id)
    if match is None:
        raise ValueError(f"bad prompt id {prompt_id!r}: expected '<name>_v<N>'")
    path = PROMPTS_DIR / f"{prompt_id}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"prompt file missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt file is empty: {path}")
    return Prompt(match["name"], int(match["version"]), text)


@lru_cache
def _tag_pattern(tags: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"<(/?)\s*(" + "|".join(tags) + r")\b", re.IGNORECASE)


def escape_block(text: str, extra_tags: tuple[str, ...] = ()) -> str:
    """Untrusted text, made safe to put inside a delimited block (see the module doc).

    `extra_tags` adds the tag names of the prompt being built (EVAL_TAGS for the
    evaluation's prompts) to BLOCK_TAGS."""
    return _tag_pattern(BLOCK_TAGS + extra_tags).sub(lambda m: f"&lt;{m[1]}{m[2]}", text)

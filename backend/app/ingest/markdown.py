"""Small markdown helpers shared by cleaning and chunking.

Only what this pipeline needs: ATX headings (`## Title`), fenced code blocks, display
math blocks (`$$ ... $$`) and pipe tables. All three converters we use (Hugging Face .md,
arXiv HTML -> markdown, PDF -> markdown) write headings in ATX form.
"""

import re

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def parse_heading(line: str) -> tuple[int, str] | None:
    """'## 3.1 Model' -> (2, '3.1 Model'); None if the line is not a heading.

    Bold/italic wrappers that PDF converters put around heading text are removed:
    '## **1 Introduction**' -> (2, '1 Introduction').
    """
    m = _HEADING_RE.match(line)
    if not m or not m.group(2):
        return None
    text = re.sub(r"(\*\*|__|\*|_)(.+?)\1", r"\2", m.group(2)).strip()
    return (len(m.group(1)), text) if text else None


def is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def protected_spans(lines: list[str]) -> list[tuple[int, int, str]]:
    """Code and display math blocks as (start, end, kind): lines[start:end], fences included,
    kind 'code' or 'math'. An unclosed block runs to the end of the text."""
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if _FENCE_RE.match(lines[i]):
            j = i + 1
            while j < len(lines) and not _FENCE_RE.match(lines[j]):
                j += 1
            end = min(j + 1, len(lines))
            spans.append((i, end, "code"))
            i = end
        elif stripped.startswith("$$"):
            if len(stripped) > 2 and stripped.endswith("$$"):  # '$$ x = 1 $$' on one line
                end = i + 1
            else:
                j = i + 1
                while j < len(lines) and not lines[j].strip().endswith("$$"):
                    j += 1
                end = min(j + 1, len(lines))
            spans.append((i, end, "math"))
            i = end
        else:
            i += 1
    return spans


def protected_lines(lines: list[str]) -> list[bool]:
    """True for lines inside fenced code or a display math block, fences included.

    Their content must stay exactly as it is: a '# comment' in code is not a heading,
    and a lone '2' inside an equation is not a page number.
    """
    mask = [False] * len(lines)
    for start, end, _ in protected_spans(lines):
        mask[start:end] = [True] * (end - start)
    return mask

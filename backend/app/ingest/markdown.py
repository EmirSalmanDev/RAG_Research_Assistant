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


def protected_lines(lines: list[str]) -> list[bool]:
    """True for lines inside fenced code or a display math block, fences included.

    Their content must stay exactly as it is: a '# comment' in code is not a heading,
    and a lone '2' inside an equation is not a page number.
    """
    mask: list[bool] = []
    in_code = in_math = False
    for line in lines:
        stripped = line.strip()
        if not in_math and _FENCE_RE.match(line):
            mask.append(True)
            in_code = not in_code
            continue
        if not in_code and stripped.startswith("$$"):
            mask.append(True)
            # '$$' alone or '$$ x = 1' opens/closes a block; '$$ x = 1 $$' is one line.
            if stripped == "$$" or not (len(stripped) > 2 and stripped.endswith("$$")):
                in_math = not in_math
            continue
        if not in_code and in_math and stripped.endswith("$$"):
            mask.append(True)
            in_math = False
            continue
        mask.append(in_code or in_math)
    return mask

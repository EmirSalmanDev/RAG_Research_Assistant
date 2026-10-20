"""A small BibTeX checker for the tests (and the real-data proof): balanced braces, unique
keys, required fields. It reads exactly the shape `app.features.bibtex` writes."""

import re

REQUIRED = ("author", "title", "year")
_ENTRY = re.compile(r"@(\w+)\{([^,\s]+),\n(.*?)\n\}\n?(?=\n@|\Z)", re.DOTALL)
_FIELD = re.compile(r"^\s{2}(\w+) = \{(.*)\},$", re.MULTILINE)


def balanced(text: str) -> bool:
    """Braces balance, not counting an escaped one (`\\{` and `\\}`)."""
    depth = 0
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        depth += c == "{"
        depth -= c == "}"
        if depth < 0:
            return False
        i += 1
    return depth == 0


def validate(bib: str) -> list[str]:
    """Problems found in a .bib text (empty list: it is fine)."""
    problems: list[str] = []
    if not balanced(bib):
        problems.append("unbalanced braces")
    seen: set[str] = set()
    entries = _ENTRY.findall(bib)
    if bib.count("\n@") + bib.startswith("@") != len(entries):
        problems.append("an entry could not be read")
    for kind, key, body in entries:
        if kind != "misc":
            problems.append(f"{key}: unexpected type {kind}")
        if key in seen:
            problems.append(f"duplicate key {key}")
        seen.add(key)
        fields = dict(_FIELD.findall(body))
        for name in REQUIRED:
            if not fields.get(name):
                problems.append(f"{key}: missing {name}")
        if not balanced(body):
            problems.append(f"{key}: unbalanced braces")
    return problems


def keys(bib: str) -> list[str]:
    return [key for _, key, _ in _ENTRY.findall(bib)]

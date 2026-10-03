"""Keep secrets and bulky data out of traces and stored error texts.

Error messages can quote credentials: OpenAI's 401 says "Incorrect API key provided:
sk-...abcd" (masked, but still part of a key). Everything that goes to the tracer, and
the error text stored in `messages.meta`, passes through `redact_secrets`. `sanitize`
also drops base64 `data:` URIs (the Langfuse SDK would upload those as media files) and
cuts very long strings.
"""

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[redacted]"
MAX_STRING_CHARS = 20_000

_SECRET_PATTERNS = [
    re.compile(r"\b(?:sk|pk)-lf-[A-Za-z0-9_\-*]{4,}"),  # Langfuse keys
    re.compile(r"\bsk-[A-Za-z0-9_\-*]{8,}"),  # OpenAI keys (also masked ones)
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/=*]+"),  # Authorization header values
    re.compile(r"(?i)\b(?:api[_-]?key|secret|password|token)=[^\s&\"']+"),  # key=value
]
_DATA_URI = re.compile(r"^\s*data:[^,]{0,100};base64,", re.IGNORECASE)


def redact_secrets(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def sanitize(value: Any) -> Any:
    """A JSON-friendly copy of `value` that is safe to send to the tracer."""
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if _DATA_URI.match(value):
            return "[data URI omitted]"
        text = redact_secrets(value)
        if len(text) > MAX_STRING_CHARS:
            text = text[:MAX_STRING_CHARS] + f" [... {len(text) - MAX_STRING_CHARS} chars cut]"
        return text
    if isinstance(value, Mapping):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [sanitize(v) for v in value]
    return sanitize(str(value))

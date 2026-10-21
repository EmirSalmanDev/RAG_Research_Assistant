"""Apply one configuration's settings to the whole process for one run, then undo it.

Passing a Settings object to a run is not enough: the stages read their settings from
`get_settings()`, one cached object per process. `run_chat_turn` builds its retrieval
config from it, `search` takes the collection from it, `DenseEmbedder` its model and
dimension, `embedding_cost` its price, `get_cross_encoder` the reranker. So an evaluation
that runs several configurations in one process must change what `get_settings()`
returns, for every module at once, and put it back afterwards.

`applied_settings(overrides)` does that through the environment, the same way the app
is configured (environment variables win over `.env`):
1. each override becomes an environment variable (`search_mode` -> `SEARCH_MODE`);
2. the settings cache is cleared, so the next `get_settings()` reads them;
3. every override is read back and compared: a value that did not arrive (a typo, a type
   pydantic parses differently) stops the run before anything is spent;
4. on the way out, in every case (also after an error), the environment is restored
   exactly (a variable that was unset is unset again) and the caches are cleared again.

Objects built from settings and kept per process are handled too: the shared dense
embedder is dropped with the settings cache, and the cross-encoder is keyed by its
settings (app/rag/rerank.py), so a new configuration gets its own model.

Scopes do not nest (a second one inside the first raises): two layers of overrides would
make "what ran" hard to state. Secrets, store addresses and the evaluation's own
yardstick (`eval_*`: judge model, prices, relevancy embeddings) can never be overridden.
"""

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from app.config import Settings, get_settings
from app.ingest.embed import get_dense_embedder

# Never changed by a configuration: keys, the stores a run reads and writes, and the
# judge (an ablation must not change the yardstick it is measured with).
FORBIDDEN = frozenset(
    {
        "database_url", "qdrant_url", "redis_url", "openai_api_key", "langfuse_public_key",
        "langfuse_secret_key", "langfuse_host", "github_token", "s2_api_key",
    }
)  # fmt: skip
FORBIDDEN_PREFIXES = ("eval_",)


class ScopeError(RuntimeError):
    """The overrides cannot be applied as asked (nothing is left changed)."""


_active: list[Mapping[str, Any]] = []


def check_overrides(overrides: Mapping[str, Any]) -> None:
    """Raise ScopeError for an unknown, forbidden or unsettable override."""
    fields = Settings.model_fields
    unknown = sorted(k for k in overrides if k not in fields)
    if unknown:
        raise ScopeError(f"unknown settings {unknown}")
    blocked = sorted(k for k in overrides if k in FORBIDDEN or k.startswith(FORBIDDEN_PREFIXES))
    if blocked:
        raise ScopeError(f"a configuration may not change {blocked}")
    unset = sorted(k for k, v in overrides.items() if v is None)
    if unset:
        # An environment variable cannot say "None"; leave such a field at its default.
        raise ScopeError(f"cannot set {unset} to None; leave them out instead")


def _env_value(value: Any) -> str:
    """The value as an environment variable: strings as they are, everything else as
    JSON (booleans true/false, numbers, lists and dicts), which pydantic parses back."""
    return value if isinstance(value, str) else json.dumps(value)


def clear_settings_caches() -> None:
    """Forget the cached settings and the process-wide objects built from them."""
    get_settings.cache_clear()
    get_dense_embedder.cache_clear()


@contextmanager
def applied_settings(overrides: Mapping[str, Any]) -> Iterator[Settings]:
    """The process's settings with `overrides` applied, for the duration of the block."""
    if _active:
        raise ScopeError("settings scopes do not nest")
    check_overrides(overrides)
    names = {field: field.upper() for field in overrides}
    saved = {env: os.environ.get(env) for env in names.values()}
    _active.append(overrides)
    try:
        for field, value in overrides.items():
            os.environ[names[field]] = _env_value(value)
        clear_settings_caches()
        settings = get_settings()
        wrong = {
            k: (getattr(settings, k), v) for k, v in overrides.items() if getattr(settings, k) != v
        }
        if wrong:
            raise ScopeError(f"settings did not take these values (got, wanted): {wrong}")
        yield settings
    finally:
        for env, old in saved.items():
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old
        clear_settings_caches()
        _active.clear()

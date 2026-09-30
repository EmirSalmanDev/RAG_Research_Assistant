from pathlib import Path

from alembic.config import Config

BACKEND_DIR = Path(__file__).resolve().parents[1]


def alembic_config() -> Config:
    return Config(str(BACKEND_DIR / "alembic.ini"))


# --- fake HTTP server helpers for client unit tests (no network) ---
from collections.abc import Callable, Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402

import httpx  # noqa: E402

from app.sources.http import make_client  # noqa: E402


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """An httpx client whose 'server' is a Python function."""
    return make_client("https://example.test", transport=httpx.MockTransport(handler))


class CountingLimiter:
    """Stands in for the Redis limiter: counts how many request slots were taken."""

    def __init__(self) -> None:
        self.slots = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        self.slots += 1
        yield

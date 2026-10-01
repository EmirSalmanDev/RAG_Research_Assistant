"""Shared fixtures: a throwaway Postgres database per test, and cassette hygiene."""

import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session

from app.config import Settings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class NetworkBlockedError(RuntimeError):
    """A test (or a fixture) tried to look up a host outside --allowed-hosts."""


@pytest.fixture(scope="session", autouse=True)
def block_network_for_the_whole_session(request: pytest.FixtureRequest) -> Iterator[None]:
    """Second network guard, covering module- and session-scoped fixtures too.

    pytest-recording's --block-network is a function-scoped fixture: it only covers each
    test's own setup and body, so a module- or session-scoped fixture could still reach
    the internet (that is how a tokenizer download slipped through once). Its block is
    also undone after every test, so it cannot simply be widened. This guard works one
    layer lower, on DNS lookups (`socket.getaddrinfo`, which every connection to a named
    host makes first), with the same allowed hosts, for the whole session.
    Off when recording cassettes (--record-mode other than none, or no --block-network).
    """
    import re
    import socket

    record_mode = request.config.getoption("--record-mode") or "none"
    if not request.config.getoption("--block-network") or record_mode != "none":
        yield
        return
    allowed = [h for h in (request.config.getoption("--allowed-hosts") or "").split(",") if h]
    allowed_re = re.compile("(" + ")|(".join(allowed) + ")") if allowed else None
    original = socket.getaddrinfo

    def guarded(host: bytes | str | None, *args: object, **kwargs: object) -> object:
        name = host.decode() if isinstance(host, bytes) else host
        if name is None or (allowed_re is not None and allowed_re.match(name)):
            return original(host, *args, **kwargs)  # type: ignore[arg-type]
        raise NetworkBlockedError(f"network is blocked in tests: lookup of {name!r}")

    socket.getaddrinfo = guarded  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.getaddrinfo = original


@pytest.fixture(autouse=True)
def no_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never use the developer's real keys (they could leak into cassettes).

    Empty environment variables take priority over `.env`, so clients see "no key".
    """
    from app.config import get_settings

    for name in ("OPENAI_API_KEY", "GITHUB_TOKEN", "S2_API_KEY", "LANGFUSE_PUBLIC_KEY",
                 "LANGFUSE_SECRET_KEY"):  # fmt: skip
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()


@pytest.fixture
def scratch_db_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[URL]:
    """Create an empty database `copilot_test_<random>`, yield its URL, drop it afterwards.

    `DATABASE_URL` is pointed at it (and the settings cache cleared) so Alembic's
    env.py, which reads the URL from app settings, migrates the scratch DB.
    """
    from app.config import get_settings
    from app.db.session import get_engine

    base = make_url(Settings(_env_file=None).database_url)
    name = f"copilot_test_{uuid.uuid4().hex[:8]}"
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = base.set(database=name)
    monkeypatch.setenv("DATABASE_URL", url.render_as_string(hide_password=False))
    get_settings.cache_clear()
    get_engine.cache_clear()
    try:
        yield url
    finally:
        get_engine().dispose()
        get_settings.cache_clear()
        get_engine.cache_clear()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def migrated_session(scratch_db_url: URL) -> Iterator[Session]:
    """A session on a scratch database with all migrations applied."""
    from alembic import command
    from tests.helpers import alembic_config

    command.upgrade(alembic_config(), "head")
    engine = create_engine(scratch_db_url)
    with Session(engine) as session:
        yield session
    engine.dispose()


# ---------------------------------------------------------------------------
# Qdrant: throwaway collections only. Tests never write to the real collection.
# ---------------------------------------------------------------------------


@pytest.fixture
def qdrant() -> Iterator["QdrantClient"]:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=Settings(_env_file=None).qdrant_url)
    yield client
    client.close()


def _real_collection_state(client: "QdrantClient") -> tuple[bool, int | None]:
    """(exists, point count) of the real collection, e.g. `chunks`."""
    from app.config import get_settings

    name = get_settings().qdrant_collection
    if not client.collection_exists(name):
        return False, None
    return True, client.count(name, exact=True).count


@pytest.fixture
def real_collection_untouched(qdrant: "QdrantClient") -> Iterator[None]:
    """Fails the test if the real collection's existence or point count changed."""
    before = _real_collection_state(qdrant)
    yield
    assert _real_collection_state(qdrant) == before, "the real collection changed during the test"


@pytest.fixture
def scratch_collection(qdrant: "QdrantClient") -> Iterator[str]:
    """A unique `chunks_test_<random>` name; the collection is deleted afterwards, even if
    the test fails."""
    from app.config import get_settings

    name = f"chunks_test_{uuid.uuid4().hex[:8]}"
    assert name != get_settings().qdrant_collection
    try:
        yield name
    finally:
        if qdrant.collection_exists(name):
            qdrant.delete_collection(name)


# ---------------------------------------------------------------------------
# Cassette (recorded HTTP) configuration. CLAUDE.md rules 3 and 14: strip credentials,
# and keep only metadata / abstracts / short text in committed cassettes.
# ---------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

CASSETTE_DIR = Path(__file__).parent / "cassettes"
MAX_TEXT_BYTES = 1000  # READMEs are cut to this many bytes
INVENTED_MARKDOWN = b"# Invented paper\n\nThis body replaces the real full text in the cassette.\n"
ARTIFACTS_KEPT = 10  # HF lists up to 100 models/datasets/spaces per paper; keep 10
MAX_BODY_BYTES = 300_000  # hard stop: a bigger body probably holds full paper text
TRUNCATED_TYPES = ("text/plain", "application/vnd.github.raw")  # README-like text
HF_PAPER_KEYS = {"id", "title", "summary", "upvotes", "githubRepo", "githubStars", "publishedAt"}
DAILY_ENTRIES_KEPT = 3  # daily_papers returns ~30 entries with submitter profiles; keep 3


def _scrub_hf_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Keep only paper metadata; drop submitter/user profile objects."""
    kept = {k: v for k, v in paper.items() if k in HF_PAPER_KEYS}
    if "authors" in paper:
        kept["authors"] = [{"name": a.get("name")} for a in paper["authors"]]
    return kept


def _scrub_json(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except ValueError:
        return body
    if isinstance(data, list) and data and isinstance(data[0], dict) and "paper" in data[0]:
        data = [{"paper": _scrub_hf_paper(e["paper"])} for e in data[:DAILY_ENTRIES_KEPT]]
    elif isinstance(data, dict) and "upvotes" in data and "id" in data:
        data = _scrub_hf_paper(data)
    elif isinstance(data, list) and all(isinstance(e, dict) and "id" in e for e in data):
        data = data[:ARTIFACTS_KEPT]  # HF models/datasets/spaces listing
    else:
        return body
    return json.dumps(data).encode()


def before_record_response(response: dict[str, Any]) -> dict[str, Any]:
    headers = response.get("headers", {})
    for name in list(headers):
        if name.lower() in {"set-cookie", "authorization"}:
            del headers[name]
    body: bytes = response["body"]["string"]
    content_type = next((v[0] for k, v in headers.items() if k.lower() == "content-type"), "")
    if content_type.startswith("text/markdown"):
        body = INVENTED_MARKDOWN  # HF full text of a paper: never stored (rule 14)
    elif content_type.startswith(TRUNCATED_TYPES):
        body = body[:MAX_TEXT_BYTES]  # never store whole READMEs
    elif "json" in content_type:
        body = _scrub_json(body)
    if len(body) > MAX_BODY_BYTES:
        raise RuntimeError(f"refusing to record a {len(body)}-byte body (could be full text)")
    response["body"]["string"] = body
    for name in list(headers):  # the body changed, so the old length/encoding no longer apply
        if name.lower() in {"content-length", "content-encoding"}:
            del headers[name]
    return response


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": ["authorization", "x-api-key", "cookie", "set-cookie"],
        "filter_query_parameters": ["token", "api_key", "access_token", "key"],
        "before_record_response": before_record_response,
        "decode_compressed_response": True,
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    return str(CASSETTE_DIR / request.module.__name__.split(".")[-1])


def _limiter(request: pytest.FixtureRequest, name: str, interval_s: float) -> Any:
    """Replaying a cassette needs no waiting, so use a counting stand-in. When actually
    recording (real network), use the real Redis limiter so arXiv/S2 limits are honoured."""
    from app.sources.ratelimit import RedisRateLimiter, get_redis
    from tests.helpers import CountingLimiter

    if request.config.getoption("--record-mode") in (None, "none"):
        return CountingLimiter()
    return RedisRateLimiter(get_redis(), name, min_interval_s=interval_s)


@pytest.fixture
def arxiv_limiter(request: pytest.FixtureRequest) -> Any:
    return _limiter(request, "arxiv", 3.0)


@pytest.fixture
def s2_limiter(request: pytest.FixtureRequest) -> Any:
    return _limiter(request, "s2", 1.0)

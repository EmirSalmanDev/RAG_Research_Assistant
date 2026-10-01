"""The session-wide network guard in tests/conftest.py also covers module-scoped fixtures,
which pytest-recording's per-test --block-network does not."""

import socket
from collections.abc import Iterator

import pytest

from tests.conftest import NetworkBlockedError

seen_in_module_fixture: dict[str, str] = {}


@pytest.fixture(scope="module", autouse=True)
def module_fixture() -> Iterator[None]:
    # Runs before pytest-recording's function-scoped block: our guard must already be on.
    seen_in_module_fixture["getaddrinfo"] = socket.getaddrinfo.__name__
    yield


def test_guard_is_active_in_module_scoped_fixtures() -> None:
    assert seen_in_module_fixture["getaddrinfo"] == "guarded"


def test_lookups_outside_allowed_hosts_are_refused() -> None:
    assert socket.getaddrinfo.__name__ == "guarded"  # checked first: no real lookup if not
    with pytest.raises(NetworkBlockedError, match="example.invalid"):
        socket.getaddrinfo("example.invalid", 443)


def test_allowed_hosts_still_resolve() -> None:
    assert socket.getaddrinfo("localhost", 5432)  # the local stack stays reachable

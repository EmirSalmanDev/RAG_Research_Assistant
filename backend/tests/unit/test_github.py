import httpx
import pytest

from app.sources.github import GitHubClient, parse_github_url
from tests.helpers import mock_client


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/owner/name", "owner/name"),
        ("https://github.com/owner/name/", "owner/name"),
        ("https://github.com/owner/name.git", "owner/name"),
        ("http://github.com/owner/na.me/tree/main/src", "owner/na.me"),
        ("git@github.com:owner/name.git", "owner/name"),
        ("https://example.com/owner/name", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_github_url(url: str | None, expected: str | None) -> None:
    assert parse_github_url(url) == expected


def test_repo_parses_stats_and_license_and_sends_no_token_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json={"full_name": "a/b", "stargazers_count": 42,
                                         "pushed_at": "2026-01-02T03:04:05Z",
                                         "license": {"spdx_id": "MIT"}, "other": 1})  # fmt: skip

    repo = GitHubClient(client=mock_client(handler), token=None).repo("a/b")
    assert repo is not None
    assert (repo.full_name, repo.stars, repo.license) == ("a/b", 42, "MIT")
    assert repo.last_push is not None and repo.last_push.year == 2026
    assert "authorization" not in seen[0].headers


def test_token_is_sent_only_when_set() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json={"full_name": "a/b"})

    GitHubClient(client=mock_client(handler), token="secret-token").repo("a/b")
    assert seen[0].headers["authorization"] == "Bearer secret-token"


@pytest.mark.parametrize("spdx", [None, "NOASSERTION"])
def test_unknown_license_becomes_none(spdx: str | None) -> None:
    client = GitHubClient(
        client=mock_client(
            lambda r: httpx.Response(200, json={"full_name": "a/b", "license": {"spdx_id": spdx}})
        ),
        token=None,
    )
    repo = client.repo("a/b")
    assert repo is not None and repo.license is None


def test_missing_repo_and_readme_return_none() -> None:
    client = GitHubClient(client=mock_client(lambda r: httpx.Response(404)), token=None)
    assert client.repo("a/b") is None
    assert client.readme("a/b") is None


def test_readme_asks_for_raw_text() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, text="# Invented README")

    assert (
        GitHubClient(client=mock_client(handler), token=None).readme("a/b") == "# Invented README"
    )
    assert seen[0].headers["accept"] == "application/vnd.github.raw+json"

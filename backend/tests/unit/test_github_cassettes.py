"""GitHubClient replayed from recorded responses (tests/cassettes/test_github_cassettes/)."""

import pytest

from app.sources.github import GitHubClient

REPO = "tensorflow/tensor2tensor"


@pytest.mark.vcr
def test_repo_from_cassette() -> None:
    repo = GitHubClient(token=None).repo(REPO)
    assert repo is not None and repo.full_name.lower() == REPO and repo.stars > 0


@pytest.mark.vcr
def test_readme_from_cassette_is_short() -> None:
    text = GitHubClient(token=None).readme(REPO)
    assert text and len(text.encode()) <= 1000  # truncated when recorded

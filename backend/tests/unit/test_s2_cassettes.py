"""S2Client replayed from recorded responses (tests/cassettes/test_s2_cassettes/).

Both recorded calls hit a real `429 Too Many Requests` first (S2's shared public pool,
no API key) and succeeded on the retry, so these cassettes also document that path.
"""

from typing import Any

import pytest

from app.sources.s2 import S2Client, S2Paper

PAPER_ID = "1706.03762"


def _client(limiter: Any) -> S2Client:
    # sleep is a no-op so replaying the backoff does not slow the test down
    return S2Client(limiter=limiter, api_key=None, sleep=lambda s: None)


@pytest.mark.vcr
def test_paper_from_cassette(s2_limiter: Any) -> None:
    result = _client(s2_limiter).paper(PAPER_ID)
    assert isinstance(result, S2Paper)
    assert result.citation_count and result.citation_count > 1000
    assert result.references  # the paper cites other arXiv papers
    if hasattr(s2_limiter, "slots"):  # the counting stand-in (replay), not the real limiter
        assert s2_limiter.slots == 2  # 429, then the retry: each attempt took its own slot


@pytest.mark.vcr
def test_batch_from_cassette(s2_limiter: Any) -> None:
    result = _client(s2_limiter).batch([PAPER_ID])
    assert isinstance(result[PAPER_ID], S2Paper)
    if hasattr(s2_limiter, "slots"):
        assert s2_limiter.slots == 2

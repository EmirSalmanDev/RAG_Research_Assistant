"""Pure helpers of app/ingest/enrich.py (no database)."""

import httpx
import pytest

from app.ingest.enrich import is_rate_limited, s2_counts
from app.sources.http import RetryableError
from app.sources.s2 import S2Paper


def _status_error(status: int) -> httpx.HTTPStatusError:
    return httpx.HTTPStatusError(
        "error", request=httpx.Request("GET", "https://example.org"),
        response=httpx.Response(status),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("err", "limited"),
    [
        (RetryableError("x", status=429), True),
        (RetryableError("x", status=503), False),  # a server problem, not our quota
        (_status_error(403), True),  # GitHub answers an exhausted quota with 403
        (_status_error(429), True),
        (_status_error(404), False),
        (RuntimeError("x"), False),
    ],
)
def test_is_rate_limited(err: BaseException, limited: bool) -> None:
    assert is_rate_limited(err) is limited


def test_s2_counts_report_dropped_references() -> None:
    paper = S2Paper(arxiv_id="2601.00001", citation_count=1, references=["a", "b"],
                    references_total=5)  # fmt: skip
    assert s2_counts(paper, 2) == {
        "citations": 2, "references_total": 5, "references_stored": 2,
        "references_dropped_no_arxiv_id": 3,
    }  # fmt: skip

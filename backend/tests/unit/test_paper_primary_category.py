"""primary_category_of: arXiv's explicit field first, the first listed category as fallback."""

from datetime import UTC, datetime

import pytest

from app.ingest.paper import primary_category_of
from app.sources.arxiv import ArxivPaper


def _paper(categories: list[str], primary: str) -> ArxivPaper:
    return ArxivPaper(
        arxiv_id="2601.00001", version=1, title="Invented", abstract="Invented abstract.",
        authors=["Ada Example"], categories=categories, primary_category=primary,
        published_at=datetime(2026, 1, 2, tzinfo=UTC), updated_at=datetime(2026, 1, 2, tzinfo=UTC),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("categories", "primary", "expected"),
    [
        (["cs.LG", "cs.CL"], "cs.LG", "cs.LG"),  # normal case: explicit field
        (["cs.LG", "cs.CL"], "", "cs.LG"),  # feed without primary_category: first listed
        ([], "", None),  # nothing to go on
    ],
)
def test_primary_category_of(categories: list[str], primary: str, expected: str | None) -> None:
    assert primary_category_of(_paper(categories, primary)) == expected

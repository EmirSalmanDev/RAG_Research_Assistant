"""Migration 0002 on a database that already holds rows written by M1.

The two papers are the ones M1's `make ingest-one` stored locally (1706.03762 and
2609.24972), with the same version, categories (in arXiv's order), dates and status.
Abstracts and author lists are shortened: they are not what the migration touches.
Runs on a throwaway database only (`scratch_db_url`), never on the real one.
"""

import json
from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Connection, Engine, create_engine, inspect, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

from alembic import command
from tests.helpers import alembic_config

pytestmark = pytest.mark.integration

# As M1 left them in the local database (checked with a read-only SELECT on 2026-10-01).
M1_PAPERS: list[dict[str, Any]] = [
    {
        "arxiv_id": "1706.03762",
        "version": 7,
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on ...",
        "authors": [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}],
        # The recorded arXiv cassette shows primary_category cs.CL, listed first.
        "categories": ["cs.CL", "cs.LG"],
        "published_at": "2017-06-12T17:57:34+00:00",
        "hf_upvotes": 144,
        "citation_count": 194461,
        "github_url": None,
    },
    {
        "arxiv_id": "2609.24972",
        "version": 2,
        "title": "RRSI: Regularized Recursive Self-Improvement of Agent Harnesses",
        "abstract": "Shortened abstract for the migration test.",
        "authors": [{"name": "Invented Author"}],
        "categories": ["cs.LG", "cs.AI", "cs.CL"],
        "published_at": "2026-09-21T17:54:49+00:00",
        "hf_upvotes": 218,
        "citation_count": 1,
        "github_url": "https://github.com/google-research/rrsi",
    },
]
M1_IDS = [p["arxiv_id"] for p in M1_PAPERS]

# Invented extras so every UPDATE in the migration has something to work on:
# M1 stored no full text, and only ever wrote fulltext_status 'none'.
INVENTED_TEXT = "# Invented\n\nInvented full text used only by this test.\n"
ODD_STATUS_ID = "2601.00001"  # invented paper with a status outside the allowed set


@pytest.fixture
def engine(scratch_db_url: URL) -> Iterator[Engine]:
    eng = create_engine(scratch_db_url)
    yield eng
    eng.dispose()


def _insert_m1_rows(conn: Connection) -> None:
    for paper in M1_PAPERS:
        conn.execute(
            text(
                "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,"
                " published_at, hf_upvotes, citation_count, github_url)"
                " VALUES (:arxiv_id, :version, :title, :abstract, CAST(:authors AS jsonb),"
                " :categories, :published_at, :hf_upvotes, :citation_count, :github_url)"
            ),
            {**paper, "authors": json.dumps(paper["authors"])},
        )
    conn.execute(
        text("INSERT INTO paper_texts (arxiv_id, markdown, token_count) VALUES (:id, :md, 12)"),
        {"id": "1706.03762", "md": INVENTED_TEXT},
    )
    conn.execute(
        text(
            "INSERT INTO papers (arxiv_id, version, title, abstract, authors, categories,"
            " published_at, fulltext_status)"
            " VALUES (:id, 1, 'Invented', 'Invented.', '[]', '{cs.IR}', now(), 'pending')"
        ),
        {"id": ODD_STATUS_ID},
    )


def _papers(conn: Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(text("SELECT * FROM papers ORDER BY arxiv_id")).mappings()
    return {row["arxiv_id"]: dict(row) for row in rows}


def _assert_m1_rows_intact(conn: Connection) -> None:
    """The columns M1 wrote still hold exactly what M1 wrote, in the same order."""
    papers = _papers(conn)
    for expected in M1_PAPERS:
        row = papers[expected["arxiv_id"]]
        assert row["version"] == expected["version"]
        assert row["title"] == expected["title"]
        assert row["authors"] == expected["authors"]
        assert row["categories"] == expected["categories"]  # order kept, primary first
        assert row["hf_upvotes"] == expected["hf_upvotes"]
        assert row["citation_count"] == expected["citation_count"]
        assert row["github_url"] == expected["github_url"]
        assert row["fulltext_status"] == "none"
    md = conn.execute(text("SELECT markdown FROM paper_texts WHERE arxiv_id = '1706.03762'"))
    assert md.scalar_one() == INVENTED_TEXT


def _assert_head_state(conn: Connection) -> None:
    _assert_m1_rows_intact(conn)
    papers = _papers(conn)
    for arxiv_id in M1_IDS:
        row = papers[arxiv_id]
        assert row["primary_category"] == row["categories"][0]
        assert row["project_url"] is None
    assert papers["1706.03762"]["primary_category"] == "cs.CL"  # arXiv's primary category
    assert papers["2609.24972"]["primary_category"] == "cs.LG"
    assert papers[ODD_STATUS_ID]["fulltext_status"] == "none"  # unknown status was mapped

    pt = conn.execute(
        text("SELECT version, raw_markdown, markdown, \"references\" FROM paper_texts")
    ).mappings().one()  # fmt: skip
    assert pt["version"] == 7  # filled from papers.version via the join
    assert pt["raw_markdown"] == pt["markdown"] == INVENTED_TEXT
    assert pt["references"] is None


def test_0002_on_m1_rows_upgrade_downgrade_upgrade(engine: Engine) -> None:
    cfg = alembic_config()
    command.upgrade(cfg, "0001")
    with engine.begin() as conn:
        _insert_m1_rows(conn)

    # 0001 -> head: rows survive and the new columns are filled.
    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        _assert_head_state(conn)

    # The CHECK accepts 'none' (the M1 value) and rejects anything outside the list.
    with engine.begin() as conn:
        conn.execute(text("UPDATE papers SET fulltext_status = 'none' WHERE arxiv_id = :id"),
                     {"id": "1706.03762"})  # fmt: skip
    with pytest.raises(IntegrityError, match="ck_papers_fulltext_status"):
        with engine.begin() as conn:
            conn.execute(text("UPDATE papers SET fulltext_status = 'bogus'"))

    # head -> 0001: only the added columns and the CHECK go; M1 rows survive.
    command.downgrade(cfg, "0001")
    insp = inspect(engine)
    paper_cols = {c["name"] for c in insp.get_columns("papers")}
    text_cols = {c["name"] for c in insp.get_columns("paper_texts")}
    assert paper_cols.isdisjoint({"primary_category", "project_url"})
    assert text_cols.isdisjoint({"version", "raw_markdown", "references"})
    assert "ck_papers_fulltext_status" not in {
        c["name"] for c in insp.get_check_constraints("papers")
    }
    with engine.connect() as conn:
        _assert_m1_rows_intact(conn)

    # 0001 -> head again: same result as the first upgrade.
    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        _assert_head_state(conn)

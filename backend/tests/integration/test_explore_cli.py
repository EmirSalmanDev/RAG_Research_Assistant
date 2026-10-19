"""`make compare` / `make timeline` on a scratch database: the estimate reads Postgres only
(no key, no network) and the paid path refuses cleanly without a key."""

import json
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.db.models import Chunk, Paper
from app.features.explore_cli import main

pytestmark = pytest.mark.integration


@pytest.fixture
def papers(migrated_session: Session) -> None:
    for arxiv_id in ("2601.30001", "2601.30002"):
        migrated_session.add(
            Paper(
                arxiv_id=arxiv_id, title=f"Invented {arxiv_id}", abstract="Invented abstract.",
                authors=[], categories=["cs.CL"], primary_category="cs.CL",
                published_at=datetime(2026, 1, 5, tzinfo=UTC),
            )
        )  # fmt: skip
    migrated_session.flush()
    for ord_, (arxiv_id, kind, tokens) in enumerate(
        [("2601.30001", "abstract", 200), ("2601.30001", "section", 5000),
         ("2601.30002", "abstract", 150), ("2601.30002", "analysis", 9000)]
    ):  # fmt: skip
        migrated_session.add(
            Chunk(
                id=uuid.uuid4(), arxiv_id=arxiv_id, chunk_type=kind, section_path=None,
                ord=ord_, text="x", token_count=tokens,
            )
        )  # fmt: skip
    migrated_session.commit()


def test_compare_estimate_is_an_upper_bound_from_stored_sizes(
    papers: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", "2601.30001", "2601.30002", "--estimate"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == "1 embeddings + 1 chat"
    # sources are cut at compare_source_max_tokens (800): the 5000-token section counts 800;
    # the analysis chunk (9000) is not evidence and is not counted
    assert 0 < out["chat_input_tokens_max"] < 20_000
    assert out["chat_output_tokens_max"] == 4000
    assert 0 < out["cost_usd_max"] < 0.01


def test_timeline_estimate(papers: None, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["timeline", "adapters", "--estimate", "--key-papers", "2"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["chat_input_tokens_max"] > 0 and out["cost_usd_max"] > 0


def test_bad_requests_exit_2_and_a_paid_run_without_a_key_exits_3(
    papers: None, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["compare", "2601.30001", "--estimate"]) == 2  # one paper
    assert main(["compare", "2601.30001", "9999.99999", "--estimate"]) == 2  # unknown paper
    assert "unknown paper" in capsys.readouterr().err
    assert main(["timeline", "ab", "--estimate"]) == 2  # topic too short
    # No key in the test environment: the paid path stops before any request.
    assert main(["compare", "2601.30001", "2601.30002"]) == 3
    assert "OPENAI_API_KEY" in capsys.readouterr().err

"""`make ingest-pdf`: the estimate parses in memory and stores nothing; refusals have exit
codes; a paid run needs a key. PDFs are generated from invented text."""

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Paper
from app.ingest.upload_cli import main
from tests.pdf_fixtures import make_blank_pdf, make_pdf

pytestmark = pytest.mark.integration


def _write(tmp_path: Path, data: bytes, name: str = "paper.pdf") -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_the_estimate_counts_chunks_and_tokens_and_stores_nothing(
    migrated_session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file = _write(tmp_path, make_pdf(3, "cli"))
    assert main([file, "--title", "An Invented Paper", "--estimate"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["id"].startswith("local-") and out["chunks"] >= 2
    assert out["embedding_tokens"] > 100 and 0 < out["cost_usd_max"] < 0.001
    assert migrated_session.scalar(select(func.count()).select_from(Paper)) == 0


def test_refusals_have_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "missing.pdf"), "--title", "T", "--estimate"]) == 2
    assert main([_write(tmp_path, b"not a pdf", "a.pdf"), "--title", "T", "--estimate"]) == 2
    assert main([_write(tmp_path, make_blank_pdf(), "b.pdf"), "--title", "T", "--estimate"]) == 2
    assert main([_write(tmp_path, make_pdf(2, "t"), "c.pdf"), "--title", " ", "--estimate"]) == 2
    err = capsys.readouterr().err
    assert "not a PDF" in err and "no text layer" in err


def test_a_paid_run_without_a_key_stops_before_parsing_or_storing(
    migrated_session: Session, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    file = _write(tmp_path, make_pdf(2, "nokey"))
    assert main([file, "--title", "T"]) == 3
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert migrated_session.scalar(select(func.count()).select_from(Paper)) == 0

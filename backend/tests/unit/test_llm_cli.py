"""app/llm/cli.py with the batch functions replaced (no database, no OpenAI)."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from app.llm import batch, cli


@contextmanager
def fake_scope() -> Iterator[object]:
    yield object()


@pytest.fixture(autouse=True)
def no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "session_scope", fake_scope)
    monkeypatch.setattr(cli, "close_qdrant", lambda: None)


def test_dry_run_creates_no_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, Any] = {}

    def fake_submit(session: object, client: object, **kw: Any) -> dict[str, Any]:
        seen.update(client=client, **kw)
        return {"selected": ["2601.00001"]}

    monkeypatch.setattr(batch, "submit", fake_submit)
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("a dry run needs no client"))
    assert cli.main(["submit", "--dry-run", "--arxiv-id", "2601.00001v2", "--max-papers", "3"]) == 0
    assert seen == {"client": None, "arxiv_ids": ["2601.00001"], "max_papers": 3, "dry_run": True}
    assert json.loads(capsys.readouterr().out) == {"selected": ["2601.00001"]}


def test_collect_without_open_batches_creates_no_client(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(batch, "open_batch_count", lambda session: 0)
    monkeypatch.setattr(cli, "_client", lambda: pytest.fail("nothing to collect"))
    assert cli.main(["collect"]) == 0
    assert json.loads(capsys.readouterr().out) == {"open": 0}


def test_an_ambiguous_submit_prints_the_row_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def ambiguous(session: object, client: object, **kw: Any) -> dict[str, Any]:
        raise batch.AmbiguousSubmitError(7, TimeoutError("invented timeout"))

    monkeypatch.setattr(batch, "submit", ambiguous)
    monkeypatch.setattr(cli, "_client", lambda: object())
    assert cli.main(["submit", "--arxiv-id", "2601.00001"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ambiguous_batch_row_id": 7}
    assert "row 7" in captured.err and "nothing was resubmitted" in captured.err

import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from app.ingest import cli
from app.ingest.fulltext import FulltextError
from app.ingest.paper import IngestError, IngestResult


@contextmanager
def fake_scope() -> Iterator[object]:
    yield object()


@pytest.fixture(autouse=True)
def no_database(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    closed: list[bool] = []
    monkeypatch.setattr(cli, "session_scope", fake_scope)
    monkeypatch.setattr(cli, "default_clients", lambda: None)
    monkeypatch.setattr(cli, "close_qdrant", lambda: closed.append(True))
    return closed


def test_metadata_only_keeps_the_old_behaviour(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], no_database: list[bool]
) -> None:
    result = IngestResult(arxiv_id="2601.00001", counts={"papers": 1}, errors={})
    monkeypatch.setattr(cli, "ingest_one", lambda session, clients, arxiv_id: result)
    monkeypatch.setattr(cli, "_run_full", lambda *a: pytest.fail("full pipeline must not run"))
    assert cli.main(["one", "2601.00001", "--metadata-only"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"] == {"papers": 1}
    assert no_database == [True]  # the Qdrant client is closed before exit


def test_the_default_runs_the_full_pipeline_with_the_forced_source(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[tuple[str, str | None]] = []

    def full(session: object, arxiv_id: str, force_source: str | None) -> dict[str, object]:
        seen.append((arxiv_id, force_source))
        return {"arxiv_id": arxiv_id, "fulltext_status": "indexed"}

    monkeypatch.setattr(cli, "_run_full", full)
    assert cli.main(["one", "2601.00001", "--force-source", "pdf"]) == 0
    assert seen == [("2601.00001", "pdf")]
    assert json.loads(capsys.readouterr().out)["fulltext_status"] == "indexed"


@pytest.mark.parametrize("error", [IngestError("arXiv has no paper"), FulltextError("no text")])
def test_failures_return_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], error: Exception
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise error

    monkeypatch.setattr(cli, "_run_full", boom)
    assert cli.main(["one", "9999.99999"]) == 1
    assert "ingest-one failed" in capsys.readouterr().err


def test_force_source_with_metadata_only_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli.main(["one", "2601.00001", "--metadata-only", "--force-source", "hf"])


def test_unknown_source_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli.main(["one", "2601.00001", "--force-source", "web"])


def test_enrich_prints_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from app.ingest import enrich

    seen: list[str] = []

    def fake(session: object, clients: object, arxiv_id: str) -> enrich.EnrichResult:
        seen.append(arxiv_id)
        return enrich.EnrichResult(counts={"repos": 1}, errors={"s2": "HTTP 429"})

    monkeypatch.setattr(enrich, "enrich_paper", fake)
    monkeypatch.setattr(enrich, "default_enrich_clients", lambda: None)
    assert cli.main(["enrich", "2601.00001v2"]) == 0
    assert seen == ["2601.00001"]  # the version suffix is stripped
    out = json.loads(capsys.readouterr().out)
    assert out == {"arxiv_id": "2601.00001", "counts": {"repos": 1}, "errors": {"s2": "HTTP 429"}}


@pytest.mark.parametrize("arxiv_id", ["2601.00001", "not-an-id/../x"])
def test_enrich_fails_for_unknown_papers_and_bad_ids(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], arxiv_id: str
) -> None:
    from app.ingest import enrich

    def not_held(*a: object) -> None:
        raise enrich.NotHeldError("paper 2601.00001 is not in the database (ingest it first)")

    monkeypatch.setattr(enrich, "enrich_paper", not_held)
    monkeypatch.setattr(enrich, "default_enrich_clients", lambda: None)
    assert cli.main(["enrich", arxiv_id]) == 1
    assert "enrich-one failed" in capsys.readouterr().err

import pytest

from app.ingest import cli
from app.ingest.paper import IngestError


def test_cli_returns_1_when_arxiv_step_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from contextlib import contextmanager

    @contextmanager
    def fake_scope():  # type: ignore[no-untyped-def]
        yield object()

    def boom(*a: object, **k: object) -> None:
        raise IngestError("arXiv has no paper 9999.99999")

    monkeypatch.setattr(cli, "session_scope", fake_scope)
    monkeypatch.setattr(cli, "default_clients", lambda: None)
    monkeypatch.setattr(cli, "ingest_one", boom)
    assert cli.main(["one", "9999.99999"]) == 1
    assert "ingest-one failed" in capsys.readouterr().err

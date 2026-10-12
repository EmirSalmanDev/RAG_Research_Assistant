"""The schema export: deterministic, covers the routes the frontend uses, --check works."""

import json
from pathlib import Path

from app.export_openapi import main, render


def test_render_is_deterministic_and_has_the_frontend_routes() -> None:
    assert render() == render()
    paths = json.loads(render())["paths"]
    for route in ("/chat", "/search", "/papers/{arxiv_id}", "/chunks/{chunk_id}"):
        assert route in paths


def test_write_then_check(tmp_path: Path) -> None:
    target = tmp_path / "openapi.json"
    assert main([str(target), "--check"]) == 1  # missing file counts as stale
    assert main([str(target)]) == 0
    assert main([str(target), "--check"]) == 0
    target.write_text("{}\n", encoding="utf-8")
    assert main([str(target), "--check"]) == 1

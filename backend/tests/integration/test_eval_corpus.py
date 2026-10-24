"""The corpus bundle the evaluation workflow restores (eval/corpus.py), on a throwaway
Postgres database and a `chunks_test_<random>` Qdrant collection. The dump is invented
(no docker call); the Qdrant snapshot is real and deleted from the server afterwards."""

import json
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient
from qdrant_client import models as m
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, Paper
from app.ingest.index import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection
from eval.corpus import (
    DUMP,
    MANIFEST,
    Counts,
    _download,
    alembic_revision,
    check_manifest,
    counts,
    export,
    pg_dump_command,
    sha256_file,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("real_collection_untouched")]


@pytest.fixture
def corpus(
    migrated_session: Session, qdrant: QdrantClient, scratch_collection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Session, QdrantClient, str]:  # fmt: skip
    monkeypatch.setenv("QDRANT_COLLECTION", scratch_collection)
    monkeypatch.setenv("EMBEDDING_DIM", "4")
    get_settings.cache_clear()
    migrated_session.add(Paper(arxiv_id="2601.00021", title="Invented", abstract="A.", authors=[],
                               categories=["cs.CL"], primary_category="cs.CL",
                               published_at=datetime(2026, 9, 1, tzinfo=UTC)))  # fmt: skip
    migrated_session.flush()
    point = uuid.uuid5(uuid.NAMESPACE_URL, "corpus-chunk")
    migrated_session.add(
        Chunk(id=point, arxiv_id="2601.00021", chunk_type="abstract", section_path="Abstract",
              ord=0, text="A.", token_count=1)
    )  # fmt: skip
    migrated_session.commit()
    ensure_collection(qdrant, scratch_collection, 4)
    qdrant.upsert(scratch_collection, points=[m.PointStruct(
        id=str(point), payload={"arxiv_id": "2601.00021"},
        vector={DENSE_VECTOR: [0.1, 0.2, 0.3, 0.4],
                SPARSE_VECTOR: m.SparseVector(indices=[1], values=[1.0])},
    )], wait=True)  # fmt: skip
    return migrated_session, qdrant, scratch_collection


def test_an_export_bundles_the_dump_the_snapshot_and_a_matching_manifest(
    corpus: tuple[Session, QdrantClient, str], tmp_path: Path
) -> None:
    session, client, collection = corpus
    s = get_settings()
    commands: list[list[str]] = []

    def run(cmd: Any, target: Path) -> None:
        commands.append(list(cmd))
        target.write_bytes(b"invented dump")

    bundle = export(tmp_path / "out", s, session, client, run=run, download=_download,
                    git_sha="invented-sha")  # fmt: skip
    assert commands == [pg_dump_command(s)]
    with tarfile.open(bundle) as tar:
        assert sorted(tar.getnames()) == sorted([MANIFEST, DUMP, f"{collection}.snapshot"])
        tar.extractall(tmp_path / "restored", filter="data")
    data = json.loads((tmp_path / "restored" / MANIFEST).read_text())
    assert data["counts"] == {"papers": 1, "chunks": 1, "points": 1}
    assert data["collection"] == collection and data["embedding_dim"] == 4
    assert client.list_snapshots(collection) == []  # the server's copy was deleted again
    found = counts(session, client, collection)
    revision = alembic_revision(session)
    assert check_manifest(data, s, found, revision, tmp_path / "restored") == []
    # A restore that lost points, a changed file and another schema are all reported.
    (tmp_path / "restored" / DUMP).write_bytes(b"changed")
    problems = check_manifest(data, s, Counts(1, 1, 0), "0005", tmp_path / "restored")
    assert any("points: 0 restored" in p for p in problems)
    assert any("SHA-256 differs" in p for p in problems)
    assert any("alembic revision 0005" in p for p in problems)
    assert sha256_file(bundle)


def test_the_dump_takes_only_the_two_corpus_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://reader:x@localhost:5432/corpusdb")
    get_settings.cache_clear()
    cmd = pg_dump_command(get_settings())
    assert cmd[:5] == ["docker", "compose", "exec", "-T", "postgres"]
    assert cmd[cmd.index("-U") + 1] == "reader" and cmd[cmd.index("-d") + 1] == "corpusdb"
    assert [cmd[i + 1] for i, a in enumerate(cmd) if a == "-t"] == ["papers", "chunks"]
    assert "--data-only" in cmd and "x" not in cmd  # no password on the command line

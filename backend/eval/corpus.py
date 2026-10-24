"""The corpus bundle the evaluation workflow restores in CI (.github/workflows/eval.yml).

An evaluation needs the indexed corpus: the questions' reference chunks and every other
chunk that competes with them. A GitHub-hosted runner starts with empty stores, and the
corpus cannot be committed (it holds the full text of papers; the repository will be
public). So the owner exports a bundle on the machine that holds the corpus, keeps it at a
private location of their choice, and gives the workflow its URL and SHA-256 as
repository secrets (EVAL_CORPUS_URL, EVAL_CORPUS_SHA256). The workflow downloads it,
checks the hash, applies the migrations, restores the two tables and the collection into
its service containers, and runs `check` before any paid call.

    python -m eval.corpus export --out DIR --dry-run   # what would be exported (read-only)
    python -m eval.corpus export --out DIR             # writes DIR/corpus.tar.gz (local only)
    python -m eval.corpus check DIR/manifest.json      # the restored stores match (CI)

A bundle (`corpus.tar.gz`) holds:
- `manifest.json`: format, when, the commit, the alembic revision, the collection, the
  embedding model and size, the counts (papers, chunks, points) and each file's SHA-256;
- `postgres.dump`: `pg_dump --data-only --format=custom` of the tables `papers` and
  `chunks` only (no conversations, library, interests or spend: nothing personal);
- `<collection>.snapshot`: a Qdrant snapshot of the app's collection.

Export writes to the stores only in Qdrant's own snapshot folder: it asks Qdrant for a
snapshot of the collection, downloads it and deletes it there again; the collection and
Postgres are only read. `pg_dump` runs inside the postgres container
(`docker compose exec -T postgres ...`), as the backups do.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Chunk, Paper

FORMAT = 1
TABLES = ("papers", "chunks")
DUMP = "postgres.dump"
MANIFEST = "manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


class CorpusError(RuntimeError):
    """The bundle or the restored stores are not what the manifest says."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Counts:
    papers: int
    chunks: int
    points: int

    def as_dict(self) -> dict[str, int]:
        return {"papers": self.papers, "chunks": self.chunks, "points": self.points}


def counts(session: Session, client: QdrantClient, collection: str) -> Counts:
    papers = int(session.scalar(select(func.count()).select_from(Paper)) or 0)
    chunks = int(session.scalar(select(func.count()).select_from(Chunk)) or 0)
    points = (
        client.count(collection, exact=True).count if client.collection_exists(collection) else 0
    )
    return Counts(papers, chunks, points)


def alembic_revision(session: Session) -> str | None:
    return session.execute(text("SELECT version_num FROM alembic_version")).scalar()


def manifest(
    s: Settings, found: Counts, revision: str | None, git_sha: str | None, files: dict[str, str]
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "alembic_revision": revision,
        "collection": s.qdrant_collection,
        "embedding_model": s.embedding_model,
        "embedding_dim": s.embedding_dim,
        "counts": found.as_dict(),
        "files": files,
    }


def check_manifest(data: dict[str, Any], s: Settings, found: Counts, revision: str | None,
                   folder: Path | None = None) -> list[str]:  # fmt: skip
    """Every way the restored stores (and, with `folder`, the files) differ from the manifest."""
    problems: list[str] = []
    if data.get("format") != FORMAT:
        problems.append(f"bundle format {data.get('format')}, this code reads {FORMAT}")
    if data.get("alembic_revision") != revision:
        problems.append(
            f"alembic revision {revision}, the bundle was made at {data.get('alembic_revision')}"
        )
    for key, value in (("collection", s.qdrant_collection), ("embedding_model", s.embedding_model),
                       ("embedding_dim", s.embedding_dim)):  # fmt: skip
        if data.get(key) != value:
            problems.append(f"{key}: settings say {value!r}, the bundle {data.get(key)!r}")
    for key, value in found.as_dict().items():
        if data.get("counts", {}).get(key) != value:
            problems.append(f"{key}: {value} restored, the bundle holds {data['counts'].get(key)}")
    if folder is not None:
        for name, digest in data.get("files", {}).items():
            path = folder / name
            if not path.is_file():
                problems.append(f"{name} is missing")
            elif sha256_file(path) != digest:
                problems.append(f"{name}: SHA-256 differs from the manifest")
    return problems


# --- export (the owner's machine) -------------------------------------------------------------


def pg_dump_command(s: Settings) -> list[str]:
    """pg_dump inside the compose postgres container: data of the two tables only. The user
    and database come from DATABASE_URL (the container trusts local connections)."""
    from sqlalchemy.engine import make_url

    url = make_url(s.database_url)
    user, database = url.username or "postgres", url.database or "postgres"
    return [
        "docker", "compose", "exec", "-T", "postgres", "pg_dump", "-U", user, "-d", database,
        "--data-only", "--format=custom", *[arg for t in TABLES for arg in ("-t", t)],
    ]  # fmt: skip


def export(
    out: Path, s: Settings, session: Session, client: QdrantClient, *,
    run: Callable[[Sequence[str], Path], None], download: Callable[[str, Path], None],
    git_sha: str | None,
) -> Path:  # fmt: skip
    """Write the bundle into `out` and return the tar.gz. `run(cmd, target)` runs a command
    with its stdout into `target`; `download(url, target)` fetches a local Qdrant URL."""
    out.mkdir(parents=True, exist_ok=True)
    found = counts(session, client, s.qdrant_collection)
    revision = alembic_revision(session)
    run(pg_dump_command(s), out / DUMP)
    description = client.create_snapshot(s.qdrant_collection, wait=True)
    if description is None:
        raise CorpusError("Qdrant did not describe the snapshot it made")
    snapshot = out / f"{s.qdrant_collection}.snapshot"
    try:
        download(f"{s.qdrant_url}/collections/{s.qdrant_collection}/snapshots/{description.name}",
                 snapshot)  # fmt: skip
    finally:
        client.delete_snapshot(s.qdrant_collection, description.name, wait=True)
    files = {DUMP: sha256_file(out / DUMP), snapshot.name: sha256_file(snapshot)}
    (out / MANIFEST).write_text(json.dumps(manifest(s, found, revision, git_sha, files), indent=1))
    bundle = out / "corpus.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        for name in (MANIFEST, DUMP, snapshot.name):
            tar.add(out / name, arcname=name)
    return bundle


def _run_to_file(cmd: Sequence[str], target: Path) -> None:
    with target.open("wb") as f:
        subprocess.run(list(cmd), stdout=f, check=True, cwd=REPO_ROOT)


def _download(url: str, target: Path) -> None:
    import httpx

    with httpx.stream("GET", url, timeout=600) as response, target.open("wb") as f:
        response.raise_for_status()
        for block in response.iter_bytes():
            f.write(block)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.corpus", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true", help="print what would be exported")
    c = sub.add_parser("check")
    c.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)

    from app.db.session import session_scope
    from app.ingest.index import close_qdrant, get_qdrant
    from eval.run import git_commit

    s = get_settings()
    client = get_qdrant()
    try:
        with session_scope() as session:
            if args.command == "check":
                data = json.loads(args.manifest.read_text())
                found = counts(session, client, s.qdrant_collection)
                problems = check_manifest(data, s, found, alembic_revision(session),
                                          args.manifest.parent)  # fmt: skip
                if problems:
                    print("the restored corpus does not match its manifest:\n  "
                          + "\n  ".join(problems), file=sys.stderr)  # fmt: skip
                    return 2
                print(f"corpus ok: {found.as_dict()}")
                return 0
            found = counts(session, client, s.qdrant_collection)
            if args.dry_run:
                print(f"would export {found.as_dict()} at alembic {alembic_revision(session)}:")
                print("  " + " ".join(pg_dump_command(s)) + f" > {args.out / DUMP}")
                print(f"  a Qdrant snapshot of {s.qdrant_collection!r}, downloaded, then deleted")
                return 0
            bundle = export(args.out, s, session, client, run=_run_to_file, download=_download,
                            git_sha=git_commit())  # fmt: skip
    finally:
        close_qdrant()
    print(f"{bundle} (sha256 {sha256_file(bundle)}); keep it private, it holds paper text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

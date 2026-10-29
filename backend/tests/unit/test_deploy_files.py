"""The files that start, build and release the stack, checked by parsing them: Docker and
GitHub Actions cannot run here, so these tests pin what each file must do and that every
command it calls exists.

PyYAML is not a direct dependency (see test_eval_workflow.py); GitHub's `on:` key is read by
PyYAML as the boolean True."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _yaml(path: str) -> dict[Any, Any]:
    data = yaml.safe_load((ROOT / path).read_text())
    assert isinstance(data, dict)
    return data


def _steps_text(job: dict[str, Any]) -> str:
    return str(yaml.dump(job["steps"]))


# --- docker-compose.yml / Makefile --------------------------------------------------------


def test_beat_is_only_started_by_its_profile_and_nothing_else_is_in_one_but_the_frontend() -> None:
    services = _yaml("docker-compose.yml")["services"]
    assert services["beat"]["profiles"] == ["scheduler"]
    profiled = {name for name, svc in services.items() if "profiles" in svc}
    assert profiled == {"beat", "frontend"}  # a plain `make up` starts everything else


def test_make_up_does_not_name_the_scheduler_profile_and_up_scheduler_does() -> None:
    makefile = (ROOT / "Makefile").read_text()
    up = re.search(r"^up:\n((?:\t.*\n)+)", makefile, re.M)
    up_scheduler = re.search(r"^up-scheduler:\n((?:\t.*\n)+)", makefile, re.M)
    assert up and up_scheduler
    assert "scheduler" not in up.group(1)
    assert "--profile scheduler" in up_scheduler.group(1)
    assert "--wait" in up.group(1) and "--wait" in up_scheduler.group(1)


def test_make_down_also_stops_the_services_in_profiles() -> None:
    # Without --profile '*', `docker compose down` leaves beat and the frontend running.
    down = re.search(r"^down:\n((?:\t.*\n)+)", (ROOT / "Makefile").read_text(), re.M)
    assert down and "--profile '*'" in down.group(1)


# --- ci.yml: the browser tests run on main only ------------------------------------------


def test_the_browser_tests_run_on_main_only() -> None:
    jobs = _yaml(".github/workflows/ci.yml")["jobs"]
    assert jobs["e2e"]["if"] == "github.ref == 'refs/heads/main'"
    assert "playwright test" in _steps_text(jobs["e2e"])
    for name in ("backend", "frontend"):
        assert "if" not in jobs[name]  # lint and unit tests run on every push


# --- release.yml ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def release() -> dict[Any, Any]:
    return _yaml(".github/workflows/release.yml")


def test_release_runs_on_version_tags_only(release: dict[Any, Any]) -> None:
    trigger = release[True]  # `on:`
    assert list(trigger) == ["push"]
    assert trigger["push"] == {"tags": ["v[0-9]+.[0-9]+.[0-9]+*"]}


def test_release_builds_both_images_for_both_platforms_and_pushes_them(
    release: dict[Any, Any],
) -> None:
    job = release["jobs"]["images"]
    contexts = {m["name"]: m["context"] for m in job["strategy"]["matrix"]["include"]}
    assert contexts == {"backend": "backend", "frontend": "frontend"}
    for context in contexts.values():
        assert (ROOT / context / "Dockerfile").is_file()
    build = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("docker/build-push"))
    assert build["with"]["platforms"] == "linux/amd64,linux/arm64"
    assert build["with"]["push"] is True
    uses = [str(s.get("uses", "")) for s in job["steps"]]
    assert any(u.startswith("docker/setup-qemu-action") for u in uses)  # arm64 on an amd64 runner
    assert any(u.startswith("docker/login-action") for u in uses)


def test_release_writes_to_ghcr_with_the_job_token_and_nothing_wider(
    release: dict[Any, Any],
) -> None:
    assert release["permissions"] == {"contents": "read"}  # the default for every job
    job = release["jobs"]["images"]
    assert job["permissions"] == {"contents": "read", "packages": "write"}
    text = _steps_text(job)
    assert "ghcr.io" in text and "secrets.GITHUB_TOKEN" in text
    secrets = set(
        re.findall(r"secrets\.([A-Z_]+)", (ROOT / ".github/workflows/release.yml").read_text())
    )
    assert secrets == {"GITHUB_TOKEN"}  # no other secret is needed to release


def test_release_only_publishes_a_commit_that_is_on_main(release: dict[Any, Any]) -> None:
    assert release["jobs"]["images"]["needs"] == "verify"
    verify = release["jobs"]["verify"]
    assert verify["steps"][0]["with"]["fetch-depth"] == 0  # the whole history, to test ancestry
    assert "merge-base --is-ancestor" in _steps_text(verify)


def test_release_tags_are_semver_and_latest_skips_prereleases(release: dict[Any, Any]) -> None:
    meta = next(
        s
        for s in release["jobs"]["images"]["steps"]
        if str(s.get("uses", "")).startswith("docker/metadata")
    )
    tags = meta["with"]["tags"]
    assert (
        "type=semver,pattern={{version}}" in tags
        and "type=semver,pattern={{major}}.{{minor}}" in tags
    )
    assert meta["with"]["flavor"] == "latest=auto"


# --- backend/.dockerignore ------------------------------------------------------------------


def _docker_ignored(path: str, patterns: list[str]) -> bool:
    """Docker's .dockerignore rule, as far as this file uses it: a pattern is matched against
    the path from the build context and against each of its parent directories (an ignored
    directory takes its contents with it). `**` matches any number of directories, `*` and
    `?` stay inside one path part. No `!` exceptions are used here."""

    def regex(pattern: str) -> re.Pattern[str]:
        out, i = "", 0
        while i < len(pattern):
            if pattern.startswith("**/", i):
                out, i = out + "(?:.*/)?", i + 3
            elif pattern[i] == "*":
                out, i = out + "[^/]*", i + 1
            elif pattern[i] == "?":
                out, i = out + "[^/]", i + 1
            else:
                out, i = out + re.escape(pattern[i]), i + 1
        return re.compile(out)

    parts = path.split("/")
    prefixes = ["/".join(parts[:n]) for n in range(1, len(parts) + 1)]
    return any(regex(p).fullmatch(prefix) for p in patterns for prefix in prefixes)


def _dockerignore() -> list[str]:
    lines = (ROOT / "backend/.dockerignore").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def test_the_matcher_reads_a_top_level_pattern_as_top_level_only() -> None:
    # What the file had before: `__pycache__` kept the nested caches in the image.
    assert _docker_ignored("__pycache__/x.pyc", ["__pycache__"])
    assert not _docker_ignored("app/ingest/__pycache__/tasks.cpython-312.pyc", ["__pycache__"])


def test_the_backend_image_gets_no_python_caches_at_any_depth() -> None:
    patterns = _dockerignore()
    assert "**/__pycache__" in patterns and "**/*.pyc" in patterns
    for cache in (
        "__pycache__/x.cpython-312.pyc",
        "app/__pycache__/config.cpython-312.pyc",
        "app/ingest/__pycache__/tasks.cpython-312.pyc",
        "eval/stray.pyc",
    ):
        assert _docker_ignored(cache, patterns), cache
    for kept in (
        "app/ingest/tasks.py",
        "app/rag/prompts/analyze_v1.txt",
        "app/ingest/data/bm25_stopwords_english.txt",
        "alembic/env.py",
        "pyproject.toml",
        "uv.lock",
    ):
        assert not _docker_ignored(kept, patterns), kept

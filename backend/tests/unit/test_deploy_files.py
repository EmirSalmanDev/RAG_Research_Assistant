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


def test_beat_is_only_started_by_its_profile_and_nothing_else_is_in_one() -> None:
    services = _yaml("docker-compose.yml")["services"]
    assert services["beat"]["profiles"] == ["scheduler"]
    profiled = {name for name, svc in services.items() if "profiles" in svc}
    assert profiled == {"beat"}  # a plain `make up` starts everything else, the frontend too


def test_make_up_starts_the_frontend_once_the_api_is_healthy() -> None:
    frontend = _yaml("docker-compose.yml")["services"]["frontend"]
    assert "profiles" not in frontend
    # Without the condition Compose starts it as soon as the api container exists.
    assert frontend["depends_on"] == {"api": {"condition": "service_healthy"}}
    # `make up --wait` waits for a healthcheck only if the service has one.
    test = frontend["healthcheck"]["test"]
    assert test[:3] == ["CMD", "node", "-e"] and "localhost:3000" in test[3]


def test_every_published_port_is_bound_to_localhost() -> None:
    # Without a host IP, Docker publishes on 0.0.0.0 and the service is reachable from the LAN.
    services = _yaml("docker-compose.yml")["services"]
    for name, svc in services.items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), (name, port)
    assert services["frontend"]["ports"] == ["127.0.0.1:3000:3000"]


def test_make_up_does_not_name_the_scheduler_profile_and_up_scheduler_does() -> None:
    makefile = (ROOT / "Makefile").read_text()
    up = re.search(r"^up:\n((?:\t.*\n)+)", makefile, re.M)
    up_scheduler = re.search(r"^up-scheduler:\n((?:\t.*\n)+)", makefile, re.M)
    assert up and up_scheduler
    assert "scheduler" not in up.group(1)
    assert "--profile scheduler" in up_scheduler.group(1)
    assert "--wait" in up.group(1) and "--wait" in up_scheduler.group(1)


def test_the_compose_frontend_is_built_to_proxy_to_the_api_service() -> None:
    build = _yaml("docker-compose.yml")["services"]["frontend"]["build"]
    assert build["context"] == "./frontend"
    # Compose fills ${VAR:-default} from the shell or .env; unset means the service name.
    assert build["args"] == {"API_ORIGIN": "${FRONTEND_API_ORIGIN:-http://api:8000}"}


def test_make_down_also_stops_the_services_in_profiles() -> None:
    # Without --profile '*', `docker compose down` leaves beat running.
    down = re.search(r"^down:\n((?:\t.*\n)+)", (ROOT / "Makefile").read_text(), re.M)
    assert down and "--profile '*'" in down.group(1)


def test_every_make_line_that_builds_skips_the_default_attestations() -> None:
    # BuildKit's default attestation is new on every build, so even a fully cached build gets
    # a new image ID and `make up` recreates the container (and kills the worker's task).
    # Compose's Bake build ignores `provenance: false` in the compose file, hence the variable.
    makefile = (ROOT / "Makefile").read_text()
    recipes = [line.strip() for line in makefile.splitlines() if line.startswith("\t")]
    building = [r for r in recipes if "docker compose" in r and re.search(r"--build|\bbuild\b", r)]
    assert len(building) == 2  # make up and make up-scheduler
    for line in building:
        assert line.startswith("BUILDX_NO_DEFAULT_ATTESTATIONS=1 docker compose "), line


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


def test_release_builds_the_frontend_with_the_dockerfile_default_api_origin(
    release: dict[Any, Any],
) -> None:
    # No build-args: the published frontend image proxies /api to http://api:8000.
    job = release["jobs"]["images"]
    build = next(s for s in job["steps"] if str(s.get("uses", "")).startswith("docker/build-push"))
    assert "build-args" not in build["with"]
    assert "API_ORIGIN" not in (ROOT / ".github/workflows/release.yml").read_text()


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


# --- frontend/Dockerfile --------------------------------------------------------------------


def _instructions(path: str) -> list[str]:
    """The Dockerfile's instructions, one per entry, comments and blank lines left out."""
    lines = (ROOT / path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def test_the_frontend_build_sets_api_origin_before_next_builds() -> None:
    # next.config.ts reads API_ORIGIN when Next builds; an ENV set after `pnpm build` (or at
    # container start) would not reach the /api proxy.
    lines = _instructions("frontend/Dockerfile")
    build_at = lines.index("RUN pnpm build")
    stage_at = max(i for i, line in enumerate(lines[:build_at]) if line.startswith("FROM "))
    stage = lines[stage_at:build_at]
    assert "ARG API_ORIGIN=http://api:8000" in stage
    assert "ENV API_ORIGIN=$API_ORIGIN" in stage
    assert stage.index("ARG API_ORIGIN=http://api:8000") < stage.index("ENV API_ORIGIN=$API_ORIGIN")
    assert "ENV NEXT_TELEMETRY_DISABLED=1" in stage  # the build sends no usage data


def test_every_dockerfile_base_image_is_pinned_by_digest() -> None:
    for path in ("frontend/Dockerfile", "backend/Dockerfile"):
        lines = _instructions(path)
        froms = [line.split() for line in lines if line.startswith("FROM ")]
        stages = {words[3] for words in froms if len(words) == 4 and words[2].upper() == "AS"}
        copied = [m for line in lines for m in re.findall(r"^COPY --from=(\S+)", line)]
        images = [words[1] for words in froms] + [m for m in copied if m not in stages]
        assert images, path
        for image in images:
            assert re.fullmatch(r"[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}", image), (path, image)
    node = [
        line.split()[1] for line in _instructions("frontend/Dockerfile") if line.startswith("FROM ")
    ]
    assert len(node) == 2 and node[0] == node[1]  # both stages use the same Node
    assert node[0].startswith("node:24-slim@sha256:")


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

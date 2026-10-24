""".github/workflows/eval.yml, checked by parsing it: GitHub Actions cannot run here, so
these tests pin what the workflow must do (triggers, who may run it, where the secrets
go, the cap, the comment, the artifact) and that every command it calls exists.

PyYAML is not a direct dependency: it is installed with huggingface-hub (FastEmbed) and
pymupdf-layout, and GitHub's YAML is plain YAML 1.1 apart from `on`, which PyYAML reads
as the boolean True."""

import importlib
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "eval.yml"


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    data = yaml.safe_load(WORKFLOW.read_text())
    assert isinstance(data, dict)
    return data


@pytest.fixture(scope="module")
def job(workflow: dict[Any, Any]) -> dict[str, Any]:
    jobs = workflow["jobs"]
    assert list(jobs) == ["eval"]
    return dict(jobs["eval"])


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job["steps"])


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in _steps(job) if s.get("name") == name)


def test_it_runs_on_labelled_pull_requests_and_by_hand_only(
    workflow: dict[Any, Any], job: dict[str, Any]
) -> None:
    triggers = workflow[True]  # `on:` parses as True
    assert set(triggers) == {"pull_request", "workflow_dispatch"}
    assert {"labeled", "synchronize"} <= set(triggers["pull_request"]["types"])
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["suite"]["options"] == ["full", "smoke"] and inputs["suite"]["default"] == "full"
    assert float(inputs["max_usd"]["default"]) > 0
    condition = " ".join(job["if"].split())
    assert "contains(github.event.pull_request.labels.*.name, 'retrieval')" in condition
    assert "head.repo.full_name == github.repository" in condition  # never from a fork
    assert "github.event.label.name == 'retrieval'" in condition  # other labels: no new run
    assert "github.event_name == 'workflow_dispatch'" in condition


def test_permissions_are_the_least_it_needs(workflow: dict[Any, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_only_a_run_that_starts_can_cancel_an_older_one(
    workflow: dict[Any, Any], job: dict[str, Any]
) -> None:
    """Concurrency on the workflow would let any pull request event (another label) queue
    a run in the group and cancel the paid one, then skip its own job. On the job, only a
    job that passed its condition joins the group."""
    assert "concurrency" not in workflow
    assert job["concurrency"]["cancel-in-progress"] is True
    assert "pull_request.number" in job["concurrency"]["group"]


def test_smoke_on_pull_requests_full_by_hand_and_always_a_cap(job: dict[str, Any]) -> None:
    env = job["env"]
    assert "'smoke'" in env["SUITE"] and "inputs.suite" in env["SUITE"]
    # The smoke cap covers 15 questions at the guard's worst case (about $0.05 each).
    smoke_cap = float(re.findall(r"'(\d+\.\d+)'", env["MAX_USD"])[0])
    assert smoke_cap >= 15 * 0.05 and "inputs.max_usd" in env["MAX_USD"]
    run = _step(job, "Run the evaluation (paid, capped)")
    command = run["run"]
    assert 'python -m eval.suite "$SUITE" --max-usd "$MAX_USD"' in command
    assert '--report-dir "$REPORT_DIR"' in command
    assert run["env"] == {"OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}"}


def test_the_key_reaches_only_the_paid_step(job: dict[str, Any]) -> None:
    """The key's value is handed to one step; elsewhere only whether it is set (a boolean)."""
    assert "OPENAI_API_KEY" not in str(job.get("env", {}))
    for step in _steps(job):
        for value in (step.get("env") or {}).values():
            if "secrets.OPENAI_API_KEY" not in str(value):
                continue
            if step.get("name") == "Run the evaluation (paid, capped)":
                assert value == "${{ secrets.OPENAI_API_KEY }}"
            else:
                assert value == "${{ secrets.OPENAI_API_KEY != '' }}"


def test_the_corpus_is_verified_restored_and_checked_before_the_paid_step(
    job: dict[str, Any],
) -> None:
    names = [s.get("name") or s.get("uses") for s in _steps(job)]
    order = [
        "Check the secrets are set",
        "Download and verify the corpus bundle",
        "Restore the corpus into the service containers",
        "Check the restored corpus against its manifest",
        "Run the evaluation (paid, capped)",
    ]
    assert [n for n in names if n in order] == order
    download = _step(job, "Download and verify the corpus bundle")["run"]
    assert "sha256sum -c -" in download
    restore = _step(job, "Restore the corpus into the service containers")["run"]
    assert "alembic upgrade head" in restore and "pg_restore" in restore
    assert "snapshots/upload" in restore
    assert (
        "eval.corpus check" in _step(job, "Check the restored corpus against its manifest")["run"]
    )
    assert job["services"]["postgres"]["image"] == "postgres:16"
    assert job["services"]["qdrant"]["image"] == "qdrant/qdrant:v1.19.1"


def test_the_results_are_uploaded_and_commented_even_after_a_stop(job: dict[str, Any]) -> None:
    upload = next(
        s for s in _steps(job) if str(s.get("uses", "")).startswith("actions/upload-artifact")
    )
    assert upload["if"] == "always()" and upload["uses"].endswith("@v4")
    comment = _step(job, "Comment the metrics on the pull request")
    assert comment["if"].startswith("always() && github.event_name == 'pull_request'")
    assert "gh pr comment" in comment["run"] and "--body-file" in comment["run"]
    assert comment["env"]["GH_TOKEN"] == "${{ github.token }}"
    fail = _step(job, "Fail when the evaluation did not complete")
    assert "steps.eval.outputs.code != '0'" in fail["if"]


def test_every_action_is_pinned_and_every_module_called_exists(job: dict[str, Any]) -> None:
    for step in _steps(job):
        if "uses" in step:
            assert re.fullmatch(r"[\w-]+/[\w-]+@v\d+", step["uses"]), step["uses"]
    text = WORKFLOW.read_text()
    for module in set(re.findall(r"python -m ([\w.]+)", text)):
        importlib.import_module(module)  # eval.suite, eval.corpus
    report = Path(__file__).resolve().parents[2] / "eval" / "suite.py"
    assert "report.md" in report.read_text()  # the file the comment step posts

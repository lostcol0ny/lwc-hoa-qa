"""Guard the independent secret scan and workflow security settings."""

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).parents[1] / ".github/workflows").glob("*.yml"))


def assert_sha_pin(uses: str) -> None:
    if not uses.startswith(("./", "docker://")):
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda path: path.name)
def test_ci_security_contract(path: Path) -> None:
    # PyYAML parses the YAML 1.1 key on: as boolean True.
    workflow = yaml.safe_load(path.read_text())
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"]["UV_LOCKED"] == "1"
    jobs = workflow["jobs"]
    if path.name == "ci.yml":
        secrets = jobs["secrets"]
        assert "needs" not in secrets
        assert "if" not in secrets
        assert secrets["permissions"] == {"contents": "read", "pull-requests": "read"}
        scans = [
            step
            for step in secrets.get("steps", [])
            if step.get("uses", "").startswith("gitleaks/gitleaks-action@")
        ]
        assert len(scans) == 1
        assert scans[0]["env"]["GITLEAKS_VERSION"] == "8.24.3"
    for name, job in jobs.items():
        if "uses" in job:
            assert_sha_pin(job["uses"])
        for step in job.get("steps", []):
            if "uses" in step:
                assert_sha_pin(step["uses"])
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False
                if name == "secrets":
                    assert step["with"]["fetch-depth"] == 0


def load_workflow(name: str) -> dict:
    return yaml.safe_load(
        (Path(__file__).parents[1] / ".github/workflows" / name).read_text()
    )


def test_build_corpus_llm_cleanup_is_opt_in() -> None:
    workflow = load_workflow("build-corpus.yml")
    dispatch = workflow[True]["workflow_dispatch"]
    assert dispatch["inputs"]["llm_cleanup"] == {
        "description": dispatch["inputs"]["llm_cleanup"]["description"],
        "type": "boolean",
        "default": False,
    }
    (step,) = [
        s
        for s in workflow["jobs"]["build"]["steps"]
        if "hoa_qa.ingest build" in s.get("run", "")
    ]
    # The key reaches the step only on a manual run that asked for cleanup.
    key = step["env"]["ANTHROPIC_API_KEY"]
    assert "workflow_dispatch" in key and "inputs.llm_cleanup" in key
    assert "workflow_dispatch" in step["env"]["LLM_CLEANUP"]
    # Every other path builds with --no-llm.
    assert step["run"].count("build --out build/ --no-llm") == 1
    assert '[ "$LLM_CLEANUP" = true ] && [ -n "$ANTHROPIC_API_KEY" ]' in step["run"]


def test_eval_workflow_is_manual_and_skips_without_secrets() -> None:
    workflow = load_workflow("eval.yml")
    assert set(workflow[True]) == {"workflow_dispatch"}  # never on push or PRs
    job = workflow["jobs"]["eval"]
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    steps = job["steps"]
    check = steps[0]
    assert check["id"] == "secrets"
    assert set(check["env"]) == {"TYPESAFE_API_KEY", "ANTHROPIC_API_KEY"}
    assert "GITHUB_STEP_SUMMARY" in check["run"]
    # Every later step is gated on the secrets check.
    for step in steps[1:]:
        assert "steps.secrets.outputs.enabled == 'true'" in step["if"]
    (run,) = [s for s in steps if "hoa-qa eval" in s.get("run", "")]
    assert "--json eval-results.json" in run["run"]
    assert "${{" not in run["run"]  # inputs reach the shell via env only
    assert run["env"]["TYPESAFE_API_KEY"] == "${{ secrets.TYPESAFE_API_KEY }}"
    (build,) = [s for s in steps if "hoa_qa.ingest build" in s.get("run", "")]
    assert "--no-llm" in build["run"]
    uploads = [s for s in steps if s.get("uses", "").startswith("actions/upload-")]
    assert uploads and "eval-results.json" in uploads[0]["with"]["path"]


def test_deploy_corpus_trigger_and_production_security() -> None:
    workflow = load_workflow("deploy.yml")
    assert workflow[True]["workflow_run"] == {
        "workflows": ["Build corpus"],
        "types": ["completed"],
        "branches": ["main"],
    }
    assert workflow["concurrency"] == {
        "group": (
            "${{ github.event_name == 'pull_request' && "
            "format('deploy-pull_request-{0}', github.ref) || 'deploy-production' }}"
        ),
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }
    job = workflow["jobs"]["deploy"]
    assert job["if"] == (
        "github.event_name != 'workflow_run' || "
        "github.event.workflow_run.conclusion == 'success'"
    )
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    assert job["env"]["VERCEL_TARGET"] == (
        "${{ github.event_name == 'pull_request' && 'preview' || 'production' }}"
    )
    (checkout,) = [
        s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")
    ]
    assert checkout["with"]["ref"] == (
        "${{ github.event_name == 'pull_request' && github.ref || 'main' }}"
    )
    assert "workflow_run.head_sha" not in str(workflow)
    assert "workflow_run.head_branch" not in str(workflow)
    selection_index = next(
        i for i, s in enumerate(job["steps"]) if s.get("id") == "corpus"
    )
    selection = job["steps"][selection_index]
    assert selection["env"]["TRIGGER_RUN_ID"] == "${{ github.event.workflow_run.id }}"
    assert "--branch main --status success" in selection["run"]
    for step in job["steps"][selection_index + 1 :]:
        assert step["if"] == "steps.corpus.outputs.run_id != ''"
    (download,) = [s for s in job["steps"] if "gh run download" in s.get("run", "")]
    assert download["env"]["run_id"] == "${{ steps.corpus.outputs.run_id }}"
    assert 'gh run download "$run_id"' in download["run"]


@pytest.mark.parametrize(
    ("event", "trigger", "latest", "expected"),
    [
        ("workflow_run", "20", "20", "run_id=20\n"),
        ("workflow_run", "19", "20", ""),
        ("workflow_run", "21", "20", "run_id=21\n"),
        ("push", "", "20", "run_id=20\n"),
        ("pull_request", "", "20", "run_id=20\n"),
    ],
)
def test_deploy_corpus_selection(
    tmp_path: Path, event: str, trigger: str, latest: str, expected: str
) -> None:
    import os
    import subprocess

    steps = load_workflow("deploy.yml")["jobs"]["deploy"]["steps"]
    selection = next(s for s in steps if s.get("id") == "corpus")
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_LATEST"\n')
    gh.chmod(0o755)
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    output.touch()
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", selection["run"]],
        check=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REPOSITORY": "test/repo",
            "TRIGGER_RUN_ID": trigger,
            "TEST_LATEST": latest,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
    )
    assert output.read_text() == expected
    if not expected:
        assert "Deploy skipped" in summary.read_text()

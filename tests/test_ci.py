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

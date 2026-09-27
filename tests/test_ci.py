"""Guard the independent secret scan and workflow security settings."""

import re
from pathlib import Path

import yaml


def test_ci_security_contract() -> None:
    workflow = yaml.safe_load(
        (Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text()
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["env"]["UV_LOCKED"] == "1"
    jobs = workflow["jobs"]
    assert "needs" not in jobs["secrets"]
    assert "if" not in jobs["secrets"]
    assert any(
        "gitleaks/gitleaks-action@" in step.get("uses", "")
        for step in jobs["secrets"]["steps"]
    )
    for job in jobs.values():
        for step in job["steps"]:
            if "uses" in step:
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", step["uses"])
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] is False
                assert step["with"]["fetch-depth"] == 0

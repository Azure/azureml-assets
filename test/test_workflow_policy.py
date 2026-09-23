# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test discovery, dependency enforcement and trusted-base workflow validation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "workflow_policy", ROOT / "scripts" / "validation" / "workflow_policy.py"
)
policy_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy_check)


@pytest.fixture
def repository():
    """Return independent copies of the current workflow inventory and policy."""
    workflows = policy_check.read_workflows(ROOT)
    policy = json.loads((ROOT / ".github" / "workflow-policy.json").read_text(encoding="utf-8"))
    return workflows, policy


def workflow_bytes(document):
    """Serialize a test workflow without sharing mutable fixture objects."""
    return yaml.safe_dump(document, sort_keys=False).encode("utf-8")


def new_workflow():
    """Create a previously unknown workflow following the documented template."""
    return {
        "on": {"pull_request": {"branches": ["main"]}, "workflow_dispatch": None},
        "jobs": {
            policy_check.GATE_JOB: {"uses": policy_check.GATE_USES},
            "build": {"needs": policy_check.GATE_JOB, "runs-on": "ubuntu-latest",
                      "steps": [{"run": "echo build"}]},
        },
    }


def validate_new(repository, document, name="new-workflow.yaml"):
    """Validate an added workflow alongside the existing repository inventory."""
    workflows, policy = repository
    workflows[name] = workflow_bytes(document)
    return policy_check.validate_workflows(workflows, policy)


def test_repository_policy(repository):
    """Validate every current workflow without a hard-coded caller list."""
    workflows, policy = repository
    assert policy_check.validate_workflows(workflows, policy) == []


@pytest.mark.parametrize("suffix", [".yml", ".yaml"])
def test_new_gated_workflow_is_automatically_discovered(repository, suffix):
    """Accept new workflows that use the shared gate without editing an inventory."""
    assert validate_new(repository, new_workflow(), f"brand-new{suffix}") == []


@pytest.mark.parametrize("event", [
    "pull_request", "pull_request_target", "workflow_dispatch", "workflow_call", "push",
])
def test_new_ungated_workflow_fails(repository, event):
    """Reject unknown workflows rather than scanning only existing gate callers."""
    document = {"on": event, "jobs": {"build": {"runs-on": "ubuntu-latest", "steps": [{"run": "echo hi"}]}}}
    assert validate_new(repository, document)


@pytest.mark.parametrize("override", [
    {"uses": "attacker/repo/.github/workflows/check-execution-context.yaml@main"},
    {"uses": "./.github/workflows/check-execution-context.yaml"},
    {"if": "false"},
    {"continue-on-error": "true"},
    {"strategy": {"matrix": {"skip": ["true"]}}},
    {"needs": "build"},
    {"secrets": "inherit"},
])
def test_gate_cannot_be_replaced_or_overridden(repository, override):
    """Require the exact trusted gate call without optional execution."""
    document = new_workflow()
    document["jobs"][policy_check.GATE_JOB].update(override)
    assert validate_new(repository, document)


@pytest.mark.parametrize("kind", ["ungated-root", "unknown-dependency", "cycle"])
def test_dependency_graph_must_be_gated_and_acyclic(repository, kind):
    """Reject independent work, undefined dependencies and cycles."""
    document = new_workflow()
    if kind == "ungated-root":
        del document["jobs"]["build"]["needs"]
    elif kind == "unknown-dependency":
        document["jobs"]["build"]["needs"] = "missing"
    else:
        document["jobs"]["build"]["needs"] = [policy_check.GATE_JOB, "report"]
        document["jobs"]["report"] = {"needs": "build", "runs-on": "ubuntu-latest", "steps": []}
    assert validate_new(repository, document)


def test_transitive_dependency_is_supported(repository):
    """Allow ordinary build/test chains without repeating the gate dependency."""
    document = new_workflow()
    document["jobs"]["test"] = {"needs": "build", "runs-on": "ubuntu-latest", "steps": [{"run": "echo test"}]}
    assert validate_new(repository, document) == []


@pytest.mark.parametrize("expression", [
    "always()", "!cancelled()", "failure()", "success() || true",
    "${{ always() || needs.check-execution-context.result == 'success' }}",
])
def test_status_functions_cannot_bypass_failure(repository, expression):
    """Reject status expressions that override GitHub's default success guard."""
    document = new_workflow()
    document["jobs"]["build"]["if"] = expression
    assert validate_new(repository, document)


def test_report_can_run_after_tests_fail_but_not_after_gate_failure(repository):
    """Accept only the explicit gate-approved reporting condition."""
    document = new_workflow()
    document["jobs"]["report"] = {
        "needs": [policy_check.GATE_JOB, "build"], "runs-on": "ubuntu-latest",
        "if": "${{ " + policy_check.REPORT_CONDITION + " }}", "steps": [{"run": "echo report"}],
    }
    assert validate_new(repository, document) == []
    document["jobs"]["report"]["needs"] = ["build"]
    assert validate_new(repository, document)


@pytest.mark.parametrize("name", [
    policy_check.GATE_FILE, policy_check.POLICY_FILE, policy_check.POLICY_REUSABLE,
])
def test_required_policy_workflows_cannot_be_deleted(repository, name):
    """Prevent deleting the current gate or its future enforcement entry point."""
    workflows, policy = repository
    del workflows[name]
    assert policy_check.validate_workflows(workflows, policy)


@pytest.mark.parametrize("change", ["label", "exit-zero", "shell", "output", "continue-on-error"])
def test_shared_gate_cannot_be_weakened(repository, change):
    """Reject restoration of label exceptions and failure/output overrides."""
    workflows, policy = repository
    document = policy_check.parse_workflow(workflows[policy_check.GATE_FILE])
    job = document["jobs"][policy_check.GATE_JOB]
    if change == "label":
        job["steps"][0]["if"] += " && !contains(github.event.pull_request.labels.*.name, 'safe to test')"
    elif change == "exit-zero":
        job["steps"][0]["run"] = job["steps"][0]["run"].replace("exit 1", "exit 0")
    elif change == "shell":
        job["defaults"] = {"run": {"shell": "bash {0}; exit 0"}}
    elif change == "output":
        job["outputs"]["continue"] = "true"
    else:
        job["continue-on-error"] = "true"
    workflows[policy_check.GATE_FILE] = workflow_bytes(document)
    assert policy_check.validate_workflows(workflows, policy)


@pytest.mark.parametrize("name,event", [
    ("scripts-syntax.yaml", "pull_request_target"),
    ("assets-release.yaml", "pull_request"),
    ("check-changed-files.yaml", "push"),
])
def test_exception_does_not_permit_new_trigger_types(repository, name, event):
    """Keep existing exception scopes from becoming privileged PR entry points."""
    workflows, policy = repository
    document = policy_check.parse_workflow(workflows[name])
    document["on"][event] = None
    workflows[name] = workflow_bytes(document)
    assert policy_check.validate_workflows(workflows, policy)


@pytest.mark.parametrize("content", [
    b"on: pull_request\non: pull_request_target\njobs: {}",
    b"on: [pull_request\n", b"- not\n- a\n- workflow", b"\xff",
    b"on: pull_request\njobs: {<<: {build: {}}}",
])
def test_malformed_workflows_fail_closed(repository, content):
    """Report invalid YAML, duplicate keys, merge keys and invalid encodings."""
    workflows, policy = repository
    workflows["invalid.yaml"] = content
    assert policy_check.validate_workflows(workflows, policy)


@pytest.mark.parametrize("change", ["path-filter", "optional-job", "secrets"])
def test_entrypoint_cannot_be_filtered_or_overridden(repository, change):
    """Keep the independent policy check mandatory and read-only."""
    workflows, policy = repository
    document = policy_check.parse_workflow(workflows[policy_check.POLICY_FILE])
    if change == "path-filter":
        document["on"]["pull_request_target"]["paths"] = [".github/**"]
    elif change == "optional-job":
        document["jobs"]["workflow-policy"]["if"] = "false"
    else:
        document["jobs"]["workflow-policy"]["secrets"] = "inherit"
    workflows[policy_check.POLICY_FILE] = workflow_bytes(document)
    assert policy_check.validate_workflows(workflows, policy)


@pytest.mark.parametrize("change", ["head-checkout", "credentials", "candidate-code", "permissions", "shell"])
def test_reusable_validator_keeps_candidate_code_as_data(repository, change):
    """Reject PR checkout, retained credentials and changes to trusted commands."""
    workflows, policy = repository
    document = policy_check.parse_workflow(workflows[policy_check.POLICY_REUSABLE])
    job = document["jobs"]["validate"]
    if change == "head-checkout":
        job["steps"][0]["with"]["ref"] = "${{ github.event.pull_request.head.sha }}"
    elif change == "credentials":
        job["steps"][0]["with"]["persist-credentials"] = "true"
    elif change == "candidate-code":
        job["steps"][-1]["run"] = "git checkout \"$PR_HEAD_SHA\"\npython scripts/validation/workflow_policy.py"
    elif change == "permissions":
        job["permissions"] = {"contents": "write"}
    else:
        document["defaults"] = {"run": {"shell": "bash {0}; exit 0"}}
    workflows[policy_check.POLICY_REUSABLE] = workflow_bytes(document)
    assert policy_check.validate_workflows(workflows, policy)


def test_ref_reader_reads_blobs_without_checkout(monkeypatch, tmp_path):
    """Inspect immutable Git objects without materializing candidate files."""
    sha, oid = "1" * 40, "2" * 40
    content = workflow_bytes(new_workflow())
    responses = {
        ("ls-tree", "-z", f"{sha}:.github/workflows"): f"100644 blob {oid}\tnew.yml\0".encode(),
        ("cat-file", "-s", oid): str(len(content)).encode(),
        ("cat-file", "blob", oid): content,
    }
    calls = []

    def fake_git(repo, *args):
        calls.append(args)
        assert repo == tmp_path
        return responses[args]

    monkeypatch.setattr(policy_check, "git_output", fake_git)
    assert policy_check.read_workflows(tmp_path, sha) == {"new.yml": content}
    assert calls == list(responses)
    assert list(tmp_path.iterdir()) == []


def test_ref_reader_rejects_symlinks(monkeypatch, tmp_path):
    """Never follow a candidate workflow symlink into other repository content."""
    monkeypatch.setattr(policy_check, "git_output",
                        lambda *args: f"120000 blob {'2' * 40}\tlinked.yaml\0".encode())
    with pytest.raises(policy_check.PolicyError, match="regular workflow"):
        policy_check.read_workflows(tmp_path, "1" * 40)


@pytest.mark.parametrize("ref", ["HEAD", "--help", "main", "a" * 39, "g" * 40])
def test_ref_requires_immutable_commit_sha(tmp_path, ref):
    """Reject symbolic refs and option-shaped inputs before invoking Git."""
    with pytest.raises(policy_check.PolicyError, match="full commit SHA"):
        policy_check.read_workflows(tmp_path, ref)


def test_candidate_cannot_add_its_own_exception(repository, monkeypatch, tmp_path, capsys):
    """Load policy from trusted code, not from the candidate checkout argument."""
    workflows, policy = repository
    workflows["ungated.yaml"] = b"on: pull_request\njobs: {build: {runs-on: ubuntu-latest}}"
    trusted, candidate = tmp_path / "trusted", tmp_path / "candidate"
    for directory in (trusted, candidate):
        (directory / ".github").mkdir(parents=True)
    (trusted / ".github" / "workflow-policy.json").write_text(json.dumps(policy), encoding="utf-8")
    forged = copy.deepcopy(policy)
    forged["exceptions"]["ungated.yaml"] = {"events": ["pull_request"], "reason": "Trust me"}
    (candidate / ".github" / "workflow-policy.json").write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(policy_check, "ROOT", trusted)
    monkeypatch.setattr(policy_check, "read_workflows", lambda repo, ref: workflows)
    monkeypatch.setattr("sys.argv", ["workflow_policy.py", "--repo", str(candidate), "--ref", "1" * 40])
    assert policy_check.main() == 1
    assert "ungated.yaml" in capsys.readouterr().out


def test_bad_policy_is_not_treated_as_an_empty_exception_list(repository):
    """Reject malformed trusted policy rather than silently skipping enforcement."""
    workflows, _ = repository
    with pytest.raises(policy_check.PolicyError):
        policy_check.validate_workflows(workflows, {"schema_version": 1, "exceptions": None})


def test_workflow_size_is_bounded():
    """Fail before parsing oversized candidate input."""
    with pytest.raises(policy_check.PolicyError, match="1 MiB"):
        policy_check.parse_workflow(b" " * (policy_check.MAX_WORKFLOW_BYTES + 1))

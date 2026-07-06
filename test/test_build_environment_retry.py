# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test ACR retries during environment builds."""

from importlib import import_module
from subprocess import CompletedProcess


def test_retry_when_acr_registry_is_not_ready(monkeypatch):
    """Retry ACR builds when the registry has not propagated yet."""
    environment_build = import_module("azureml.assets.environment.build")
    calls = []
    sleeps = []
    warnings = []
    responses = [
        CompletedProcess(
            args=["az", "acr", "run"],
            returncode=3,
            stdout=(
                b"ERROR: (ResourceNotFound) The Resource "
                b"'Microsoft.ContainerRegistry/registries/testregistry' "
                b"under resource group 'testrg' was not found."
            ),
        ),
        CompletedProcess(args=["az", "acr", "run"], returncode=0, stdout=b"build complete"),
    ]

    def fake_run(cmd, cwd=None, stdout=None, stderr=None):
        calls.append((cmd, cwd))
        return responses.pop(0)

    monkeypatch.setattr(environment_build, "run", fake_run)
    monkeypatch.setattr(environment_build.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(environment_build.logger, "log_warning", lambda message: warnings.append(message))

    result = environment_build._run_with_acr_retry(["az", "acr", "run"], ".", "test-image", is_acr=True)

    assert result.returncode == 0
    assert len(calls) == 2
    assert sleeps == [environment_build.ACR_THROTTLE_INITIAL_WAIT]
    assert "registry not yet available" in warnings[0]

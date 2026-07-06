# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test ACR retries during environment builds."""

from types import SimpleNamespace
from importlib import import_module
from subprocess import CompletedProcess

import azureml.assets as assets


def test_build_image_retries_when_acr_registry_is_not_ready(monkeypatch, tmp_path):
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

    context_dir = tmp_path / "context"
    context_dir.mkdir()
    (context_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    monkeypatch.setattr(environment_build, "run", fake_run)
    monkeypatch.setattr(environment_build.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(environment_build.logger, "log_warning", lambda message: warnings.append(message))

    asset_config = SimpleNamespace(name="test-image")
    env_config = SimpleNamespace(
        context_dir_with_path=context_dir,
        dockerfile="Dockerfile",
        os=assets.Os.LINUX,
    )
    build_log = tmp_path / "build.log"

    result = environment_build.build_image(
        asset_config=asset_config,
        env_config=env_config,
        image_name="test.azurecr.io/test-image:latest",
        build_log=build_log,
        resource_group="testrg",
        registry="testregistry",
        test_command="python -V",
    )

    assert result[2] == 0
    assert len(calls) == 2
    assert sleeps == [environment_build.ACR_THROTTLE_INITIAL_WAIT]
    assert "registry not yet available" in warnings[0]
    assert build_log.read_text(encoding="utf-8") == "build complete"

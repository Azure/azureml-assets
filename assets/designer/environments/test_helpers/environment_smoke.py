# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Designer environment smoke tests in a connected Azure ML workspace."""

import os
import time
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import AzureCliCredential

TERMINAL_STATUSES = {"Completed", "Failed", "Canceled", "CancelRequested"}


def run_environment_smoke_test(asset_dir: Path) -> None:
    """Build a draft environment and run a minimal command job with it."""
    subscription_id = os.environ["subscription_id"]
    resource_group = os.environ["resource_group"]
    workspace_name = os.environ["workspace"]
    compute = os.environ["cpu_cluster"]
    timeout_minutes = int(os.environ.get("timeout_minutes", "90"))

    ml_client = MLClient(
        AzureCliCredential(),
        subscription_id,
        resource_group,
        workspace_name,
    )

    environment_name = f"gate-{asset_dir.name}".replace(".", "-")
    environment = Environment(
        build=BuildContext(path=asset_dir / "context"),
        name=environment_name,
        description=f"Gate test draft for the {asset_dir.name} environment.",
    )
    draft_environment = ml_client.environments.create_or_update(environment)

    job = command(
        command=(
            "python -c \"import platform; "
            "print(f'Designer environment smoke test: {platform.python_version()}')\""
        ),
        environment=draft_environment,
        compute=compute,
        display_name=f"{asset_dir.name}-gate-smoke-test",
        experiment_name="designer-environment-gate-tests",
    )
    submitted_job = ml_client.jobs.create_or_update(job)

    timeout = time.time() + timeout_minutes * 60
    status = submitted_job.status
    while time.time() <= timeout:
        status = ml_client.jobs.get(submitted_job.name).status
        if status in TERMINAL_STATUSES:
            break
        time.sleep(30)
    else:
        ml_client.jobs.cancel(submitted_job.name)
        raise TimeoutError(
            f"Job {submitted_job.name} exceeded {timeout_minutes} minutes; "
            f"last status was {status}."
        )

    if status != "Completed":
        ml_client.jobs.stream(submitted_job.name)

    assert status == "Completed", f"Job {submitted_job.name} finished with status {status}."

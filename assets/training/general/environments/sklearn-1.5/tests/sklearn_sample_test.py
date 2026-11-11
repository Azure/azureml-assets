# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run the Scikit-learn 1.5 image smoke in both training modes."""

import os
from pathlib import Path
import time
from uuid import uuid4

from azure.ai.ml import command, Input, MLClient
from azure.ai.ml.entities import BuildContext, CommandJobLimits, Environment
from azure.identity import AzureCliCredential
import pytest


BUILD_CONTEXT = Path("..") / "context"
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = int(os.environ.get("timeout_minutes", 45))


@pytest.mark.parametrize("intel_extension", [False, True])
def test_sklearn_1_5(intel_extension, tmp_path):
    """Build and test an explicitly versioned environment without shared latest aliases."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(process_timeout=60), subscription_id, resource_group, workspace_name
    )

    env_name = "sklearn-1-5-test"
    env_docker_context = Environment(
        name=env_name,
        version=uuid4().hex,
        build=BuildContext(path=str(this_dir / BUILD_CONTEXT)),
        description="Scikit-learn 1.5 Python 3.11 MCR image test.",
    )
    environment = ml_client.environments.create_or_update(env_docker_context)

    mode = " --intel-extension True" if intel_extension else ""
    job = command(
        code=str(this_dir / JOB_SOURCE_CODE),
        command="python main.py --diabetes-csv ${{inputs.diabetes}}" + mode,
        inputs={"diabetes": Input(
            type="uri_file",
            path="https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv",
        )},
        environment=environment.id,
        compute=os.environ.get("cpu_cluster"),
        display_name=f"sklearn-1-5-intel-{intel_extension}",
        experiment_name="sklearn-1-5-validation",
        limits=CommandJobLimits(timeout=TIMEOUT_MINUTES * 60),
    )
    returned_job = ml_client.jobs.create_or_update(job)

    timeout = time.monotonic() + TIMEOUT_MINUTES * 60
    while time.monotonic() < timeout:
        job = ml_client.jobs.get(returned_job.name)
        status = job.status
        if status in {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}:
            break
        time.sleep(30)
    else:
        ml_client.jobs.cancel(returned_job.name)
        pytest.fail(f"Job {returned_job.name} exceeded {TIMEOUT_MINUTES} minutes; status={job.status}")
    ml_client.jobs.download(name=returned_job.name, download_path=str(tmp_path), all=True)
    if status != "Completed":
        for log in tmp_path.rglob("std_log*.txt"):
            print(log.read_text(errors="replace"))
    assert status == "Completed"
    logs = [log.read_text(errors="replace") for log in tmp_path.rglob("std_log*.txt")]
    marker = f"SKLEARN_1_5_TRAIN_RELOAD_PASSED intel={intel_extension}"
    assert any(marker in log for log in logs), f"Missing smoke marker: {marker}"

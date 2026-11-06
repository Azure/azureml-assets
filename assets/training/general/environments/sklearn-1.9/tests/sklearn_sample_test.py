# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run the Scikit-learn 1.9 image smoke in both training modes."""

import os
from pathlib import Path
import time
from uuid import uuid4

from azure.ai.ml import command, Input, MLClient
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import AzureCliCredential
import pytest


@pytest.mark.parametrize("intel_extension", [False, True])
def test_sklearn_1_9(intel_extension, tmp_path):
    """Build and test an explicitly versioned environment without shared latest aliases."""
    directory = Path(__file__).parent
    client = MLClient(
        AzureCliCredential(process_timeout=60),
        os.environ["subscription_id"],
        os.environ["resource_group"],
        os.environ["workspace"],
    )
    environment = client.environments.create_or_update(Environment(
        name="sklearn-1-9-test",
        version=uuid4().hex,
        build=BuildContext(path=str(directory.parent / "context")),
        description="Scikit-learn 1.9 Python 3.11 MCR image test.",
    ))
    mode = " --intel-extension" if intel_extension else ""
    job = client.jobs.create_or_update(command(
        code=str(directory / "src"),
        command="python main.py --diabetes-csv ${{inputs.diabetes}}" + mode,
        inputs={"diabetes": Input(
            type="uri_file",
            path="https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv",
        )},
        environment=environment.id,
        compute=os.environ["cpu_cluster"],
        display_name=f"sklearn-1-9-intel-{intel_extension}",
        experiment_name="sklearn-1-9-validation",
    ))
    timeout_minutes = int(os.environ.get("timeout_minutes", "60"))
    deadline = time.monotonic() + timeout_minutes * 60
    while time.monotonic() < deadline:
        job = client.jobs.get(job.name)
        if job.status in {"Completed", "Failed", "Canceled", "Cancelled", "NotResponding"}:
            break
        time.sleep(30)
    else:
        client.jobs.cancel(job.name)
        pytest.fail(f"Job {job.name} exceeded {timeout_minutes} minutes; status={job.status}")
    client.jobs.download(name=job.name, download_path=str(tmp_path), all=True)
    if job.status != "Completed":
        for log in tmp_path.rglob("std_log*.txt"):
            print(log.read_text(errors="replace"))
    assert job.status == "Completed"
    logs = [log.read_text(errors="replace") for log in tmp_path.rglob("std_log*.txt")]
    marker = f"SKLEARN_1_9_TRAIN_RELOAD_PASSED intel={intel_extension}"
    assert any(marker in log for log in logs), f"Missing smoke marker: {marker}"

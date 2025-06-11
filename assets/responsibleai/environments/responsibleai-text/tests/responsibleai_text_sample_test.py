# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the responsibleai text environment."""
import os
import time
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.operations._run_history_constants import JobStatus
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 60)
STD_LOG = Path("artifacts/user_logs/std_log.txt")


def test_responsibleai_text():
    """Tests a sample job using responsibleai text image as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "responsibleai-text"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="ResponsibleAI Text environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="python main.py",
        environment=f"{env_name}@latest",
        compute=os.environ.get("cpu_cluster"),
        display_name="responsibleai-text-example",
        description="A test run of the responsibleai text curated environment",
        experiment_name="responsibleaiTextExperiment"
    )

    returned_job = ml_client.create_or_update(job)
    assert returned_job is not None

    # Poll until final status is reached, or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        current_status = ml_client.jobs.get(returned_job.name).status
        if current_status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(30)  # sleep 30 seconds

    if current_status == JobStatus.FAILED:
        ml_client.jobs.download(returned_job.name)
        if STD_LOG.exists():
            print(f"*** BEGIN {STD_LOG} ***")
            with open(STD_LOG, "r") as f:
                print(f.read(), end="")
            print(f"*** END {STD_LOG} ***")
        else:
            ml_client.jobs.stream(returned_job.name)

    assert current_status == JobStatus.COMPLETED

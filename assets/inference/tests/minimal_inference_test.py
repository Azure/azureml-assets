# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the minimal 20.04 py38 cpu environment."""
import os
import time
from pathlib import Path
from azure.ai.ml import command, MLClient
from azure.ai.ml._restclient.models import JobStatus
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
print(f"BUILD_CONTEXT : {BUILD_CONTEXT}")
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 30)
STD_LOG = Path("artifacts/user_logs/std_log.txt")


def test_minimal_cpu_inference():
    """Tests a sample job using minimal 20.04 py38 cpu as the environment."""
    this_dir = Path(__file__).parent
    print(f"this_dir : {this_dir}")
    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "minimal_cpu_inference"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name="minimal_cpu_inference",
        description="minimal 20.04 py38 cpu inference environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)
    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="python main.py --score ${{inputs.score}}",
        inputs=dict(
            score="valid_score.py",
        ),
        environment=f"{env_name}@latest",
        compute=os.environ.get("cpu_cluster"),
        display_name="minimal-cpu-inference-example",
        description="A test run of the minimal 20.04 py38 cpu inference curated environment",
        experiment_name="minimalCPUInferenceExperiment"
    )

    returned_job = ml_client.create_or_update(job)
    assert returned_job is not None

    # Poll until final status is reached or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        job = ml_client.jobs.get(returned_job.name)
        status = job.status
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(30)  # sleep 30 seconds
    else:
        # Timeout
        ml_client.jobs.cancel(returned_job.name)
        raise Exception(f"Test aborted because the job took longer than {TIMEOUT_MINUTES} minutes. "
                        f"Last status was {status}.")

    if status == JobStatus.FAILED:
        ml_client.jobs.download(returned_job.name)
        if STD_LOG.exists():
            print(f"*** BEGIN {STD_LOG} ***")
            with open(STD_LOG, "r") as f:
                print(f.read(), end="")
            print(f"*** END {STD_LOG} ***")
        else:
            ml_client.jobs.stream(returned_job.name)

    assert status == JobStatus.COMPLETED

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the minimal ubuntu20.04 py38 gpu environment."""
import os
import time
from pathlib import Path
from azure.ai.ml import command, MLClient
from azure.ai.ml._restclient.models import JobStatus
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 30)
STD_LOG = Path("artifacts/user_logs/std_log.txt")


def test_minimal_cpu_inference():
    """Tests a sample job using minimal ubuntu20.04 py38 gpu as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "minimal_gpu_inference"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name="minimal_gpu_inference",
        description="minimal ubuntu20.04 py38 gpu inference environment created from a Docker context.",
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
        compute=os.environ.get("gpu_cluster"),
        display_name="minimal-gpu-inference-example",
        description="A test run of the minimal ubuntu20.04 py38 gpu inference curated environment",
        experiment_name="minimalGPUInferenceExperiment"
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

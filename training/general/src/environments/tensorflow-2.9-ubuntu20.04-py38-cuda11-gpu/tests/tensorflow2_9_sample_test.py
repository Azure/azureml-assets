# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the tensorflow 2.9 environment."""
import os
import polling
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"


def test_tensorflow_2_9():
    """Tests a sample job using Tensorflow 2.9 as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("sub_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "tensorflow2_9"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="Tensorflow 2.9 environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="python main.py",
        environment=f"{env_name}@latest",
        compute=os.environ.get("gpu_cluster"),
        display_name="tensorflow-mnist-example",
        description="A test run of the tensorflow 2_9 curated environment",
        experiment_name="tensorflow29Experiment"
    )

    returned_job = ml_client.create_or_update(job)

    polling.poll(
        lambda: returned_job.status == "Completed",
        timeout=1200,  # 20 minute timeout
        step=30       # poll every 30 seconds
    )

    assert returned_job is not None
    assert returned_job.status == "Completed"

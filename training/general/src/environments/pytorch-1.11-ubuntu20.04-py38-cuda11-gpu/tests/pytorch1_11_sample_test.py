# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the pytorch 1.11 environment."""
import os
import polling2
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential
from polling2 import TimeoutException

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"


def test_pytorch_1_11():
    """Tests a sample job using pytorch 1.11 as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("sub_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "pytorch1_11"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="Pytorch 1.11 environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="pip install -r requirements.txt && python main.py --iris-csv ${{inputs.iris_csv}} "
                "--epochs ${{inputs.epochs}} --lr ${{inputs.lr}}",
        inputs={
            "iris_csv": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
            ),
            "epochs": 10,
            "lr": 0.1,
        },
        environment=f"{env_name}@latest",
        compute=os.environ.get("gpu_cluster"),
        display_name="pytorch-iris-example",
        description="Train a neural network with PyTorch on the Iris dataset.",
        experiment_name="pytorch111Experiment"
    )

    returned_job = ml_client.create_or_update(job)

    print("job just created and status is " + returned_job.status)
    print("the type of the job is " + str(type(job)))
    print("the type of the returned job is " + str(type(returned_job)))

    try:
        polling2.poll(
            lambda: (ml_client.jobs.get(returned_job.name).status == "Completed" 
            or ml_client.jobs.get(returned_job.name).status == "Failed"),
            step=30,       # poll every 30 seconds
            timeout=1200  # 20 minute timeout
        )
    except TimeoutException:
        print("in except")
        print("status of returned job is " + returned_job.status)

    assert returned_job is not None
    assert returned_job.status == "Completed"

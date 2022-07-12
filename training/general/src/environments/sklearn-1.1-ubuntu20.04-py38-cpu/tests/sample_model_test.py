# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential

# from azureml.core import Environment, Workspace

# DOCKERFILE = Path("../context/Dockerfile")
# CONDA_SPEC = Path("../context/conda_dependencies.yaml")
BUILD_CONTEXT = Path("../context")


def test_sklearn_1_1():
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("sub_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        InteractiveBrowserCredential(), subscription_id, resource_group, workspace_name
    )

    # env = Environment.from_dockerfile("sklearn1_1", this_dir / DOCKERFILE, this_dir / CONDA_SPEC)
    # ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)
    # env = env.register(ws)

    env_name = "sklearn1_1"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name="sklearn1_1",
        description="Sklearn 1.1 environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    # create the command
    job = command(
        code="./src",  # local path where the code is stored
        command="python main.py --diabetes-csv ${{inputs.diabetes}}",
        inputs={
            "diabetes": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv",
            )
        },
        environment=env_name,
        compute=os.environ.get("cpu_cluster"),
        display_name="sklearn-diabetes-example",
        # description,
        # experiment_name
    )

    returned_job = ml_client.create_or_update(job)

    assert returned_job is not None

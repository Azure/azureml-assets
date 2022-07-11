# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml import command, Input
from azure.identity import DefaultAzureCredential

from azureml.core import Environment, Workspace

DOCKERFILE = Path("../context/Dockerfile")
CONDA_SPEC = Path("../context/conda_dependencies.yaml")


def test_sklearn_1_1():
    subscription_id = os.environ.get("sub_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace_name
    )

    env = Environment.from_dockerfile("sklearn1_1", DOCKERFILE, CONDA_SPEC)
    ws = Workspace.get(workspace_name, subscription_id, resource_group)
    env = env.register(ws)

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
        environment=env,
        compute=os.environ.get("cpu_cluster"),
        display_name="sklearn-diabetes-example",
        # description,
        # experiment_name
    )

    returned_job = ml_client.create_or_update(job)

    assert returned_job is not None

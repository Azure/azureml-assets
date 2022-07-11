# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import pytest
from azure.ai.ml import MLClient
from azure.ai.ml import command, MpiDistribution
from azure.identity import DefaultAzureCredential

from azureml.core import Environment, Workspace

import azureml.assets as assets

CONTEXT_DIR = Path("../context")
DOCKERFILE = Path("../context/Dockerfile")
CONDA_SPEC = Path("../context/conda_dependencies.yaml")

def test_sklearn_1_1():

    #retrieve default workspace?
    # create environment doing Environment.from_dockerfile(name, dockerfile, conda_specification=None, pip_requirements=None)
    #register env with the workspace
    #build the environment
    #run the sample in the built environment
        #follow the example from the ipynb file
        #add the cpu cluster to the assets-test workflow
        #run the job and validate that it was successful
    subscription_id = os.environ.get("sub_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
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

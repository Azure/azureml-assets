# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests running a sample job in the responsibleai vision environment."""
import os
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 120)
STD_LOG = Path("artifacts/user_logs/std_log.txt")


def test_responsibleai_vision():
    """Tests a sample job using responsibleai vision image as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "responsibleai-vision"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name=env_name,
        description="ResponsibleAI Vision environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

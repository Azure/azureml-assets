# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AutoML component test's conftest."""

import os
import pytest
import logging

from typing import Optional
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from .utils import load_json, make_request


logger = logging.getLogger(__name__)

PIPELINE_DRAFT_ENDPOINT_MASTER = (
    "https://master.api.azureml-test.ms/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/"
    + "pipelinedrafts"
)
PIPELINE_DRAFT_ENDPOINT = (
    "https://ml.azure.com/api/{}/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/pipelinedrafts"
)

UI_SERVICE_ENDPOINT_MASTER = (
    "https://master.api.azureml-test.ms/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/"
    + "pipelinedrafts/{}/run?nodeCompositionMode=None&asyncCall=true"
)
UI_SERVICE_ENDPOINT = (
    "https://ml.azure.com/api/{}/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/"
    + "pipelinedrafts/{}/run?nodeCompositionMode=None&asyncCall=true"
)

PIPELINE_DRAFT_PAYLOAD_PATH = "./automl/tests/test_configs/payload/create_pipeline_draft_payload.json"


def _get_env(key: str) -> Optional[str]:
    return os.getenv(key)


@pytest.fixture
def version_suffix():
    """Return version suffix from env."""
    return _get_env("version_suffix")


@pytest.fixture
def subscription_id():
    """Return subscription id from env."""
    return _get_env("subscription_id")


@pytest.fixture
def resource_group():
    """Return resource group from env."""
    return _get_env("resource_group")


@pytest.fixture
def workspace_name():
    """Return workspace name from env."""
    return _get_env("workspace")


@pytest.fixture
def resource_group_region():
    """Return resource group region from env."""
    return _get_env("RESOURCE_GROUP_REGION")


@pytest.fixture
def workspace_location():
    """Return workspace location from env."""
    return _get_env("WORKSPACE_REGION_TEST") or "centraluseuap"


@pytest.fixture
def workspace_id(subscription_id, resource_group, workspace_name):
    """Return workspace GUID for a workspace."""
    from azure.ai.ml._restclient.v2022_10_01_preview import (
        AzureMachineLearningWorkspaces,
    )

    serviceClient = AzureMachineLearningWorkspaces(
        credential=DefaultAzureCredential(), subscription_id=subscription_id
    )
    workspace = serviceClient.workspaces.get(
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
    workspace_guid = workspace.workspace_id
    return workspace_guid


@pytest.fixture
def registry_name():
    """Return registry name from env."""
    return _get_env("REGISTRY_NAME")


@pytest.fixture
def mlclient(subscription_id, resource_group, workspace_name):
    """Return mlclient object."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


@pytest.fixture
def auth_token() -> str:
    """Load token from env."""
    return _get_env("token")


@pytest.fixture
def http_headers(auth_token):
    """Create a basic http header for a json payload."""
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "content-type": "application/json",
    }
    return headers


@pytest.fixture
def pipeline_draft_endpoint(subscription_id, resource_group, workspace_name, workspace_location):
    """Pipeline draft endpoint."""
    if workspace_location == "centraluseuap":
        return PIPELINE_DRAFT_ENDPOINT_MASTER.format(subscription_id, resource_group, workspace_name)
    return PIPELINE_DRAFT_ENDPOINT.format(workspace_location, subscription_id, resource_group, workspace_name)


@pytest.fixture
def pipeline_draft_id(http_headers, pipeline_draft_endpoint) -> str:
    """Pipeline draft id."""
    post_data = load_json(PIPELINE_DRAFT_PAYLOAD_PATH)
    status, resp = make_request(pipeline_draft_endpoint, "POST", http_headers, post_data)
    logger.info("pipeline_draft_id : {resp}")
    assert status == 200
    return resp


@pytest.fixture
def ui_service_endpoint(
    subscription_id,
    resource_group,
    workspace_name,
    workspace_location,
    pipeline_draft_id,
):
    """Create and return UI service endpoint."""
    if workspace_location == "centraluseuap":
        return UI_SERVICE_ENDPOINT_MASTER.format(subscription_id, resource_group, workspace_name, pipeline_draft_id)
    return UI_SERVICE_ENDPOINT.format(
        workspace_location,
        subscription_id,
        resource_group,
        workspace_name,
        pipeline_draft_id,
    )

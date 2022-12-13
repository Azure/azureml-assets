# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pytest
import logging

from typing import Optional
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from .test_utilities import load_json, make_request


logger = logging.getLogger(__name__)

PIPELINE_DRAFT_ENDPOINT = (
    "https://ml.azure.com/api/{}/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/pipelinedrafts"
)

UI_SERVICE_ENDPOINT = (
    "https://ml.azure.com/api/{}/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/pipelinedrafts/" + \
        "{}/run?nodeCompositionMode=None&asyncCall=true"
)


def get_env(key: str) -> Optional[str]:
    return os.getenv(key)


@pytest.fixture
def version_suffix():
    return get_env("version_suffix")


@pytest.fixture
def subscription_id():
    return get_env("subscription_id")


@pytest.fixture
def resource_group():
    return get_env("resource_group")


@pytest.fixture
def workspace_name():
    return get_env("workspace")


@pytest.fixture
def resource_group_region():
    return get_env("RESOURCE_GROUP_REGION")


@pytest.fixture
def workspace_location():
    return get_env("WORKSPACE_REGION_TEST")


@pytest.fixture
def workspace_id(subscription_id, resource_group, workspace_name):
    from azure.ai.ml._restclient.v2022_10_01_preview import AzureMachineLearningWorkspaces

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
    return get_env("REGISTRY_NAME")


@pytest.fixture
def mlclient(subscription_id, resource_group, workspace_name):
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


@pytest.fixture
def pipeline_draft_endpoint(subscription_id, resource_group, workspace_name, workspace_location):
    return PIPELINE_DRAFT_ENDPOINT.format(workspace_location, subscription_id, resource_group, workspace_name)


@pytest.fixture
def ui_service_endpoint(
    subscription_id,
    resource_group,
    workspace_name,
    workspace_location,
    pipeline_draft_id,
):
    return UI_SERVICE_ENDPOINT.format(
        workspace_location,
        subscription_id,
        resource_group,
        workspace_name,
        pipeline_draft_id,
    )


@pytest.fixture
def auth_token() -> str:
    return get_env("token")


@pytest.fixture
def http_headers(auth_token):
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "content-type": "application/json",
    }
    logger.info(headers)
    return headers


@pytest.fixture
def pipeline_draft_id(http_headers, pipeline_draft_endpoint) -> str:
    pipeline_draft_payload = "training/automl/tests/test_configs/payload/create_pipeline_draft_payload.json"
    post_data = load_json(pipeline_draft_payload)
    status, resp = make_request(pipeline_draft_endpoint, "POST", http_headers, post_data)
    assert status == 200
    return resp

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

AUTH_TOKEN = "TOKEN"
PIPELINE_DRAFT_CANARY_ENDPOINT = "https://ml.azure.com/api/eastus2euap/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/pipelinedrafts"
UI_SERVICE_CANARY_ENDPOINT = "https://ml.azure.com/api/eastus2euap/studioservice/apiv2/subscriptions/{}/resourceGroups/{}/workspaces/{}/pipelinedrafts/{}/run?nodeCompositionMode=None&asyncCall=true"


def get_env(key: str) -> Optional[str]:
    return os.getenv(key)


@pytest.fixture
def subscription_id():
    return "381b38e9-9840-4719-a5a0-61d9585e1e91"


@pytest.fixture
def workspace_name():
    return "ayushmishra-canary-ws"


@pytest.fixture
def workspace_location():
    return "eastuseuap"


@pytest.fixture
def workspace_id(mlclient, workspace_name):
    return "a65b4ed5-7ab5-474d-a670-1067befb8e20" #canary


@pytest.fixture
def resource_group():
    return "ayush_mishra_res01"


@pytest.fixture
def registry_name():
    return "azureml-staging"


@pytest.fixture
def resource_group_region():
    return "eastuseuap"


@pytest.fixture
def mlclient(subscription_id, resource_group, workspace_name):
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


@pytest.fixture
def pipeline_draft_endpoint(subscription_id, resource_group, workspace_name):
    return PIPELINE_DRAFT_CANARY_ENDPOINT.format(
        subscription_id, resource_group, workspace_name
    )

@pytest.fixture
def ui_service_endpoint(subscription_id, resource_group, workspace_name, pipeline_draft_id):
    return UI_SERVICE_CANARY_ENDPOINT.format(
        subscription_id, resource_group, workspace_name, pipeline_draft_id
    )


@pytest.fixture
def auth_token() -> str:
    return get_env(AUTH_TOKEN)


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

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""API key helper functions."""

import json
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
from typing import Tuple, Optional

from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun
from azureml._common._error_definition.azureml_error import AzureMLError

from azureml.acft.common_components.utils.error_handling.exceptions import (
    ACFTSystemException,
    ACFTValidationException,
)
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    ACFTSystemError,
    ACFTUserError,
)

def _create_session_with_retry(retry: int = 3) -> requests.Session:
    """
    Create requests.session with retry.

    :type retry: int
    rtype: Response
    """
    retry_policy = _get_retry_policy(num_retry=retry)

    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_policy))
    session.mount("http://", HTTPAdapter(max_retries=retry_policy))
    return session


def _get_retry_policy(num_retry: int = 3) -> Retry:
    """
    Request retry policy with increasing backoff.

    :return: Returns the msrest or requests REST client retry policy.
    :rtype: urllib3.Retry
    """
    backoff_factor = 0.4
    retry_policy = Retry(
        total=num_retry,
        read=num_retry,
        connect=num_retry,
        backoff_factor=backoff_factor,
        status_forcelist={413, 429, 500, 502, 503, 504, None},
        # By default this is True. We set it to false to get the full error trace, including url and
        # status code of the last retry. Otherwise, the error message is too many 500 error responses',
        # which is not useful.
        raise_on_status=False,
    )
    return retry_policy


def _send_post_request(url: str, headers: dict, payload: dict):
    """Send a POST request."""
    try:
        with _create_session_with_retry() as session:
            response = session.post(url, data=json.dumps(payload), headers=headers)
            # Raise an exception if the response contains an HTTP error status code
            response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        raise ACFTSystemException._with_error(
            AzureMLError.create(ACFTSystemError, error_details=f"HTTP Error: {errh}")
        )
    return response


def get_target_from_connection(connections_name: str) -> Tuple[str, Optional[str]]:
    """
    Get target from connections_name.

    :param connections_name: Name of the connection.
    :return: target.
    """
    run = Run.get_context()
    if isinstance(run, _OfflineRun):
        curr_ws = Workspace.from_config("config.json")
    else:
        curr_ws = run.experiment.workspace

    if hasattr(curr_ws._auth, "get_token"):
        bearer_token = curr_ws._auth.get_token(
            "https://management.azure.com/.default"
        ).token
    else:
        bearer_token = curr_ws._auth.token

    endpoint = curr_ws.service_context._get_endpoint("api")
    url_list = [
        endpoint,
        "rp/workspaces/subscriptions",
        curr_ws.subscription_id,
        "resourcegroups",
        curr_ws.resource_group,
        "providers",
        "Microsoft.MachineLearningServices",
        "workspaces",
        curr_ws.name,
        "connections",
        connections_name,
        "listsecrets?api-version=2023-02-01-preview",
    ]

    resp = _send_post_request(
        "/".join(url_list),
        {"Authorization": f"Bearer {bearer_token}", "content-type": "application/json"},
        {},
    )
    target = resp.json().get("properties", {}).get("target")
    if target is None:
        msg = "Target not found in response"
        raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(msg),
                    )
                )
    return target

def get_api_key_from_connection(connections_name: str) -> Tuple[str, Optional[str]]:
    """
    Get api_key from connections_name.

    :param connections_name: Name of the connection.
    :return: api_key, api_version.
    """
    run = Run.get_context()
    if isinstance(run, _OfflineRun):
        curr_ws = Workspace.from_config()
    else:
        curr_ws = run.experiment.workspace

    if hasattr(curr_ws._auth, "get_token"):
        bearer_token = curr_ws._auth.get_token(
            "https://management.azure.com/.default"
        ).token
    else:
        bearer_token = curr_ws._auth.token

    endpoint = curr_ws.service_context._get_endpoint("api")
    url_list = [
        endpoint,
        "rp/workspaces/subscriptions",
        curr_ws.subscription_id,
        "resourcegroups",
        curr_ws.resource_group,
        "providers",
        "Microsoft.MachineLearningServices",
        "workspaces",
        curr_ws.name,
        "connections",
        connections_name,
        "listsecrets?api-version=2023-02-01-preview",
    ]

    resp = _send_post_request(
        "/".join(url_list),
        {"Authorization": f"Bearer {bearer_token}", "content-type": "application/json"},
        {},
    )

    credentials = resp.json()["properties"]["credentials"]
    metadata = resp.json()["properties"].get("metadata", {})
    if "key" in credentials:
        return credentials["key"], metadata.get("ApiVersion")
    else:
        if "secretAccessKey" not in credentials and "keys" in credentials:
            credentials = credentials["keys"]
        return credentials["secretAccessKey"], None

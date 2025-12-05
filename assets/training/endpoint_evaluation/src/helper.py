# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun


RETRIABLE_STATUS_CODES = {413, 429, 500, 502, 503, 504, None}
# Define loggable metrics - only mean metrics for TPOT, TTFT, E2E latency, and ITL
LOGGABLE_METRIC_NAMES = {
    # Core throughput metrics (lowercase from JSON files)
    "request_throughput",
    "input_throughput",
    "output_throughput",
    # Mean latency metrics only (no median)
    "mean_e2e_latency_ms",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
    "total_input_tokens",
    "total_output_tokens",
    # Uppercase versions for AML logging
    "REQUEST_THROUGHPUT",
    "INPUT_THROUGHPUT",
    "OUTPUT_THROUGHPUT",
    "MEAN_E2E_LATENCY_MS",
    "MEAN_TTFT_MS",
    "MEAN_TPOT_MS",
    "MEAN_ITL_MS",
    "TOTAL_INPUT_TOKENS",
    "TOTAL_OUTPUT_TOKENS",
}


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
        status_forcelist=RETRIABLE_STATUS_CODES,
        # By default this is True. We set it to false to get the full error trace, including url and
        # status code of the last retry. Otherwise, the error message is too many 500 error responses',
        # which is not useful.
        raise_on_status=False,
    )
    return retry_policy


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


def _send_post_request(url: str, headers: dict, payload: dict):
    """Send a POST request."""
    try:
        with _create_session_with_retry() as session:
            response = session.post(url, data=json.dumps(payload), headers=headers)
            # Raise an exception if the response contains an HTTP error status code
            response.raise_for_status()
    except requests.exceptions.HTTPError as errh:
        raise
    return response


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
        bearer_token = curr_ws._auth.get_token("https://management.azure.com/.default").token
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


def _get_azureml_run():
    """Get AzureML Run context if available."""
    try:
        azureml_run = Run.get_context()
        if azureml_run and "OfflineRun" not in azureml_run.id:
            return azureml_run
    except Exception as e:
        print(f"Warning: Failed to get AzureML run context: {e}")
    return None


def log_metrics(metrics: dict):
    """Log metrics to AzureML Run if available."""
    azureml_run = _get_azureml_run()
    if azureml_run:
        for key, value in metrics.items():
            # Check if the key (or key without prefix) is a loggable metric
            # Support both prefixed (base_/target_) and non-prefixed keys
            key_to_check = key
            if key.startswith("base_") or key.startswith("target_"):
                key_to_check = key.split("_", 1)[1]  # Remove prefix

            if key_to_check in LOGGABLE_METRIC_NAMES:
                azureml_run.log(key, value)
        azureml_run.flush()

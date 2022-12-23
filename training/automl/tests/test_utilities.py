# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility file."""

from typing import Any, Dict, List, Tuple
import json
import logging
import time
from urllib3 import HTTPResponse, Retry, poolmanager
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.operations._run_history_constants import JobStatus, RunHistoryConstants


logger = logging.getLogger(name=__file__)


def make_request(uri: str, method: str, headers: Dict, data: Dict) -> Tuple[int, Any]:
    """Make HTTP request. Returns HTTP status code and response body."""
    http = poolmanager.PoolManager(retries=Retry(connect=3))
    if method == "POST":
        body = data
        if headers.get("content-type") == "application/json":
            body = json.dumps(data).encode("utf-8")
        response = http.request(method=method, url=uri, headers=headers, body=body)
        status = response.status
        response_data = response.data
        if headers.get("content-type") == "application/json":
            response_data = response_data.decode("utf-8")
    else:
        response: HTTPResponse = http.request(method=method, url=uri, headers=headers, fields=data)
        status = response.status
        response_data = response.data
    return status, response_data


def load_json(file_path: str) -> Dict:
    """Load JSON and returns loaded dictionary."""
    try:
        with open(file_path) as f:
            json_dict = json.load(f)
        return json_dict
    except Exception as e:
        logger.warning(f"Caught exception in loading json file at {file_path}. Exception => {e}")
        raise e


def validate_successful_run(mlclient: MLClient, run_id: str):
    """Assert that job with run_id is successful."""
    run_id = run_id.strip()
    # sleep for 10s
    time.sleep(10)
    job = mlclient.jobs.get(run_id)
    while job.status not in RunHistoryConstants.TERMINAL_STATUSES:
        job = mlclient.jobs.get(run_id)
        time.sleep(30)
    assert job.status == JobStatus.COMPLETED


def register_data_assets(mlclient: MLClient, data_assets: List) -> List:
    """Register data assets.

    :param mlclient: mlclient object
    :type mlclient: MLClient
    :param data_assets: List of data assets to register
    :type data_assets: List
    :return: registered list of data assets
    :rtype: List
    """
    registered_assets = []
    for asset in data_assets:
        data = Data(
            name=asset.get("name"),
            version=asset.get("version"),
            path=asset.get("path"),
            type=asset.get("type"),
        )
        try:
            asset = mlclient.data.get(name=data.name, version=data.version)
        except Exception:
            logger.warning("asset is not registered, registering...")
            asset = mlclient.data.create_or_update(data)

        registered_assets.append(asset)
    return registered_assets


def _update_payload_with_registered_data_assets(
    data_node_id, asset_details, payload, workspace_id, workspace_location
):
    dataset_nodes = payload["graph"]["datasetNodes"]
    for node in dataset_nodes:
        if node["id"] == data_node_id:
            asset_definition = node["dataSetDefinition"]["value"]["assetDefinition"]
            asset_id = "azureml://locations/{}/workspaces/{}/data/{}/versions/{}"
            asset_id = asset_id.format(
                workspace_location,
                workspace_id,
                asset_details.name,
                asset_details.version,
            )
            asset_definition["assetId"] = asset_id
            asset_definition["path"] = asset_details.path
            asset_definition["type"] = asset_details.type
            break
    return payload


def update_payload_with_registered_data_assets(assets, payload, workspace_id, workspace_location):
    """Update payload with registered data asset values."""
    logger.info("update payload with assets")
    for asset in assets:
        if "training_data" in asset.name:
            payload = _update_payload_with_registered_data_assets(
                data_node_id="training_data",
                asset_details=asset,
                payload=payload,
                workspace_id=workspace_id,
                workspace_location=workspace_location,
            )
        elif "test_data" in asset.name:
            payload = _update_payload_with_registered_data_assets(
                data_node_id="test_data",
                asset_details=asset,
                payload=payload,
                workspace_id=workspace_id,
                workspace_location=workspace_location,
            )
        elif "validation_data" in asset.name:
            payload = _update_payload_with_registered_data_assets(
                data_node_id="validation_data",
                asset_details=asset,
                payload=payload,
                workspace_id=workspace_id,
                workspace_location=workspace_location,
            )
    return payload


def update_payload_module_id(payload, node_id, module_id):
    """Update automl payload with module id."""
    # update in graph
    module_nodes = payload["graph"]["moduleNodes"]
    for node in module_nodes:
        if node["id"] == node_id:
            node["moduleId"] = module_id
            break

    # update in moduleNodeRunSettings
    all_module_run_settings = payload["moduleNodeRunSettings"]
    for module_run_settings in all_module_run_settings:
        if module_run_settings["nodeId"] == node_id:
            module_run_settings["moduleId"] = module_id
            break

    # update in moduleNodeUIInputSettings
    all_node_ui_input_settings = payload["moduleNodeUIInputSettings"]
    for ui_input_settings in all_node_ui_input_settings:
        if ui_input_settings["nodeId"] == node_id:
            ui_input_settings["moduleId"] = module_id
            break

    logger.info("Updated moduleId in payload")
    return payload

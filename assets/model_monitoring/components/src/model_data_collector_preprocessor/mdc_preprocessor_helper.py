# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions for Model Data Collector Preprocessor."""

from typing import Tuple
import os
from urllib.parse import urlparse
from azureml.core.run import Run
from azure.ai.ml import MLClient


def convert_to_azureml_long_form(url_str: str, datastore: str, sub_id=None, rg_name=None, ws_name=None) -> str:
    """Convert path to AzureML path."""
    url = urlparse(url_str)
    if url.scheme in ["https", "http"]:
        idx = url.path.find('/', 1)
        path = url.path[idx+1:]
    elif url.scheme in ["wasbs", "wasb", "abfss", "abfs"]:
        path = url.path[1:]
    elif url.scheme == "azureml" and url.hostname == "datastores":  # azrueml short form
        idx = url.path.find('/paths/')
        path = url.path[idx+7:]
    else:
        return url_str  # azureml long form, azureml asset, file or other scheme, return original path directly

    sub_id = sub_id or os.environ.get("AZUREML_ARM_SUBSCRIPTION", None)
    rg_name = rg_name or os.environ.get("AZUREML_ARM_RESOURCEGROUP", None)
    ws_name = ws_name or os.environ.get("AZUREML_ARM_WORKSPACE_NAME", None)

    return f"azureml://subscriptions/{sub_id}/resourcegroups/{rg_name}/workspaces/{ws_name}/datastores" \
           f"/{datastore}/paths/{path}"


def get_datastore_from_input_path(input_path: str, ml_client=None) -> str:
    url = urlparse(input_path)
    if url.scheme == "azureml":
        if ':' in url.path:  # azureml asset path
            return _get_datastore_from_asset_path(input_path, ml_client)
        else:  # azureml long or short form
            return _get_datastore_from_azureml_path(input_path)
    else:
        # todo: raise ModelMonitoringException
        return "workspaceblobstore"


def _get_workspace_info() -> Tuple[str, str, str]:
    """Get workspace info from environment variables."""
    ws = Run.get_context().experiment.workspace
    sub_id = ws.subscription_id or os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg_name = ws.resource_group or os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws_name = ws.name or os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    return sub_id, rg_name, ws_name


def _get_datastore_from_azureml_path(azureml_path: str) -> str:
    start_idx = azureml_path.find('/datastores/')
    end_idx = azureml_path.find('/paths/')
    return azureml_path[start_idx+12:end_idx]


def _get_datastore_from_asset_path(asset_path: str, ml_client=None) -> str:
    if not ml_client:
        sub_id, rg_name, ws_name = _get_workspace_info()
        ml_client = MLClient(subscription_id=sub_id, resource_group=rg_name, workspace_name=ws_name)

    # todo: validation
    asset_sections = asset_path.split(':')
    asset_name = asset_sections[1]
    asset_version = asset_sections[2]

    data_asset = ml_client.data.get(asset_name, asset_version)
    return data_asset.datastore or get_datastore_from_input_path(data_asset.path)

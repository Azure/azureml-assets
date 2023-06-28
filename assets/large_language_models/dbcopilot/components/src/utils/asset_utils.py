"""Asset utils."""
import logging
import re
from typing import Tuple

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azureml.core import Workspace


def parse_registry_asset_path(asset_type: str, path: str) -> Tuple[str, str, str]:
    """parse_registry_asset_path."""
    pattern = r"\/registries\/(?P<registry>[^\/]+)\/{}\/(?P<asset_name>[^\/]+)\/versions\/(?P<version>[^\/]+)"
    pattern2 = r"\/registries\/(?P<registry>[^\/]+)\/{}\/(?P<asset_name>[^\/]+)\/?"
    match = re.search(pattern.format(asset_type), path)
    if match:
        registry = match.group("registry")
        asset_name = match.group("asset_name")
        version = match.group("version")
        return registry, asset_name, version
    else:
        match = re.search(pattern2.format(asset_type), path)
        if match:
            registry = match.group("registry")
            asset_name = match.group("asset_name")
            version = None
            return registry, asset_name, version
        else:
            logging.info(f"failed to parse env path {path}")
            return None, None, None


def parse_asset_path(asset_type: str, path: str) -> Tuple[str, str]:
    """parse_asset_path."""
    pattern = r"\/{}\/(?P<asset_name>[^\/]+)\/versions\/(?P<version>[^\/]+)".format(
        asset_type
    )
    match = re.search(pattern, path)
    if match:
        asset_name = match.group("asset_name")
        version = match.group("version")
        return asset_name, version
    else:
        logging.info(f"failed to parse asset path {path}")
        raise ValueError(f"Invalid path: {path}")


def parse_data_path(path: str) -> Tuple[str, str]:
    """parse_data_path."""
    pattern = (
        r"\/datastores\/(?P<datastore_name>[^\/]+)\/paths\/(?P<relative_path>.+\/)"
    )
    match = re.search(pattern, path)
    if match:
        datastore_name = match.group("datastore_name")
        relative_path = match.group("relative_path")
        return datastore_name, relative_path
    else:
        raise ValueError(f"Invalid path: {path}")


def parse_data_asset(asset_id: str) -> Tuple[str, str]:
    """parse_data_asset."""
    return parse_asset_path("data", asset_id)


def parse_datastore_uri(uri: str):
    """parse_datastore_uri."""
    # azureml://subscriptions/1b75927d-563b-49d2-bf8a-772f7d6a170e/resourcegroups/RAGDev/workspaces/RAGDev/datastores/test_sql_database
    regexes = [
        r"azureml://subscriptions/(?P<subscription_id>[^/]+)/resourcegroups/(?P<resource_group>[^/]+)/workspaces/(?P<workspace_name>[^/]+)/datastores/(?P<datastore_name>[^/]+)/paths/(?P<relative_path>[^/]+)",
        r"azureml://subscriptions/(?P<subscription_id>[^/]+)/resourcegroups/(?P<resource_group>[^/]+)/workspaces/(?P<workspace_name>[^/]+)/datastores/(?P<datastore_name>[^/]+)",
        r"azureml://datastores/(?P<datastore_name>[^/]+)/paths/(?P<relative_path>[^/]+)",
        r"azureml://datastores/(?P<datastore_name>[^/]+)",
    ]
    for regex in regexes:
        match = re.match(regex, uri)
        if match:
            return match.groupdict()
    return None


def get_datastore_uri(workspace: Workspace, datastore_uri: str):
    """get_datastore_uri."""
    if datastore_uri.startswith("azureml://"):
        parsed_uri = parse_datastore_uri(datastore_uri)
        if parsed_uri is None:
            raise ValueError(f"Unable to parse datastore uri {datastore_uri}")
        if "subscription_id" in parsed_uri:
            datastore_uri = datastore_uri
        else:
            datastore_uri = f"azureml://subscriptions/{workspace.subscription_id}/resourcegroups/{workspace.resource_group}/workspaces/{workspace.name}/datastores/{parsed_uri['datastore_name']}"
            if "relative_path" in parsed_uri:
                datastore_uri = datastore_uri + f"/paths/{parsed_uri['relative_path']}"
    else:
        datastore_uri = f"azureml://subscriptions/{workspace.subscription_id}/resourcegroups/{workspace.resource_group}/workspaces/{workspace.name}/datastores/{datastore_uri}"
    return datastore_uri


def get_full_env_path(credential, path: str) -> str:
    """get_full_env_path."""
    if path.startswith("azureml://"):
        registry, environment_name, version = parse_registry_asset_path(
            "environments", path
        )
        if registry and version is None:
            ml_client = MLClient(credential, registry_name=registry)
            envs = ml_client.environments.list(environment_name)
            envs_sorted = sorted(
                envs, key=lambda x: x.creation_context.created_at, reverse=True
            )
            # Select the latest environment
            latest_env: Environment = envs_sorted[0]
            logging.info(f"Using latest environment {latest_env.id}")
            return latest_env.id
    return path


def parse_connection(connection_uri: str):
    """parse_connection."""
    regex = r"/subscriptions/(?P<subscription_id>[^/]+)/resourceGroups/(?P<resource_group>[^/]+)/providers/Microsoft.MachineLearningServices/workspaces/(?P<workspace_name>[^/]+)/connections/(?P<connection_name>[^/]+)"
    match = re.match(regex, connection_uri)
    if match:
        return match.groupdict()
    else:
        return None

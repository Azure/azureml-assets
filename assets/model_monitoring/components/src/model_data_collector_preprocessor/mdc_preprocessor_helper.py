import os
import re
from typing import Tuple, List
from datetime import datetime, timedelta
from urllib.parse import urlparse
from azureml.core.run import Run
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential, AzureCliCredential


class HdfsPath:
    def __init__(self, scheme, account_name, container_name, path):
        self.scheme = scheme
        self.account_name = account_name
        self.container_name = container_name
        self.path = path

    def __str__(self):
        return f"{self.scheme}://{self.container_name}@{self.account_name}.{store_type}.core.windows.net/{self.path}"


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


def convert_to_hdfs_path(uri_folder_path: str, ml_client=None) -> str:
    """Convert url_path to HDFS path."""
    url = urlparse(uri_folder_path)
    # TODO sovereign endpoint
    if url.scheme in ["https", "http"]:
        pattern = "(?P<scheme>http|https)://(?P<account_name>[^\.]+).(?P<store_type>blob|dfs).core.windows.net/" \
                  "(?P<container>[^/]+)/(?P<path>.+)"
        matches = re.match(pattern, uri_folder_path)
        if not matches:
            raise ValueError(f"Unsupported uri as uri_folder: {uri_folder_path}")
        # use abfss to access both blob and dfs, to workaround the append block issue
        scheme = "abfss" if matches.group("scheme") == "https" else "abfs"
        store_type = "dfs"
        account_name = matches.group("account_name")
        container = matches.group("container")
        path = matches.group("path")
        return f"{scheme}://{container}@{account_name}.{store_type}.core.windows.net/{path}"
    elif url.scheme in ["wasbs", "wasb", "abfss", "abfs"]:
        pattern = "(?P<scheme>wasbs|abfss|wasb|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+).(?P<store_type>blob|dfs).core.windows.net/(?P<path>.+)"  # noqa
        matches = re.match(pattern, uri_folder_path)
        if not matches:
            raise ValueError(f"Unsupported uri as uri_folder: {uri_folder_path}")
        scheme = "abfss" if matches.group("scheme") in ["wasbs", "abfss"] else "abfs"
        store_type = "dfs"
        account_name = matches.group("account_name")
        container = matches.group("container")
        path = matches.group("path")
        return f"{scheme}://{container}@{account_name}.{store_type}.core.windows.net/{path}"
    elif url.scheme == "azureml":
        if ':' in url.path:  # azureml asset path
            return _get_hdfs_path_from_asset_uri(uri_folder_path, ml_client)
        else:  # azureml long or short form
            datastore_name, path = _get_datastore_and_path_from_azureml_path(uri_folder_path)
            if not ml_client:
                sub_id, rg_name, ws_name = _get_workspace_info()
                ml_client = MLClient(DefaultAzureCredential(), sub_id, rg_name, ws_name)
            datastore = ml_client.datastores.get(datastore_name)
            datastore_type = datastore.type.__str__()
            if datastore_type not in ["DatastoreType.AZURE_BLOB", "DatastoreType.AZURE_DATA_LAKE_GEN2"]:
                raise ValueError(f"Only Azure Blob and Azure Data Lake Gen2 are supported, but got {datastore.type}.")
            scheme = "abfss" if datastore.protocol == "https" else "abfs"
            store_type = "dfs"
            account_name = datastore.account_name
            container = datastore.container_name if datastore_type == "DatastoreType.AZURE_BLOB" \
                else datastore.filesystem
            return f"{scheme}://{container}@{account_name}.{store_type}.core.windows.net/{path}"
    else:
        return uri_folder_path  # file or other scheme, return original path directly


def get_datastore_from_input_path(input_path: str, ml_client=None) -> str:
    """Get datastore name from input path."""
    url = urlparse(input_path)
    if url.scheme == "azureml":
        if ':' in url.path:  # azureml asset path
            return _get_datastore_from_asset_uri(input_path, ml_client)
        else:  # azureml long or short form
            datastore, _ = _get_datastore_and_path_from_azureml_path(input_path)
            return datastore
    elif url.scheme == "file" or os.path.isdir(input_path):
        return None  # local path for testing, datastore is not needed
    else:
        raise ValueError("Only azureml path(long, short or asset) is supported as input path of the MDC preprocessor.")


def get_file_list(start_datetime: datetime, end_datetime: datetime, input_data: str) -> List[str]:
    file_list = []
    root_uri_folder = convert_to_hdfs_path(input_data)
    if end_datetime.minute == 0 and end_datetime.second == 0:
        # if end_datetime is a whole hour, the last hour folder is not needed
        end_datetime -= timedelta(seconds=1)
    cur_datetime = start_datetime
    while cur_datetime <= end_datetime:
        cur_folder = get_datetime_folder(root_uri_folder, cur_datetime)
        if folder_exists(cur_folder):
            file_list.append(cur_folder)
        cur_datetime += timedelta(hours=1)
    return file_list


def folder_exists(folder_path: str) -> bool:
    """Check if folder exists."""
    return _folder_exists(folder_path)


def get_datetime_folder(root_uri_folder: str, cur_datetime: datetime) -> str:
    root_uri_folder = root_uri_folder.rstrip('/')
    return f"{root_uri_folder}/{cur_datetime.strftime('%Y/%m/%d/%H')}"


def _get_workspace_info() -> Tuple[str, str, str]:
    """Get workspace info from Run context and environment variables."""
    ws = Run.get_context().experiment.workspace
    sub_id = ws.subscription_id or os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg_name = ws.resource_group or os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws_name = ws.name or os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    return sub_id, rg_name, ws_name


def _get_datastore_and_path_from_azureml_path(azureml_path: str) -> Tuple[str, str]:
    start_idx = azureml_path.find('/datastores/')
    end_idx = azureml_path.find('/paths/')
    return azureml_path[start_idx+12:end_idx], azureml_path[end_idx+7:]


def _get_dataasset_from_asset_uri(asset_uri: str, ml_client=None) -> Data:
    if not ml_client:
        sub_id, rg_name, ws_name = _get_workspace_info()
        ml_client = MLClient(subscription_id=sub_id, resource_group=rg_name, workspace_name=ws_name)

    # todo: validation
    asset_sections = asset_uri.split(':')
    asset_name = asset_sections[1]
    asset_version = asset_sections[2]

    data_asset = ml_client.data.get(asset_name, asset_version)
    return data_asset


def _get_datastore_from_asset_uri(asset_path: str, ml_client=None) -> str:
    data_asset = _get_dataasset_from_asset_uri(asset_path, ml_client)
    return data_asset.datastore or get_datastore_from_input_path(data_asset.path)


def _get_hdfs_path_from_asset_uri(asset_uri: str, ml_client=None) -> str:
    data_asset = _get_dataasset_from_asset_uri(asset_uri, ml_client)
    if data_asset.type != 'uri_folder':
        raise ValueError(f"Only uri_folder type asset is supported, but got {data_asset.type}.")
    return convert_to_hdfs_path(data_asset.path, ml_client)

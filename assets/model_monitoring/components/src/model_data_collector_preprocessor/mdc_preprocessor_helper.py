import os
import re
from typing import Tuple, List
from datetime import datetime, timedelta
from urllib.parse import urlparse
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient
from azure.core.exceptions import ResourceNotFoundError


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


def convert_to_hdfs_path(uri_folder_path: str, ws: Workspace = None) -> str:
    """Convert url_path to HDFS path."""
    url = urlparse(uri_folder_path)
    # TODO sovereign endpoint
    if url.scheme in ["https", "http"]:
        pattern = r"(?P<scheme>http|https)://(?P<account_name>[^\.]+).(?P<store_type>blob|dfs).core.windows.net/" \
                  r"(?P<container>[^/]+)/(?P<path>.+)"
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
        pattern = r"(?P<scheme>wasbs|abfss|wasb|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+).(?P<store_type>blob|dfs).core.windows.net/(?P<path>.+)"  # noqa
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
            # asset path should be translated to azureml or hdfs path in service, should not reach here
            raise ValueError("AzureML asset path is not supported as uri_folder.")
        else:  # azureml long or short form
            datastore_name, path = _get_datastore_and_path_from_azureml_path(uri_folder_path)
            ws = ws or Run.get_context().experiment.workspace
            datastore = ws.datastores.get(datastore_name)
            datastore_type = datastore.datastore_type
            if datastore_type not in ["AzureBlob", "AzureDataLakeGen2"]:
                raise ValueError(f"Only Azure Blob and Azure Data Lake Gen2 are supported, but got {datastore.type}.")
            scheme = "abfss" if datastore.protocol == "https" else "abfs"
            store_type = "dfs"
            account_name = datastore.account_name
            container = datastore.container_name
            return f"{scheme}://{container}@{account_name}.{store_type}.core.windows.net/{path}"
    else:
        return uri_folder_path  # file or other scheme, return original path directly


def get_datastore_from_input_path(input_path: str) -> str:
    """Get datastore name from input path."""
    url = urlparse(input_path)
    if url.scheme == "azureml":
        if ':' in url.path:  # azureml asset path
            raise ValueError("AzureML asset path is not supported as input path.")
        else:  # azureml long or short form
            datastore, _ = _get_datastore_and_path_from_azureml_path(input_path)
            return datastore
    elif url.scheme == "file" or os.path.isdir(input_path):
        return None  # local path for testing, datastore is not needed
    else:
        raise ValueError("Only azureml path(long, short) is supported as input path of the MDC preprocessor.")


def get_file_list(start_datetime: datetime, end_datetime: datetime, input_data: str,
                  root_uri_folder: str = None, service_client: DataLakeServiceClient = None) -> List[str]:
    if _is_local_path(input_data):
        # for lcoal testing
        return _get_local_file_list(start_datetime, end_datetime, input_data)

    file_list = []
    root_uri_folder = root_uri_folder or convert_to_hdfs_path(input_data).rstrip('/')
    # get meta from root_uri_folder
    pattern = r"(?P<scheme>abfss|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+).dfs.core.windows.net(?P<path>$|/.*)"  # noqa
    matches = re.match(pattern, root_uri_folder)
    if not matches:
        raise ValueError(f"Unsupported uri as uri_folder: {root_uri_folder}")
    scheme = "https" if matches.group("scheme") == "abfss" else "http"
    account_url = f"{scheme}://{matches.group('account_name')}.dfs.core.windows.net"
    # TODO use SAS token
    print("getting service client...")
    service_client = service_client or DataLakeServiceClient(account_url, credential=DefaultAzureCredential())
    print("done getting service client.")
    file_system_client = service_client.get_file_system_client(matches.group("container"))
    root_path = matches.group("path").rstrip('/')
    if end_datetime.minute == 0 and end_datetime.second == 0:
        # if end_datetime is a whole hour, the last hour folder is not needed
        end_datetime -= timedelta(seconds=1)
    cur_datetime = start_datetime
    while cur_datetime <= end_datetime:
        if _folder_exists(file_system_client, f"{root_path}/{cur_datetime.strftime('%Y/%m/%d/%H')}"):
            cur_folder = f"{root_uri_folder}/{cur_datetime.strftime('%Y/%m/%d/%H')}"
            file_list.append(f"{cur_folder}/*.jsonl")
        cur_datetime += timedelta(hours=1)
    return file_list


def _get_local_file_list(start_datetime: datetime, end_datetime: datetime, input_data: str) -> List[str]:
    file_list = []
    root_folder = input_data.rstrip('/')
    if end_datetime.minute == 0 and end_datetime.second == 0:
        # if end_datetime is a whole hour, the last hour folder is not needed
        end_datetime -= timedelta(seconds=1)
    cur_datetime = start_datetime
    while cur_datetime <= end_datetime:
        cur_folder = f"{root_folder}/{cur_datetime.strftime('%Y/%m/%d/%H')}"
        if os.path.isdir(cur_folder):
            for file in os.listdir(cur_folder):
                full_qualify_file = os.path.join(cur_folder, file)
                if os.path.isfile(full_qualify_file) and file.endswith(".jsonl"):
                    file_list.append(full_qualify_file)  # local winutils/hadoop package doesn't support wildcard
        cur_datetime += timedelta(hours=1)
    return file_list


def _is_local_path(path: str) -> bool:
    return os.path.isdir(path) or path.startswith("file://") or path.startswith("/") or path.startswith(".") or \
        re.match(r"^[a-zA-Z]:[/\\]", path)


def _folder_exists(file_system_client: FileSystemClient, folder_path: str) -> bool:
    """Check if hdfs folder exists."""
    # DataLakeDirectoryClient.exists() doesn't work for blob, need to use DataLakeFileSystemClient.get_paths()
    files = file_system_client.get_paths(folder_path, recursive=False, max_results=1)
    try:
        _ = next(files)
        return True
    except (ResourceNotFoundError, StopIteration):
        return False


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

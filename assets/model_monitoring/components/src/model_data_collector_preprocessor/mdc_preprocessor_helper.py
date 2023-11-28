# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper class for MDC preprocessor Component."""

import os
import re
from typing import Tuple, List, Union
from datetime import datetime, timedelta
from urllib.parse import urlparse
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from azureml.data.azure_data_lake_datastore import AzureDataLakeGen2Datastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient
from azure.storage.blob import ContainerClient
from azure.identity import ClientSecretCredential
from azure.core.credentials import AzureSasCredential
from pyspark.sql import SparkSession
from shared_utilities.io_utils import init_spark


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


def get_hdfs_path_and_container_client(uri_folder_path: str, ws: Workspace = None, spark: SparkSession = None) \
        -> Tuple[str, Union[FileSystemClient, ContainerClient]]:
    """Get HDFS path and container/filesystem client from url_folder_path."""
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
        container_name = matches.group("container")
        path = matches.group("path")
        filesystem_client = _get_filesystem_client(account_name, container_name, matches.group("store_type"), spark)
        return f"{scheme}://{container_name}@{account_name}.{store_type}.core.windows.net/{path}", filesystem_client
    elif url.scheme in ["wasbs", "wasb", "abfss", "abfs"]:
        pattern = r"(?P<scheme>wasbs|abfss|wasb|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+).(?P<store_type>blob|dfs).core.windows.net/(?P<path>.+)"  # noqa
        matches = re.match(pattern, uri_folder_path)
        if not matches:
            raise ValueError(f"Unsupported uri as uri_folder: {uri_folder_path}")
        scheme = "abfss" if matches.group("scheme") in ["wasbs", "abfss"] else "abfs"
        store_type = "dfs"
        account_name = matches.group("account_name")
        container_name = matches.group("container")
        path = matches.group("path")
        filesystem_client = _get_filesystem_client(account_name, container_name, matches.group("store_type"), spark)
        return f"{scheme}://{container_name}@{account_name}.{store_type}.core.windows.net/{path}", filesystem_client
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
            container_name = datastore.container_name
            container_client = _get_container_client(datastore)
            return f"{scheme}://{container_name}@{account_name}.{store_type}.core.windows.net/{path}", container_client
    else:
        return uri_folder_path, None  # file or other scheme, return original path directly


def get_datastore_name_from_input_path(input_path: str) -> str:
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
                  root_uri_folder: str = None,
                  container_client: Union[ContainerClient, FileSystemClient] = None) -> List[str]:
    """Get the available file list for the given time window under the input_data folder."""
    if _is_local_path(input_data):
        # for lcoal testing
        return _get_local_file_list(start_datetime, end_datetime, input_data)

    file_list = []
    if not root_uri_folder:
        root_uri_folder, container_client = get_hdfs_path_and_container_client(input_data)
    root_uri_folder = root_uri_folder.rstrip('/')

    # get meta from root_uri_folder
    pattern = r"(?P<scheme>abfss|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+).dfs.core.windows.net(?P<path>$|/.*)"  # noqa
    matches = re.match(pattern, root_uri_folder)
    if not matches:
        raise ValueError(f"Unsupported uri as uri_folder: {root_uri_folder}")
    root_path = matches.group("path").strip('/')
    if end_datetime.minute == 0 and end_datetime.second == 0:
        # if end_datetime is a whole hour, the last hour folder is not needed
        end_datetime -= timedelta(seconds=1)
    cur_datetime = start_datetime
    # TODO check day folder and use */*.jsonl if day folder exists
    while cur_datetime <= end_datetime:
        if _folder_exists(container_client, f"{root_path}/{cur_datetime.strftime('%Y/%m/%d/%H')}"):
            cur_folder = f"{root_uri_folder}/{cur_datetime.strftime('%Y/%m/%d/%H')}"
            file_list.append(f"{cur_folder}/*.jsonl")
        cur_datetime += timedelta(hours=1)
    return file_list


def set_data_access_config(container_client: Union[FileSystemClient, ContainerClient], spark: SparkSession):
    """
    Set data access relative spark configs.

    This is a special handling to access blob storage with abfs protocol, which enable Spark to access
    append blob in blob storage.
    """
    if isinstance(container_client, FileSystemClient) or container_client is None:
        # already a gen2 store, or local path, neither need to set config
        return

    container_name = container_client.container_name
    account_name = container_client.account_name
    sas_token = spark.conf.get(
        f"spark.hadoop.fs.azure.sas.{container_name}.{account_name}.blob.core.windows.net", None)
    if not sas_token:
        raise ValueError("Credential less datastore is not supported as input of the Model Monitoring")

    # in Synapse doc, it uses sc._jsc.hadoopConfiguration().set to set first two configs
    # per testing, use spark.conf.set() also works
    spark.conf.set(f"fs.azure.account.auth.type.{account_name}.dfs.core.windows.net", "SAS")
    spark.conf.set(f"fs.azure.sas.token.provider.type.{account_name}.dfs.core.windows.net",
                   "com.microsoft.azure.synapse.tokenlibrary.ConfBasedSASProvider")
    spark.conf.set(f"spark.storage.synapse.{container_name}.{account_name}.dfs.core.windows.net.sas", sas_token)


def read_json_content(path: str) -> str:
    """Read json content from path."""
    if not path:
        return None
    if _is_local_path(path):
        with open(path) as f:
            json_str = f.read()
        return json_str
    hdfs_path, container_client = get_hdfs_path_and_container_client(path)
    idx = hdfs_path.find('.core.windows.net/') + len('.core.windows.net/')
    file_path = hdfs_path[idx:]
    if not container_client:
        raise ValueError(f"Fail to get credential to access dataref file {hdfs_path}")
    elif isinstance(container_client, FileSystemClient):
        with container_client.get_file_client(file_path) as file_client:
            json_bytes = file_client.download_file().readall()
    else:  # must be ContainerClient
        with container_client.get_blob_client(file_path) as blob_client:
            json_bytes = blob_client.download_blob().readall()

    return json_bytes.decode()


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
    if not path:
        return False
    return os.path.isdir(path) or path.startswith("file://") or path.startswith("/") or path.startswith(".") or \
        re.match(r"^[a-zA-Z]:[/\\]", path)


def _folder_exists(container_client: Union[ContainerClient, FileSystemClient], folder_path: str) -> bool:
    """Check if hdfs folder exists."""
    if isinstance(container_client, FileSystemClient):
        return container_client.get_directory_client(folder_path).exists()
    else:
        blobs = container_client.list_blobs(name_starts_with=folder_path)
        return any(blobs)


def _get_datastore_and_path_from_azureml_path(azureml_path: str) -> Tuple[str, str]:
    start_idx = azureml_path.find('/datastores/')
    end_idx = azureml_path.find('/paths/')
    return azureml_path[start_idx+12:end_idx], azureml_path[end_idx+7:]


def _get_container_client(datastore: Union[AzureDataLakeGen2Datastore, AzureBlobDatastore]) \
        -> Union[FileSystemClient, ContainerClient]:
    if datastore.datastore_type == "AzureBlob":
        return datastore.blob_service.get_container_client(datastore.container_name) if datastore.credential_type \
               else None
    elif datastore.datastore_type == "AzureDataLakeGen2":
        account_url = f"{datastore.protocol}://{datastore.account_name}.dfs.{datastore.endpoint}"
        client_secret_credential = \
            ClientSecretCredential(tenant_id=datastore.tenant_id,
                                   client_id=datastore.client_id,
                                   client_secret=datastore.client_secret) \
            if datastore.client_id else None
        service_client = DataLakeServiceClient(account_url, credential=client_secret_credential)
        return service_client.get_file_system_client(datastore.container_name)


def _get_filesystem_client(account_name: str, container_name: str, store_type: str,
                           spark: SparkSession = None) -> FileSystemClient:
    # TODO assuming we always access the same container which have the MDC logs, optimize it by cache the client
    account_url = f"https://{account_name}.dfs.core.windows.net"  # return filesystem client even for blob storage
    spark = spark or init_spark()
    # TODO this is only for blob, if gen2, credential may not be sas
    sas_token = spark.conf.get(
        f"spark.hadoop.fs.azure.sas.{container_name}.{account_name}.{store_type}.core.windows.net", None)
    if not sas_token:
        return None
    # TODO maybe we can return ContainerClient for blob storage, and FileSystemClient for gen2 storage
    sas_credential = AzureSasCredential(sas_token)
    service_client = DataLakeServiceClient(account_url, credential=sas_credential)
    filesystem_client = service_client.get_file_system_client(container_name)
    return filesystem_client

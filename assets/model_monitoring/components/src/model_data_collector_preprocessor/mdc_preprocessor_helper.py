# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper methods for MDC preprocessor Component."""

import os
import re
from typing import List, Union
from datetime import datetime, timedelta
from azure.storage.filedatalake import FileSystemClient
from azure.storage.blob import ContainerClient
from pyspark.sql import SparkSession
from shared_utilities.momo_exceptions import InvalidInputError
from store_url import StoreUrl


def get_file_list(start_datetime: datetime, end_datetime: datetime, input_data: str,
                  store_url: StoreUrl = None) -> List[str]:
    """Get the available file list for the given time window under the input_data folder."""
    if _is_local_path(input_data):
        # for local testing
        return _get_local_file_list(start_datetime, end_datetime, input_data)

    file_list = []
    store_url = store_url or StoreUrl(input_data)

    if end_datetime.minute == 0 and end_datetime.second == 0:
        # if end_datetime is a whole hour, the last hour folder is not needed
        end_datetime -= timedelta(seconds=1)
    cur_datetime = start_datetime
    # TODO check day folder and use */*.jsonl if day folder exists
    while cur_datetime <= end_datetime:
        if store_url.is_folder_exists(f"{cur_datetime.strftime('%Y/%m/%d/%H')}"):
            cur_folder = store_url.get_abfs_url(f"{cur_datetime.strftime('%Y/%m/%d/%H')}")
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
        raise InvalidInputError("Credential less datastore is not supported as input of the Model Monitoring for now.")

    # in Synapse doc, it uses sc._jsc.hadoopConfiguration().set to set first two configs
    # per testing, use spark.conf.set() also works
    spark.conf.set(f"fs.azure.account.auth.type.{account_name}.dfs.core.windows.net", "SAS")
    spark.conf.set(f"fs.azure.sas.token.provider.type.{account_name}.dfs.core.windows.net",
                   "com.microsoft.azure.synapse.tokenlibrary.ConfBasedSASProvider")
    spark.conf.set(f"spark.storage.synapse.{container_name}.{account_name}.dfs.core.windows.net.sas", sas_token)


def read_json_content(path: str, credential_info: str) -> str:
    """Read json content from path."""
    if not path:
        return None

    if _is_local_path(path):
        with open(path) as f:
            json_str = f.read()
        return json_str

    store_url = StoreUrl(path)
    file_path = store_url.path
    container_client = store_url.get_container_client(credential_info)
    if isinstance(container_client, FileSystemClient):
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
    return os.path.isdir(path) or os.path.isfile(path) or path.startswith("file://") or path.startswith("/") \
        or path.startswith(".") or re.match(r"^[a-zA-Z]:[/\\]", path)


def _folder_exists(container_client: Union[ContainerClient, FileSystemClient], folder_path: str) -> bool:
    """Check if hdfs folder exists."""
    if isinstance(container_client, FileSystemClient):
        return container_client.get_directory_client(folder_path).exists()
    else:
        blobs = container_client.list_blobs(name_starts_with=folder_path)
        return any(blobs)

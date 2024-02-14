# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper methods for MDC preprocessor Component."""

import os
import subprocess
from typing import List, Union
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import calendar
from azure.core.credentials import AzureSasCredential
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient, ContainerClient, ContainerSasPermissions, generate_container_sas
from pyspark.sql import SparkSession
from model_data_collector_preprocessor.store_url import StoreUrl
from shared_utilities.momo_exceptions import InvalidInputError


def get_file_list(start_datetime: datetime, end_datetime: datetime, store_url: StoreUrl = None,
                  input_data: str = None, scheme: str = "abfs") -> List[str]:
    """Get the available file list for the given time window under the input_data folder."""
    def date_range(start, end, step):
        cur = start
        while cur < end:
            yield cur
            cur += step

    def is_same_year(start, end) -> bool:
        return start.year == end.year

    def is_same_month(start, end) -> bool:
        return is_same_year(start, end) and start.month == end.month

    def is_same_day(start, end) -> bool:
        return is_same_month(start, end) and start.day == end.day

    def is_start_of_year(d) -> bool:
        return (d.month, d.day, d.hour) == (1, 1, 0)

    def is_end_of_year(d) -> bool:
        return (d.month, d.day, d.hour) == (12, 31, 23)

    def is_start_of_month(d) -> bool:
        return (d.day, d.hour) == (1, 0)

    def is_end_of_month(d) -> bool:
        _, month_days = calendar.monthrange(d.year, d.month)
        return (d.day, d.hour) == (month_days, 23)

    def is_start_of_day(d) -> bool:
        return d.hour == 0

    def is_end_of_day(d) -> bool:
        return d.hour == 23

    def get_url(path) -> str:
        if scheme == "abfs":
            return store_url.get_abfs_url(path)
        elif scheme == "azureml":
            return store_url.get_azureml_url(path)
        else:
            raise ValueError(f"Unsupported scheme {scheme}")

    def same_year(start, end) -> List[str]:
        if is_same_month(start, end):
            return same_month(start, end)
        if is_start_of_year(start) and is_end_of_year(end):
            return [get_url(f"{start.strftime('%Y')}/*/*/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{start.strftime('%Y')}") else []
        return cross_month(start, end)

    def same_month(start, end) -> List[str]:
        if is_same_day(start, end):
            return same_day(start, end)
        if is_start_of_month(start) and is_end_of_month(end):
            return [get_url(f"{start.strftime('%Y/%m')}/*/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{start.strftime('%Y/%m')}") else []
        return cross_day(start, end)

    def same_day(start, end) -> List[str]:
        if is_start_of_day(start) and is_end_of_day(end):
            return [get_url(f"{start.strftime('%Y/%m/%d')}/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{start.strftime('%Y/%m/%d')}") else []
        return cross_hour(start, end)

    def start_of_year(y) -> List[str]:
        if is_end_of_year(y):
            return [get_url(f"{y.strftime('%Y')}/*/*/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{y.strftime('%Y')}") else []
        return same_year(y.replace(month=1, day=1, hour=0, minute=0, second=0), y)

    def end_of_year(y) -> List[str]:
        if is_start_of_year(y):
            return [get_url(f"{y.strftime('%Y')}/*/*/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{y.strftime('%Y')}") else []
        return same_year(y, y.replace(month=12, day=31, hour=23, minute=59, second=59))

    def start_of_month(m) -> List[str]:
        if is_end_of_month(m):
            return [get_url(f"{m.strftime('%Y/%m')}/*/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{m.strftime('%Y/%m')}") else []
        return same_month(m.replace(day=1, hour=0, minute=0, second=0), m)

    def end_of_month(m) -> List[str]:
        if is_start_of_month(m):
            return [get_url(f"{m.strftime('%Y/%m')}/*/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{m.strftime('%Y/%m')}") else []
        _, month_days = calendar.monthrange(m.year, m.month)
        return same_month(m, m.replace(day=month_days, hour=23, minute=59, second=59))

    def start_of_day(d) -> List[str]:
        if is_end_of_day(d):
            return [get_url(f"{d.strftime('%Y/%m/%d')}/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{d.strftime('%Y/%m/%d')}") else []
        return same_day(d.replace(hour=0, minute=0, second=0), d)

    def end_of_day(d) -> List[str]:
        if is_start_of_day(d):
            return [get_url(f"{d.strftime('%Y/%m/%d')}/*/*.jsonl")] \
                if store_url.is_folder_exists(f"{d.strftime('%Y/%m/%d')}") else []
        return same_day(d, d.replace(hour=23, minute=59, second=59))

    def cross_year(start, end) -> List[str]:
        middle_years = [
            get_url(f"{y}/*/*/*/*.jsonl") for y in range(start.year+1, end.year)
            if store_url.is_folder_exists(f"{y}")
        ]
        return end_of_year(start) + middle_years + start_of_year(end)

    def cross_month(start: datetime, end) -> List[str]:
        _start = (start + relativedelta(months=1)).replace(day=1, hour=1)
        _end = end.replace(day=1, hour=0)  # skip last month
        middle_months = [
            get_url(f"{m.strftime('%Y/%m')}/*/*/*.jsonl")
            for m in date_range(_start, _end, relativedelta(months=1))
            if store_url.is_folder_exists(f"{m.strftime('%Y/%m')}")
        ]
        return end_of_month(start) + middle_months + start_of_month(end)

    def cross_day(start, end) -> List[str]:
        _start = (start + timedelta(days=1)).replace(hour=1)
        _end = end.replace(hour=0)  # skip last day
        middle_days = [
            get_url(f"{d.strftime('%Y/%m/%d')}/*/*.jsonl")
            for d in date_range(_start, _end, timedelta(days=1))
            if store_url.is_folder_exists(f"{d.strftime('%Y/%m/%d')}")
        ]
        return end_of_day(start) + middle_days + start_of_day(end)

    def cross_hour(start, end) -> List[str]:
        _start = start.replace(minute=0)
        _end = end.replace(minute=59)
        return [
            get_url(f"{h.strftime('%Y/%m/%d/%H')}/*.jsonl")
            for h in date_range(_start, _end, timedelta(hours=1))
            if store_url.is_folder_exists(f"{h.strftime('%Y/%m/%d/%H')}")
        ]

    # ensure start_datetime and end_datetime both or neither have tzinfo
    if start_datetime.tzinfo is None:
        start_datetime = start_datetime.replace(tzinfo=end_datetime.tzinfo)
    if end_datetime.tzinfo is None:
        end_datetime = end_datetime.replace(tzinfo=start_datetime.tzinfo)

    store_url = store_url or StoreUrl(input_data)
    if store_url.is_local_path():
        # for local testing
        return _get_local_file_list(start_datetime, end_datetime, store_url._base_url)

    if end_datetime.minute == 0 and end_datetime.second == 0:
        # if end_datetime is a whole hour, the last hour folder is not needed
        end_datetime -= timedelta(seconds=1)
    if is_same_year(start_datetime, end_datetime):
        return same_year(start_datetime, end_datetime)
    return cross_year(start_datetime, end_datetime)


def copy_appendblob_to_blockblob(appendblob_url: StoreUrl,
                                 start_datetime: datetime, end_datetime: datetime) -> StoreUrl:
    """Copy append blob to block blob and return the StoreUrl of block blob."""
    datastore = appendblob_url._datastore
    datastore_type = None if datastore is None else datastore.datastore_type
    if datastore_type == "AzureBlob":
        sas_token = _get_sas_token(appendblob_url.account_name, appendblob_url.container_name,
                                   appendblob_url.get_credential())
        base_path = appendblob_url.path.strip('/')
        blockblob_folder_suffix = "block_blob"
        _azcopy_appendblob_to_blockblob(appendblob_url.account_name, appendblob_url.container_name, base_path,
                                        blockblob_folder_suffix, start_datetime, end_datetime, sas_token)
        # return the StoreUrl for block blob
        sub_id = datastore.workspace.subscription_id
        rg = datastore.workspace.resource_group
        ws_name = datastore.workspace.name
        datastore_name = datastore.name
        blockblob_url = (f"azureml://subscriptions/{sub_id}/resourcegroups/{rg}/workspaces/{ws_name}"
                         f"/datastores/{datastore_name}/paths/{base_path}_{blockblob_folder_suffix}")
        return StoreUrl(blockblob_url)
    elif datastore_type == "AzureDataLakeStorageGen2":
        raise NotImplementedError("Append blob in AdlsGen2 storage should be accessible via abfss protocol "
                                  "even soft delete is enabled, should not need to copy to block blob.")
    elif datastore_type is None:
        raise InvalidInputError("Credential-less input data is not supported.")
    else:
        raise InvalidInputError(f"Storage account type {datastore_type} is not supported!")


def _get_sas_token(account_name, container_name, credential) -> str:
    """Get sas token for append blob in blob storage."""
    def get_blob_sas_token_from_client_secret_credential(client_secret_credential: ClientSecretCredential):
        account_url = f"https://{account_name}.blob.core.windows.net"
        blob_service_client = BlobServiceClient(account_url=account_url, credential=client_secret_credential)
        # get a user delegation key for the Blob service that's valid for 1 day
        key_start_time = datetime.utcnow() - timedelta(minutes=15)
        key_expiry_time = datetime.utcnow() + timedelta(days=1)
        # TODO raise validation error if the SP has no permission to generate user delegation key
        user_delegation_key = blob_service_client.get_user_delegation_key(key_start_time, key_expiry_time)
        # get a sas token with the user delegation key
        return generate_container_sas(
            account_name=account_name, container_name=container_name,
            user_delegation_key=user_delegation_key,
            permission=ContainerSasPermissions(read=True, write=True, list=True),
            expiry=datetime.utcnow() + timedelta(hours=8))

    if credential is None:
        raise InvalidInputError("credential-less input data is NOT supported!")
    if isinstance(credential, AzureSasCredential):
        return credential.signature
    if isinstance(credential, str):  # account key
        return generate_container_sas(
            account_name=account_name, container_name=container_name,
            account_key=credential, permission=ContainerSasPermissions(read=True, write=True, list=True),
            expiry=datetime.utcnow() + timedelta(hours=8))
    if isinstance(credential, ClientSecretCredential):
        return get_blob_sas_token_from_client_secret_credential(credential)

    raise InvalidInputError(f"Credential type {type(credential)} is not supported, "
                            "please use account key or SAS token.")


def _copy_appendblob_to_blockblob(container_client: ContainerClient, base_path: str, start_datetime, end_datetime,
                                  sas_token: str):
    cur_datetime = start_datetime
    while cur_datetime <= end_datetime:
        datetime_path = cur_datetime.strftime('%Y/%m/%d/%H')
        appendblob_names = container_client.list_blob_names(name_starts_with=f"{base_path}/{datetime_path}")
        for appendblob_name in appendblob_names:
            if not appendblob_name.endswith(".jsonl"):
                continue
            appendblob_client = container_client.get_blob_client(appendblob_name)
            blockblob_client = container_client.get_blob_client(f"{base_path}/block_blob/{datetime_path}"
                                                                f"/{appendblob_name.split('/')[-1]}")
            if not blockblob_client.exists():
                blockblob_client.upload_blob_from_url(f"{appendblob_client.url}?{sas_token}", overwrite=False)

        cur_datetime += timedelta(hours=1)


def _azcopy_appendblob_to_blockblob(account_name: str, container_name: str, base_path: str, blockblob_suffix: str,
                                    start_datetime: datetime, end_datetime: datetime, sas_token: str):
    """
    Copy append blob to block blob using azcopy.

    The block blob base path will be same as append blob base path with 'blockblob_suffix' as suffix.
    """
    base_path = base_path.strip('/')
    start_datetime = start_datetime.replace(hour=0, minute=0, second=0)
    end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
    cur_datetime = start_datetime
    # copy day by day to utilize parallelism of azcopy
    while cur_datetime <= end_datetime:
        datetime_path = cur_datetime.strftime('%Y/%m/%d')
        src_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{base_path}/{datetime_path}/*?{sas_token}"  # noqa: E501
        dst_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{base_path}_{blockblob_suffix}/{datetime_path}?{sas_token}"  # noqa: E501
        cmd = (f'azcopy copy "{src_url}" "{dst_url}" --recursive --overwrite false --blob-type BlockBlob')
        # Execute the command
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Get the output and error messages, if any
        stdout, stderr = process.communicate()
        # Print the output
        if process.returncode == 0:
            print(f"Success: {stdout.decode()}")
        else:
            print(f"Error: {stdout.decode()}\n{stderr.decode()}")
            raise RuntimeError(f"Failed to copy append blob to block blob: {stdout.decode()}")
        cur_datetime += timedelta(days=1)


def set_data_access_config(spark: SparkSession, input_data: str = None, store_url: StoreUrl = None):
    """
    Set data access relative spark configs.

    This is a special handling to access blob storage with abfs protocol, which enable Spark to access
    append blob in blob storage.
    """
    def set_sas_config(sas_token):
        account_name = store_url.account_name
        container_name = store_url.container_name
        spark.conf.set(f"fs.azure.account.auth.type.{account_name}.dfs.core.windows.net", "SAS")
        spark.conf.set(f"fs.azure.sas.token.provider.type.{account_name}.dfs.core.windows.net",
                       "com.microsoft.azure.synapse.tokenlibrary.ConfBasedSASProvider")
        spark.conf.set(f"spark.storage.synapse.{container_name}.{account_name}.dfs.core.windows.net.sas", sas_token)

    store_url = store_url or StoreUrl(input_data)
    if store_url.is_local_path() or store_url.store_type == "dfs":
        # already a gen2 store, local path, no need to set config
        return

    credential = store_url.get_credential()
    if not credential:
        # no credential, no need to set config
        return
    elif isinstance(credential, str):
        # account_key
        spark.conf.set(f"fs.azure.account.auth.type.{store_url.account_name}.dfs.core.windows.net", "SharedKey")
        spark.conf.set(f"fs.azure.account.key.{store_url.account_name}.dfs.core.windows.net", credential)
    elif isinstance(credential, AzureSasCredential):
        # sas token
        sas_token = credential.signature
        set_sas_config(sas_token)
    else:
        # TODO fallback to spark conf "spark.hadoop.fs.azure.sas.my_container.my_account.blob.core.windows.net"
        return


def serialize_credential(credential) -> str:
    """Serialize the credential and broadcast to all executors."""
    if isinstance(credential, str):
        # account key
        cred_dict = {"account_key": credential}
    elif isinstance(credential, AzureSasCredential):
        # sas token
        cred_dict = {"sas_token": credential.signature}
    elif isinstance(credential, ClientSecretCredential):
        # service principal
        cred_dict = {
            "tenant_id": credential._tenant_id,
            "client_id": credential._client_id,
            "client_secret": credential._client_credential
        }
    else:
        # TODO support credential pass through for credential less data
        cred_dict = {}

    return json.dumps(cred_dict)


def deserialize_credential(credential_value: str) -> Union[str, AzureSasCredential, ClientSecretCredential]:
    """Deserialize the credential from broadcasted value."""
    if not credential_value:
        # no credential
        return None

    try:
        cred_dict = json.loads(credential_value)
    except json.JSONDecodeError:
        # unrecognized credential
        return None

    account_key = cred_dict.get("account_key")
    if account_key:
        return account_key

    sas_token = cred_dict.get("sas_token")
    if sas_token:
        return AzureSasCredential(sas_token)

    tenant_id = cred_dict.get("tenant_id")
    client_id = cred_dict.get("client_id")
    client_secret = cred_dict.get("client_secret")
    if tenant_id and client_id and client_secret:
        return ClientSecretCredential(tenant_id, client_id, client_secret)

    # unrecognized credential
    return None


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

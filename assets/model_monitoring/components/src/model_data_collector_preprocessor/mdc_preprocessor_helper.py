# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper methods for MDC preprocessor Component."""

import os
from typing import List, Union
from datetime import datetime, timedelta
import json
from azure.core.credentials import AzureSasCredential
from azure.identity import ClientSecretCredential
from pyspark.sql import SparkSession
from model_data_collector_preprocessor.store_url import StoreUrl


def get_file_list(start_datetime: datetime, end_datetime: datetime, store_url: StoreUrl = None,
                  input_data: str = None) -> List[str]:
    """Get the available file list for the given time window under the input_data folder."""
    store_url = store_url or StoreUrl(input_data)
    if store_url.is_local_path():
        # for local testing
        return _get_local_file_list(start_datetime, end_datetime, store_url._base_url)

    file_list = []
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

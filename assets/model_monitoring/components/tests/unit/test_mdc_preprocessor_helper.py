# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for MDC preprocessor helper."""

from unittest.mock import Mock
import pytest
from datetime import datetime
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.credentials import AzureSasCredential
from model_data_collector_preprocessor.mdc_preprocessor_helper import (
    get_file_list, set_data_access_config, serialize_credential, deserialize_credential
)
from model_data_collector_preprocessor.store_url import StoreUrl
from test_store_url import assert_credentials_are_equal


@pytest.mark.unit
class TestMDCPreprocessorHelper:
    """Test class for MDC preprocessor helper."""

    @pytest.mark.parametrize(
        "start_hour, end_hour, expected_hours",
        [
            (7, 21, [7, 8, 13, 17, 19, 20]),
            (4, 8, [6, 7]),
            (20, 23, [20, 21]),
            (13, 19, [13, 17]),
            (3, 6, []),
            (22, 23, []),
            (9, 13, []),
        ]
    )
    def test_get_file_list(self, start_hour, end_hour, expected_hours):
        """Test get_file_list()."""
        non_empty_hours = [6, 7, 8, 13, 17, 19, 20, 21]
        non_empty_folders = [f"2023/11/20/{h:02d}" for h in non_empty_hours]

        mock_store_url = Mock(spec=StoreUrl)
        mock_store_url.is_local_path.return_value = False
        mock_store_url.is_folder_exists.side_effect = lambda path: path in non_empty_folders
        mock_store_url.get_abfs_url.side_effect = \
            lambda rpath: f"abfss://my_container@my_account.dfs.core.windows.net/path/to/folder/{rpath}"

        start = datetime(2023, 11, 20, start_hour)
        end = datetime(2023, 11, 20, end_hour)
        file_list = get_file_list(start, end, mock_store_url)

        assert file_list == [
            f"abfss://my_container@my_account.dfs.core.windows.net/path/to/folder/2023/11/20/{h:02d}/*.jsonl"
            for h in expected_hours
        ]

    @pytest.mark.parametrize(
        "is_local, store_type, credential, expected_spark_conf",
        [
            (True, None, None, {}),
            (False, "dfs", None, {}),
            (False, "blob", "account_key", {
                "fs.azure.account.auth.type.my_account.dfs.core.windows.net": "SharedKey",
                "fs.azure.account.key.my_account.dfs.core.windows.net": "account_key"
            }),
            (False, "blob", AzureSasCredential("sas_token"), {
                "fs.azure.account.auth.type.my_account.dfs.core.windows.net": "SAS",
                "fs.azure.sas.token.provider.type.my_account.dfs.core.windows.net":
                    "com.microsoft.azure.synapse.tokenlibrary.ConfBasedSASProvider",
                "spark.storage.synapse.my_container.my_account.dfs.core.windows.net.sas": "sas_token"
            }),
            (False, "blob", ClientSecretCredential("00000", "client_id", "client_secret"), {})
        ]
    )
    def test_set_data_access_config(self, is_local, store_type, credential, expected_spark_conf):
        """Test set_data_access_config()."""
        class MyDict(dict):
            def __init__(self, d: dict):
                super().__init__(d)

            def set(self, k, v):
                self[k] = v

        mock_spark = Mock(conf=MyDict({}))
        mock_store_url = Mock(spec=StoreUrl, store_type=store_type,
                              account_name="my_account", container_name="my_container")
        mock_store_url.is_local_path.return_value = is_local
        mock_store_url.get_credential.return_value = credential

        set_data_access_config(mock_spark, store_url=mock_store_url)

        assert mock_spark.conf == expected_spark_conf

    @pytest.mark.parametrize(
        "credential",
        [
            "account_key",
            AzureSasCredential("sas_token"),
            ClientSecretCredential("00000", "client_id", "client_secret"),
            DefaultAzureCredential(),
            None
        ]
    )
    def test_serialize_deserialize_credential(self, credential):
        """Test serialize_credential() and deserialize_credential()."""
        serialized_credential = serialize_credential(credential)
        deserialized_credential = deserialize_credential(serialized_credential)

        expected_credential = None if isinstance(credential, DefaultAzureCredential) else credential
        assert_credentials_are_equal(deserialized_credential, expected_credential)

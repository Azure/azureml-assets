# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test class for mdc_preprocessor_helper.py."""

from unittest.mock import Mock
import pytest
from datetime import datetime
from azure.storage.filedatalake import FileSystemClient
from azure.storage.blob import ContainerClient
from model_data_collector_preprocessor.mdc_preprocessor_helper import get_file_list, set_data_access_config
from model_data_collector_preprocessor.store_url import StoreUrl


@pytest.mark.unit
class TestMDCPreprocessorHelper:
    """Test class for MDC Preprocessor."""
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
    def test_get_file_list2(self, start_hour, end_hour, expected_hours):
        """Test get_file_list()."""
        non_empty_hours = [6, 7, 8, 13, 17, 19, 20, 21]
        non_empty_folders = [f"2023/11/20/{h:02d}" for h in non_empty_hours]

        mock_store_url = Mock(spec=StoreUrl)
        mock_store_url.is_folder_exists.side_effect = lambda path: path in non_empty_folders
        mock_store_url.get_abfs_url.side_effect = \
            lambda rpath: f"abfss://my_container@my_account.dfs.core.windows.net/path/to/folder/{rpath}"

        start = datetime(2023, 11, 20, start_hour)
        end = datetime(2023, 11, 20, end_hour)
        file_list = get_file_list(start, end, "uri_folder_path", mock_store_url)

        assert file_list == [
            f"abfss://my_container@my_account.dfs.core.windows.net/path/to/folder/2023/11/20/{h:02d}/*.jsonl"
            for h in expected_hours
        ]

    def test_set_data_access_config(self):
        """Test set_data_access_config()."""
        class MyDict(dict):
            def __init__(self, d: dict):
                super().__init__(d)

            def set(self, k, v):
                self[k] = v

        # blob
        mock_container_client = Mock(spec=ContainerClient, account_name="my_account", container_name="my_container")
        mock_spark = Mock(conf=MyDict({
            "spark.hadoop.fs.azure.sas.my_container.my_account.blob.core.windows.net": "sas_token"
        }))
        set_data_access_config(mock_container_client, mock_spark)
        assert mock_spark.conf == {
            "fs.azure.account.auth.type.my_account.dfs.core.windows.net": "SAS",
            "fs.azure.sas.token.provider.type.my_account.dfs.core.windows.net":
                "com.microsoft.azure.synapse.tokenlibrary.ConfBasedSASProvider",
            "spark.storage.synapse.my_container.my_account.dfs.core.windows.net.sas": "sas_token",
            "spark.hadoop.fs.azure.sas.my_container.my_account.blob.core.windows.net": "sas_token"
        }

        # gen2
        mock_container_client = Mock(spec=FileSystemClient)
        mock_spark = Mock(conf=MyDict({
            "spark.hadoop.fs.azure.sas.my_container.my_account.blob.core.windows.net": "sas_token"
        }))
        set_data_access_config(mock_container_client, mock_spark)
        assert mock_spark.conf == {
            "spark.hadoop.fs.azure.sas.my_container.my_account.blob.core.windows.net": "sas_token"
        }

        # local
        mock_container_client = None
        mock_spark = Mock(conf=MyDict({}))
        set_data_access_config(mock_container_client, mock_spark)
        assert mock_spark.conf == {}

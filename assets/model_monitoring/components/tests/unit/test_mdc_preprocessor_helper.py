# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test class for mdc_preprocessor_helper.py."""

from unittest.mock import Mock
import pytest
from datetime import datetime
from azure.storage.filedatalake import FileSystemClient
from azure.storage.blob import ContainerClient
from model_data_collector_preprocessor.mdc_preprocessor_helper import (
    convert_to_azureml_long_form, get_hdfs_path_and_container_client, get_datastore_name_from_input_path,
    get_file_list, set_data_access_config
)


@pytest.mark.unit
class TestMDCPreprocessorHelper:
    """Test class for MDC Preprocessor."""

    @pytest.mark.parametrize(
        "url_str, converted",
        [
            ("https://my_account.blob.core.windows.net/my_container/path/to/file", True),
            ("wasbs://my_container@my_account.blob.core.windows.net/path/to/file", True),
            ("abfss://my_container@my_account.dfs.core.windows.net/path/to/file", True),
            (
                "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name"
                "/datastores/my_datastore/paths/path/to/file",
                True
            ),
            ("azureml://datastores/my_datastore/paths/path/to/file", True),
            ("azureml:my_asset:my_version", False),
            ("file://path/to/file", False),
            ("path/to/file", False),
        ]
    )
    def test_convert_to_azureml_long_form(self, url_str: str, converted: bool):
        """Test convert_to_azureml_long_form()."""
        converted_path = convert_to_azureml_long_form(url_str, "my_datastore", "my_sub_id",
                                                      "my_rg_name", "my_ws_name")
        azureml_long = "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name" \
            "/datastores/my_datastore/paths/path/to/file"
        expected_path = azureml_long if converted else url_str
        assert converted_path == expected_path

    @pytest.mark.parametrize(
        "uri_folder_path, expected_hdfs_path",
        [
            (
                "https://my_account.blob.core.windows.net/my_container/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "https://my_account.dfs.core.windows.net/my_container/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "http://my_account.blob.core.windows.net/my_container/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder"
            )
        ]
    )
    def test_get_hdfs_path_and_container_client(self, uri_folder_path: str, expected_hdfs_path: str):
        """Test get_hdfs_path_and_container_client()."""
        for mock_token in ["my_token", None]:
            spark = None if mock_token else Mock(conf={
                "spark.hadoop.fs.azure.sas.my_container.my_account.blob.core.windows.net": "my_token",
                # TODO below is quick workaround to pass the UT, need to refine the UT later
                "spark.hadoop.fs.azure.sas.my_container.my_account.dfs.core.windows.net": "my_token"
            })
            hdfs_path, container_client = get_hdfs_path_and_container_client(uri_folder_path, spark=spark,
                                                                             credential=mock_token)
            assert hdfs_path == expected_hdfs_path
            # TODO need to do more sophisticated assertions for container_client
            assert container_client is not None

    @pytest.mark.parametrize(
        "azureml_path, datastore_type, protocol, expected_scheme",
        [
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", "https", "abfss"
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "AzureDataLakeGen2", "https", "abfss"
            ),
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", "http", "abfs"
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "AzureDataLakeGen2", "http", "abfs"
            )
        ]
    )
    def test_get_hdfs_path_and_container_client_with_azureml_uri(self, azureml_path, datastore_type, protocol,
                                                                 expected_scheme):
        """Test get_hdfs_path_and_container_client() with AzureML URI."""
        mock_datastore = Mock(datastore_type=datastore_type, protocol=protocol, endpoint="core.windows.net",
                              account_name="my_account", container_name="my_container")
        if datastore_type == "AzureBlob":
            mock_container_client = Mock(container_name="my_container")
            mock_blob_svc = Mock()
            mock_blob_svc.get_container_client.side_effect = \
                lambda n: mock_container_client if n == "my_container" else None
            mock_datastore.credential_type = "AccountKey"
            mock_datastore.blob_service = mock_blob_svc
        elif datastore_type == "AzureDataLakeGen2":
            mock_datastore.client_id = "my_client_id" if protocol == "https" else None
            mock_datastore.client_secret = "my_client_secret" if protocol == "https" else None
            mock_datastore.tenant_id = "00000" if protocol == "https" else None
        mock_ws = Mock(datastores={"my_datastore": mock_datastore})

        hdfs_path, container_client = get_hdfs_path_and_container_client(azureml_path, mock_ws)

        assert hdfs_path == f"{expected_scheme}://my_container@my_account.dfs.core.windows.net/path/to/folder"
        if datastore_type == "AzureBlob":
            assert container_client.container_name == "my_container"
            assert container_client == mock_container_client
        else:
            assert isinstance(container_client, FileSystemClient)

    @pytest.mark.parametrize(
        "input_path, expected_datastore",
        [
            (
                "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name"
                "/datastores/long_form_datastore/paths/path/to/file",
                "long_form_datastore"
            ),
            ("azureml://datastores/short_form_datastore/paths/path/to/file", "short_form_datastore"),
            ("file://path/to/folder", None),
        ]
    )
    def test_get_datastore_from_input_path(self, input_path, expected_datastore):
        """Test get_datastore_from_input_path()."""
        datastore = get_datastore_name_from_input_path(input_path)
        assert datastore == expected_datastore

    @pytest.mark.parametrize(
        "input_path",
        [
            "wasbs://my_container@my_account.blob.core.windows.net/path/to/file",
            "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
        ]
    )
    def test_get_datastore_from_input_path_throw_error(self, input_path):
        """Test get_datastore_from_input_path() with invalid input."""
        with pytest.raises(ValueError):
            get_datastore_name_from_input_path(input_path)

    def test_get_datastore_from_input_path_with_asset_path_throw(self):
        """Test get_datastore_from_input_path() with asset path."""
        with pytest.raises(ValueError):
            _ = get_datastore_name_from_input_path("azureml:my_asset:my_version")

    @pytest.mark.parametrize(
        "start_hour, end_hour, expected_hours, root_folder",
        [
            (7, 21, [7, 8, 13, 17, 19, 20], "/path/to/folder"),
            (4, 8, [6, 7], "/path/to/folder/"),
            (20, 23, [20, 21], ""),
            (13, 19, [13, 17], "/"),
            (3, 6, [], "/folder"),
            (22, 23, [], "/"),
            (9, 13, [], ""),
        ]
    )
    def test_get_file_list(self, start_hour, end_hour, expected_hours, root_folder):
        """Test get_file_list()."""
        non_empty_hours = [6, 7, 8, 13, 17, 19, 20, 21]
        non_empty_folders = [f"{root_folder.strip('/')}/2023/11/20/{h:02d}" for h in non_empty_hours]

        def _mock_list_blobs(name_starts_with=None):
            result_list = [f"{name_starts_with}/1.jsonl"] if name_starts_with in non_empty_folders else []
            return result_list.__iter__()

        def _mock_get_dir_client(path):
            mock_dir_client = Mock(path_name=path)
            mock_dir_client.exists.return_value = path in non_empty_folders
            return mock_dir_client

        for store_type in ["AzureBlob", "AzureDataLakeGen2"]:
            if store_type == "AzureBlob":
                mock_container_client = Mock(spec=ContainerClient, account_name="my_account",
                                             container_name="my_container")
                mock_container_client.list_blobs.side_effect = _mock_list_blobs
            else:
                mock_container_client = Mock(spec=FileSystemClient, account_name="my_account",
                                             container_name="my_container")
                mock_container_client.get_directory_client.side_effect = _mock_get_dir_client
            hdfs_uri_folder = f"abfss://my_container@my_account.dfs.core.windows.net{root_folder}"
            start = datetime(2023, 11, 20, start_hour)
            end = datetime(2023, 11, 20, end_hour)

            file_list = get_file_list(start, end, "uri_folder_path", hdfs_uri_folder, mock_container_client)

            assert file_list == [
                f"abfss://my_container@my_account.dfs.core.windows.net{root_folder.rstrip('/')}/2023/11/20/{h:02d}/*.jsonl"  # noqa
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

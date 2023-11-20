from unittest.mock import Mock
import pytest

from model_data_collector_preprocessor.mdc_preprocessor_helper import (
    convert_to_azureml_long_form, convert_to_hdfs_path, get_datastore_from_input_path
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
    def test_convert_to_hdfs_path(self, uri_folder_path: str, expected_hdfs_path: str):
        hdfs_path = convert_to_hdfs_path(uri_folder_path)
        assert hdfs_path == expected_hdfs_path

    @pytest.mark.parametrize(
        "azureml_path, datastore_type, protocol, expected_scheme",
        [
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "DatastoreType.AZURE_BLOB", "https", "abfss"
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "DatastoreType.AZURE_DATA_LAKE_GEN2", "https", "abfss"
            ),
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "DatastoreType.AZURE_BLOB", "http", "abfs"
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "DatastoreType.AZURE_DATA_LAKE_GEN2", "http", "abfs"
            )
        ]
    )
    def test_convert_to_hdfs_path_with_asset_uri(self, azureml_path, datastore_type, protocol, expected_scheme):
        mock_data_asset = Mock(datastore="my_datastore", path=azureml_path, type="uri_folder")
        mock_datastore = Mock(type=datastore_type, protocol=protocol, account_name="my_account",
                              container_name="my_container" if datastore_type == "DatastoreType.AZURE_BLOB" else None,
                              filesystem="my_container" if datastore_type == "DatastoreType.AZURE_DATA_LAKE_GEN2" else None)  # noqa: E501
        mock_ml_client = Mock()
        mock_ml_client.data.get.side_effect = lambda n, v: mock_data_asset if n == "my_asset" else None
        mock_ml_client.datastores.get.side_effect = lambda n: mock_datastore if n == "my_datastore" else None

        hdfs_path = convert_to_hdfs_path("azureml:my_asset:my_version", mock_ml_client)

        assert hdfs_path == f"{expected_scheme}://my_container@my_account.dfs.core.windows.net/path/to/folder"

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
        datastore = get_datastore_from_input_path(input_path)
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
            get_datastore_from_input_path(input_path)

    @pytest.mark.parametrize(
        "datastore, path, expected_datastore",
        [
            ("asset_datastore", "some_path", "asset_datastore"),
            (
                None,
                "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name"
                "/datastores/long_form_datastore/paths/path/to/folder",
                "long_form_datastore"
            ),
            (None, "azureml://datastores/short_form_datastore/paths/path/to/folder", "short_form_datastore"),
            # (None, "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder", "workspaceblobstore")
        ]
    )
    def test_get_datastore_from_input_path_with_asset_path(self, datastore, path, expected_datastore):
        """Test get_datastore_from_input_path() with asset path."""
        mock_data_asset = Mock(datastore=datastore, path=path)
        mock_ml_client = Mock()
        mock_ml_client.data.get.return_value = mock_data_asset

        datastore = get_datastore_from_input_path("azureml:my_asset:my_version", mock_ml_client)
        assert datastore == expected_datastore

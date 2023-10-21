# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""unit tests for mdc_preprocessor_helper"""

import pytest
from unittest.mock import Mock
from model_data_collector_preprocessor.mdc_preprocessor_helper import convert_to_azureml_long_form, get_datastore_from_input_path

@pytest.mark.parametrize(
    "url_str, converted",
    [
        ("https://my_account.blob.core.windows.net/my_container/path/to/file", True),
        ("wasbs://my_container@my_account.blob.core.windows.net/path/to/file", True),
        ("abfss://my_container@my_account.dfs.core.windows.net/path/to/file", True),
        ("azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name/datastores/my_datastore/paths/path/to/file", True),
        ("azureml://datastores/my_datastore/paths/path/to/file", True),
        ("azureml:my_asset:my_version", False),
        ("file://path/to/file", False),
        ("path/to/file", False),
    ]
)
def test_convert_to_azureml_long_form(url_str: str, converted: bool):
    converted_path = convert_to_azureml_long_form(url_str, "my_datastore", "my_sub_id", "my_rg_name", "my_ws_name")
    expected_path = "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name/datastores/my_datastore/paths/path/to/file" if converted else url_str
    assert converted_path == expected_path

@pytest.mark.parametrize(
    "input_path, expected_datastore",
    [
        ("azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name/datastores/long_form_datastore/paths/path/to/file", "long_form_datastore"),
        ("azureml://datastores/short_form_datastore/paths/path/to/file", "short_form_datastore"),
        ("wasbs://my_container@my_account.blob.core.windows.net/path/to/file", "workspaceblobstore"),
    ]
)
def test_get_datastore_from_input_path(input_path, expected_datastore):
    datastore = get_datastore_from_input_path(input_path)
    assert datastore == expected_datastore

@pytest.mark.parametrize(
    "datastore, path, expected_datastore",
    [
        ("asset_datastore", None, "asset_datastore"),
        (None, "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name/datastores/long_form_datastore/paths/path/to/folder", "long_form_datastore"),
        (None, "azureml://datastores/short_form_datastore/paths/path/to/folder", "short_form_datastore"),
        (None, "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder", "workspaceblobstore")
    ]
)
def test_get_datastore_from_input_path_with_asset_path(datastore, path, expected_datastore):
    mock_data_asset = Mock(datastore=datastore, path=path)
    mock_ml_client = Mock()
    mock_ml_client.data.get.return_value = mock_data_asset

    datastore = get_datastore_from_input_path("azureml:my_asset:my_version", mock_ml_client)
    assert datastore == expected_datastore
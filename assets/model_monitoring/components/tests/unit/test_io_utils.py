import pytest
from shared_utilities.io_utils import _verify_mltable_paths
from shared_utilities.momo_exceptions import InvalidInputError
from unittest.mock import Mock


@pytest.mark.unit
class TestIOUtils:
    """Test class for io_utils."""

    @pytest.mark.parametrize(
        "mltable_path",
        [
            {"file": "https://my_account.blob.core.windows.net/my_container/path/to/data.parquet"},
            {"folder": "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder"},
            {"pattern": "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder/*/*.jsonl"},
            {"file": "http://my_account.blob.core.windows.net/my_container/path/to/data.parquet"},
            {"folder": "wasb://my_container@my_account.blob.core.windows.net/path/to/folder"},
            {"pattern": "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder/*/*.jsonl"}
        ]
    )
    def test_verify_mltable_paths_error(self, mltable_path):
        """Test _verify_mltable_paths, negative cases."""
        mltable_dict = {
            "type": "mltable",
            "paths": [mltable_path]
        }
        with pytest.raises(InvalidInputError):
            _verify_mltable_paths("foo_path", mltable_dict=mltable_dict)

    def test_verify_mltable_paths_pass(self):
        """Test _verify_mltable_paths, for azureml paths, positive cases."""
        mltable_dict = {
            "type": "mltable",
            "paths": [
                {"file": "azureml://datastores/my_datastore/paths/path/to/data.parquet"},
                {"folder": "azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore/paths/path/to/folder"},  # noqa: E501
                {"pattern": "azureml://datastores/my_datastore/paths/path/to/folder/**/*.jsonl"},
                {"pattern": "./path/to/folder/*.csv"}
            ]
        }
        mock_datastore = Mock(datastore_type="AzureBlob", protocol="https", endpoint="core.windows.net",
                              account_name="my_account", container_name="my_container")
        mock_datastore.name = "my_datastore"
        mock_datastore.credential_type = "Sas"
        mock_datastore.sas_token = "my_sas_token"
        mock_ws = Mock(datastores={"my_datastore": mock_datastore})

        _verify_mltable_paths("foo_path", mock_ws, mltable_dict)

    @pytest.mark.parametrize(
        "mltable_path",
        [
            {"file": "azureml://datastores/my_datastore/paths/path/to/data.parquet"},
            {"folder": "azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore/paths/path/to/folder"},  # noqa: E501
            {"pattern": "azureml://datastores/my_datastore/paths/path/to/folder/**/*.jsonl"},
            {"file": "azureml:my_data:1"}
        ]
    )
    def test_verify_mltable_paths_azureml_path_error(self, mltable_path):
        """Test _verify_mltable_paths, for azureml paths, negative cases."""
        mock_datastore = Mock(datastore_type="AzureDataLakeGen2", protocol="https", endpoint="core.windows.net",
                              account_name="my_account", container_name="my_container")
        mock_datastore.name = "my_datastore"
        mock_datastore.tenant_id = None
        mock_ws = Mock(datastores={"my_datastore": mock_datastore})
        mltable_dict = {
            "type": "mltable",
            "paths": [mltable_path]
        }

        with pytest.raises(InvalidInputError):
            _verify_mltable_paths("foo_path", mock_ws, mltable_dict)
